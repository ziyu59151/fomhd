"""Read-only A0 diagnostics and hard acceptance gates."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

from . import core
from .types import Failure


TAU = 2.0 * math.pi


def _xp(context):
    return context.compute.xp


def _scalar(context, value):
    if isinstance(value, complex):
        value = value.real
    try:
        value = context.compute.scalar(value)
        return float(value.real if isinstance(value, complex) else value)
    except (AttributeError, TypeError):
        return float(context.compute.to_host(value).item())


def _as_device(context, value, dtype=None):
    return context.compute.xp.asarray(value, dtype=dtype)


def _size(value):
    return int(getattr(value, "size", len(value)))


def _norm_name(norm):
    if hasattr(norm, "value"):
        norm = norm.value
    text = str(norm).lower().replace("-", "_")
    if text in ("linf", "norm.linf"):
        return "linf"
    if text in ("weighted_l2", "norm.weighted_l2"):
        return "weighted_L2"
    if text in ("value", "norm.value"):
        return "value"
    return str(norm)


def _field(record, name, default=None):
    return getattr(record, name, default)


def _record(
    *,
    name,
    tag,
    unit="1",
    population="",
    component="",
    region="all",
    norm="value",
):
    return SimpleNamespace(
        name=name,
        population=population,
        component=component,
        producer="diagnostics",
        region=region,
        tag=tag,
        unit=unit,
        value_unit=unit,
        norm=norm,
        raw=0.0,
        reference=0.0,
        threshold=0.0,
        hard=True,
        available=True,
        passed=False,
        worst=0,
        selected=0,
        reason="",
    )


def budget(context, key):
    budgets = context.p.budgets
    try:
        return budgets[key]
    except (KeyError, TypeError):
        raise Failure("MissingBudget", f"missing diagnostic budget: {key}") from None


def _budget_key(name, population, norm):
    return name + (f".{population}" if population else "") + f".{_norm_name(norm)}"


def _region_distances(name, weights, context, component=""):
    geometry = context.geometry
    n = _size(weights)
    walls = getattr(geometry, "walls", ())
    if name.startswith("wall_"):
        return [0.0] * n, "1"
    if name in ("alias_B", "equilibrium_B") and component:
        table = {
            "R": getattr(geometry, "z_faces", None),
            "phi": getattr(geometry, "vertices", None),
            "Z": getattr(geometry, "r_faces", None),
        }.get(component)
        if table is None:
            raise Failure("DiagnosticComponent", component)
    else:
        table = None
    if table is None:
        m0 = getattr(context.op, "M0", None)
        m1 = getattr(context.op, "M1", None)
        mb = getattr(context.op, "MB", None)
        wv = getattr(geometry, "wv", None)
        if m0 is not None and n == _size(m0):
            distances = getattr(geometry, "cell_wall_distance", None)
            if distances is not None and _size(distances) == n:
                return list(distances), "m^3"
        if m1 is not None and n == _size(m1):
            parts = (
                getattr(geometry, "r_faces", None),
                getattr(geometry, "cells", None),
                getattr(geometry, "z_faces", None),
            )
        elif mb is not None and n == _size(mb):
            parts = (
                getattr(geometry, "z_faces", None),
                getattr(geometry, "vertices", None),
                getattr(geometry, "r_faces", None),
            )
        elif wv is not None and n == _size(wv):
            parts = (getattr(geometry, "vertices", None),)
        elif int(getattr(context.op, "Nc", 0)) and n == 3 * int(context.op.Nc):
            distances = getattr(geometry, "cell_wall_distance", None)
            if distances is None:
                distances = getattr(geometry, "cell_wall_distances", None)
            if distances is not None:
                return list(distances) * 3, "m^3"
            parts = ()
        else:
            raise Failure(
                "DiagnosticLocation", f"unknown spatial weights for {name}"
            )
        if parts:
            distances = []
            for part in parts:
                if part is None:
                    raise Failure("DiagnosticLocation", name)
                distances.extend(_entity_wall_distances(part, context))
            if len(distances) == n:
                return distances, "m^3"
            raise Failure("DiagnosticShape", "region weights mismatch")
    if table is None:
        raise Failure("DiagnosticLocation", f"unknown spatial weights for {name}")
    distances = _entity_wall_distances(table, context)
    if len(distances) != n:
        raise Failure("DiagnosticShape", "entity wall distances are unavailable")
    return distances, "m^3"


def _point_segment_distance(point_r, point_z, start_r, start_z, end_r, end_z):
    dr = end_r - start_r
    dz = end_z - start_z
    denominator = dr * dr + dz * dz
    if denominator == 0.0:
        x = point_r - start_r
        z = point_z - start_z
    else:
        fraction = (
            (point_r - start_r) * dr + (point_z - start_z) * dz
        ) / denominator
        fraction = min(1.0, max(0.0, fraction))
        x = point_r - (start_r + fraction * dr)
        z = point_z - (start_z + fraction * dz)
    return math.hypot(x, z)


def _entity_wall_distances(table, context):
    geometry = context.geometry
    wall_vertices = geometry.vertices
    walls = getattr(geometry, "walls", ())
    distances = []
    for entity in range(_size(table.R)):
        point_r = float(table.R[entity])
        point_z = float(table.Z[entity])
        distance = math.inf
        for wall in walls:
            first, second = wall.vertices
            distance = min(
                distance,
                _point_segment_distance(
                    point_r,
                    point_z,
                    float(wall_vertices.R[first]),
                    float(wall_vertices.Z[first]),
                    float(wall_vertices.R[second]),
                    float(wall_vertices.Z[second]),
                ),
            )
        distances.append(distance)
    return distances


def measure(
    out,
    name,
    residual,
    weights,
    unit,
    tag,
    context,
    population="",
    samples=False,
    component="",
):
    """Measure one residual using physical geometry weights."""
    xp = _xp(context)
    nw = _size(weights)
    nr = _size(residual)
    walls = getattr(context.geometry, "walls", ())
    t_rows = int(getattr(getattr(context.op, "T_full", None), "shape", (0, 0))[0])
    tsel_rows = int(getattr(getattr(context.op, "T", None), "shape", (0, 0))[0])
    if nr == 0 and nw == 0 and name.startswith("wall_") and not walls:
        if t_rows or tsel_rows:
            raise Failure("DiagnosticShape", name)
        for region in ("all", "inner", "wall"):
            for norm in ("linf", "weighted_L2"):
                b = budget(context, _budget_key(name, population, norm))
                if b.unit != unit or _norm_name(b.norm) != norm:
                    raise Failure("BudgetUnit", f"wrong unit/norm for {name}")
                record = _record(
                    name=name,
                    tag=tag,
                    unit=unit,
                    population=population,
                    component=component,
                    region=region,
                    norm=norm,
                )
                record.available = False
                record.hard = False
                record.reason = "not applicable: no boundary entities"
                out.append(record)
        return
    if not nr or not nw or nr % nw:
        raise Failure("DiagnosticShape", name)
    distances, measure_unit = _region_distances(
        name, weights, context, component
    )
    if len(distances) != nw:
        raise Failure("DiagnosticShape", "region weights mismatch")
    w = _as_device(context, weights, xp.float64).reshape(-1)
    r = _as_device(context, residual).reshape(-1)
    if not bool(xp.all(xp.isfinite(w))) or not bool(xp.all(w > 0)):
        raise Failure("DiagnosticWeight", name)
    reps = nr // nw
    ww = xp.tile(w, reps)
    d = xp.tile(_as_device(context, distances, xp.float64), reps)
    inside = d >= float(context.p.inner_distance)
    for region in ("all", "inner", "wall"):
        if region == "inner":
            mask = inside
        elif region == "wall":
            mask = ~inside
        else:
            mask = xp.ones(nr, dtype=xp.bool_)
        selected_count = int(xp.count_nonzero(mask))
        for norm in ("linf", "weighted_L2"):
            b = budget(context, _budget_key(name, population, norm))
            if b.unit != unit or _norm_name(b.norm) != norm:
                raise Failure("BudgetUnit", f"wrong unit/norm for {name}")
            record = _record(
                name=name,
                tag=tag,
                unit=unit,
                population=population,
                component=component,
                region=region,
                norm=norm,
            )
            record.hard = bool(_field(b, "hard", True))
            record.value_unit = (
                f"{unit} * sqrt({measure_unit})"
                if norm == "weighted_L2" and measure_unit != "1"
                else unit
            )
            record.selected = selected_count
            if not selected_count:
                record.available = False
                record.hard = False
                record.reason = "empty region"
                out.append(record)
                continue
            selected = xp.where(mask, xp.abs(r), 0.0)
            finite = xp.isfinite(r) | ~mask
            if not bool(xp.all(finite)):
                record.available = False
                record.reason = "nonfinite residual"
                bad = xp.where(~finite, xp.arange(nr), nr)
                record.worst = int(xp.min(bad))
                out.append(record)
                continue
            if norm == "linf":
                record.raw = _scalar(context, xp.max(selected))
                record.reference = float(b.reference)
            else:
                weighted = xp.where(mask, xp.sqrt(ww) * xp.abs(r), 0.0)
                angular = TAU / context.N if samples else TAU
                record.raw = _scalar(
                    context, xp.sqrt(xp.sum(weighted * weighted) * angular)
                )
                volume = _scalar(
                    context, xp.sum(xp.where(mask, ww, xp.zeros_like(ww)))
                )
                modes = 1 if samples else nr // nw
                record.reference = float(
                    b.reference * math.sqrt(angular * volume / modes)
                )
            record.threshold = float(b.atol + b.rtol * record.reference)
            record.available = math.isfinite(record.raw)
            record.passed = record.available and record.raw <= record.threshold
            record.worst = int(xp.argmax(selected))
            out.append(record)


def scalar_check(out, name, raw, unit, tag, context, population=""):
    b = budget(context, _budget_key(name, population, "value"))
    if b.unit != unit or _norm_name(b.norm) != "value":
        raise Failure("BudgetUnit", f"wrong scalar unit for {name}")
    value = _scalar(context, raw)
    record = _record(
        name=name, tag=tag, unit=unit, population=population, norm="value"
    )
    record.hard = bool(_field(b, "hard", True))
    record.raw = value
    record.reference = float(b.reference)
    record.threshold = float(b.atol + b.rtol * b.reference)
    record.available = math.isfinite(value)
    record.passed = record.available and abs(value) <= record.threshold
    out.append(record)


def _check_identity(record, key):
    return (
        record.name,
        record.population,
        record.component,
        record.producer,
        record.region,
        _norm_name(record.norm),
        record.tag.stage,
        record.tag.step,
        record.tag.generation,
        record.tag.trial,
        record.tag.spectral_context,
        float(record.tag.time).hex(),
        int(getattr(record.tag, "micro_step", -1)),
        int(getattr(record.tag, "iteration", 0)),
        int(getattr(record.tag, "slab", 0)),
    )


def gate(records, required=None):
    seen = set()
    names = set()
    for record in records:
        key = _check_identity(record, record)
        if key in seen:
            raise Failure("DuplicateDiagnostic", "|".join(map(str, key)))
        seen.add(key)
        names.add(record.name)
        if record.hard and (not record.available or not record.passed):
            failure = Failure(
                "HardGate",
                f"hard check {record.name}/{record.population}/"
                f"{_norm_name(record.norm)} = {record.raw}, "
                f"threshold {record.threshold}",
            )
            failure.checks = list(records)
            raise failure
    for name in required or ():
        if name not in names:
            raise Failure("MissingDiagnostic", name)


def gate_records(records, required):
    gate(records)
    present = {_check_identity(record, record) for record in records}
    for record in required:
        if _check_identity(record, record) not in present:
            raise Failure(
                "MissingDiagnostic",
                f"{record.name}|{record.population}|{record.component}|"
                f"{record.producer}|{record.region}|"
                f"{_norm_name(record.norm)}|{record.tag.stage}",
            )


def stage_requirements(context, tag):
    if int(getattr(context.p, "electron_inertia", 0)) == 0:
        from .massless_checks import stage_requirements as massless_requirements

        return massless_requirements(context, tag)
    out = []

    def scalar(name, population=""):
        record = _record(name=name, tag=tag, population=population)
        out.append(record)

    def array(name, population=""):
        for region in ("all", "inner", "wall"):
            for norm in ("linf", "weighted_L2"):
                out.append(
                    _record(
                        name=name,
                        tag=tag,
                        population=population,
                        region=region,
                        norm=norm,
                    )
                )

    for name in (
        "finite",
        "hermitian_rhs",
        "source_identity",
        "memory_bound",
        "minimum_B",
    ):
        scalar(name)
    for name in (
        "poisson_linear",
        "darwin_dynamic",
        "longitudinal_projection",
        "gauss",
        "transverse_divergence",
        "current_divB",
        "current_ampere",
        "CT_increment",
        "current_response",
    ):
        array(name)
    for a in range(2):
        population = context.p.species[a].id
        for name in (
            "minimum_n",
            "minimum_face_n",
            "minimum_perp",
            "electric_pairing",
            "pressure_pairing",
            "magnetic_work",
            "wall_work",
            "convection_pairing",
            "spectral_transfer_defect",
        ):
            scalar(name, population)
        for name in (
            "stage_continuity",
            "wall_state",
            "wall_rhs",
            "wall_multiplier",
            "theta_identity",
            "wall_number_flux",
            "wall_perp_flux",
        ):
            array(name, population)
        if a:
            scalar("minimum_parallel", population)
            array("cgl_source", population)
            array("wall_parallel_flux", population)
    return out


def _finite(context, *values):
    xp = _xp(context)
    return all(bool(xp.all(xp.isfinite(v))) for v in values if v is not None)


def _max_abs(context, values):
    xp = _xp(context)
    if not _size(values):
        return 0.0
    return _scalar(context, xp.max(xp.abs(_as_device(context, values))))


def _hermitian_defect(context, values, entities):
    xp = _xp(context)
    a = _as_device(context, values)
    k = int(context.p.K)
    defect = xp.max(xp.abs(xp.imag(a[k * entities : (k + 1) * entities])))
    for mode in range(1, k + 1):
        negative = a[(k - mode) * entities : (k - mode + 1) * entities]
        positive = a[(k + mode) * entities : (k + mode + 1) * entities]
        defect = xp.maximum(defect, xp.max(xp.abs(negative - xp.conj(positive))))
    return _scalar(context, defect)


def _sub(context, left, right):
    return _as_device(context, left) - _as_device(context, right)


def _add(context, *values):
    xp = _xp(context)
    out = _as_device(context, values[0]).copy()
    for value in values[1:]:
        out = xp.add(out, _as_device(context, value))
    return out


def _mul(context, value, factor):
    return _as_device(context, value) * factor


def _mode_apply(context, matrices, values):
    return core.mode_apply(matrices, values, context.compute)


def _sample_apply(context, matrix, values, samples=None):
    return core.sample_apply(
        matrix, values, context.N if samples is None else samples, context.compute
    )


def _inverse(context, values, entities, physical=True, reference=0.0):
    return core.inverse(values, entities, context, physical, reference)


def _forward(context, values, entities):
    return core.forward(values, entities, context)


def state_totals(state, result, context):
    if int(getattr(context.p, "electron_inertia", 0)) == 0:
        cached = getattr(result, "totals", None)
        if cached is not None:
            return cached
        from .massless_checks import state_totals as massless_totals

        return massless_totals(state, result, context)
    xp = _xp(context)
    number = [0.0, 0.0]
    kinetic = [0.0, 0.0]
    ion_internal = 0.0
    electron_perp = 0.0
    electron_parallel = 0.0
    nc = int(context.op.Nc)
    m0 = _as_device(context, context.op.M0)
    w = TAU * m0 / context.N
    for a in range(2):
        fluid = result.fluid[a]
        n = _as_device(context, fluid.n)
        mom = _as_device(context, fluid.mom)
        perp = _as_device(context, fluid.perp)
        number[a] = _scalar(
            context, xp.sum(w * xp.real(n.reshape(context.N, nc)), axis=1).sum()
        )
        kinetic[a] = _scalar(
            context,
            xp.sum(
                w
                * xp.sum(
                    xp.abs(mom.reshape(context.N, 3, nc)) ** 2,
                    axis=1,
                )
                / (2.0 * context.p.species[a].mass * xp.real(n.reshape(context.N, nc))),
            ),
        )
        if a == 0:
            ion_internal = _scalar(
                context,
                xp.sum(
                    w
                    * xp.real(perp.reshape(context.N, nc))
                    / (context.p.species[0].gamma - 1.0)
                ),
            )
        else:
            electron_perp = _scalar(
                context, xp.sum(w * xp.real(perp.reshape(context.N, nc)))
            )
            electron_parallel = _scalar(
                context,
                xp.sum(
                    w
                    * xp.real(fluid.parallel.reshape(context.N, nc))
                    / 2.0
                ),
            )
    mb = _as_device(context, context.op.MB)
    magnetic = _scalar(
        context,
        xp.sum(
            TAU
            * xp.tile(mb, _size(state.B) // _size(mb))
            * xp.abs(_as_device(context, state.B)) ** 2
            / (2.0 * context.p.mu0)
        ),
    )
    m1 = _as_device(context, context.op.M1)
    longitudinal = _scalar(
        context,
        xp.sum(
            TAU
            * xp.tile(m1, _size(result.cache.EL) // _size(m1))
            * xp.abs(_as_device(context, result.cache.EL)) ** 2
            * context.p.epsilon0
            / 2.0
        ),
    )
    energy = (
        kinetic[0]
        + kinetic[1]
        + ion_internal
        + electron_perp
        + electron_parallel
        + magnetic
        + longitudinal
    )
    b0 = _as_device(context, state.B)[
        int(context.p.K) * _size(mb) : (int(context.p.K) + 1) * _size(mb)
    ]
    topology_row = _as_device(context, context.op.topology_flux_row).reshape(-1)
    topology = _scalar(context, xp.sum(topology_row * b0))
    nz = len(context.geometry.z_faces.keys)
    physical_flux = _scalar(
        context,
        xp.sum(
            _as_device(context, context.geometry.av)
            * xp.real(b0[nz : nz + _size(context.geometry.av)])
        ),
    )
    return SimpleNamespace(
        number=number,
        kinetic=kinetic,
        ion_internal=ion_internal,
        electron_perp=electron_perp,
        electron_parallel=electron_parallel,
        magnetic=magnetic,
        longitudinal=longitudinal,
        energy=energy,
        topology=topology,
        physical_flux=physical_flux,
    )


def check_source(source, context):
    """Validate a test-only prescribed source in rate layout."""
    xp = _xp(context)
    nc = int(context.op.Nc)
    modes = 2 * int(context.p.K) + 1
    expected = (
        (source.fluid[0].n, modes * nc, "source density"),
        (source.fluid[0].mom, modes * 3 * nc, "source ion momentum"),
        (source.fluid[0].perp, modes * nc, "source ion pressure"),
        (source.fluid[1].n, modes * nc, "source electron density"),
        (source.fluid[1].mom, modes * 3 * nc, "source electron momentum"),
        (source.fluid[1].perp, modes * nc, "source electron pressure"),
        (source.fluid[1].parallel, modes * nc, "source electron parallel pressure"),
        (source.B, modes * _size(context.op.MB), "source magnetic field"),
    )
    for value, count, name in expected:
        if _size(value) != count or not _finite(context, value):
            raise Failure("ManufacturedSource", f"{name} is invalid")
    checks = (
        (source.fluid[0].n, nc, context.p.ref.ni / context.p.ref.time),
        (
            source.fluid[0].mom,
            3 * nc,
            context.p.species[0].mass
            * context.p.ref.ni
            * context.p.ref.velocity
            / context.p.ref.time,
        ),
        (source.fluid[0].perp, nc, context.p.ref.pressure / context.p.ref.time),
        (source.fluid[1].n, nc, context.p.ref.ne / context.p.ref.time),
        (
            source.fluid[1].mom,
            3 * nc,
            context.p.species[1].mass
            * context.p.ref.ne
            * context.p.ref.velocity
            / context.p.ref.time,
        ),
        (source.fluid[1].perp, nc, context.p.ref.pressure / context.p.ref.time),
        (
            source.fluid[1].parallel,
            nc,
            context.p.ref.pressure / context.p.ref.time,
        ),
        (
            source.B,
            _size(context.op.MB),
            context.p.ref.magnetic / context.p.ref.time,
        ),
    )
    for value, entities, scale in checks:
        limit = 512.0 * sys.float_info.epsilon * max(
            _max_abs(context, value), scale, 1e-300
        )
        if _hermitian_defect(context, value, entities) > limit:
            raise Failure(
                "ManufacturedSource", "source retained spectrum is not Hermitian"
            )
    for a in range(2):
        momentum = _as_device(context, source.fluid[a].mom).reshape(
            2 * int(context.p.K) + 1, 3 * nc
        )
        reference = (
            context.p.species[a].mass
            * (context.p.ref.ne if a else context.p.ref.ni)
            * context.p.ref.velocity
            / context.p.ref.time
        )
        for mode in range(2 * int(context.p.K) + 1):
            limit = 2048.0 * sys.float_info.epsilon * max(
                _max_abs(context, momentum[mode]), reference, 1e-300
            )
            if _max_abs(
                context,
                _as_device(context, context.op.T_full) @ momentum[mode],
            ) > limit:
                raise Failure(
                    "ManufacturedSource",
                    "source momentum violates the wall trace",
                )
    reference = (
        context.p.ref.magnetic
        / (context.p.ref.length * context.p.ref.time)
    )
    if not math.isfinite(reference) or reference <= 0.0:
        raise Failure("ManufacturedSource", "invalid physical divergence scale")
    bound = 1e-9 * reference
    divergence = _mode_apply(context, context.op.DB, source.B)
    if _max_abs(context, divergence) > bound:
        raise Failure(
            "ManufacturedSource", "magnetic source must preserve divergence"
        )
    return True


def source_rates(source, midpoint, stage, context):
    """Return integrated-rate metadata for a prescribed source stage."""
    from .fields import electrostatic

    xp = _xp(context)
    nc = int(context.op.Nc)
    ne = _size(context.op.M1)
    mb = _size(context.op.MB)
    total_number = [0.0, 0.0]
    total_kinetic = [0.0, 0.0]
    ion_internal = 0.0
    electron_perp = 0.0
    electron_parallel = 0.0
    rho = xp.zeros((2 * int(context.p.K) + 1) * nc, dtype=xp.complex128)
    m0 = _as_device(context, context.op.M0)
    angular_weight = TAU * m0 / context.N
    inverse_momentum = []
    for a in range(2):
        source_species = source.fluid[a]
        rate = stage.fluid[a]
        number = _inverse(context, source_species.n, nc)
        momentum = _inverse(context, source_species.mom, 3 * nc)
        pressure = _inverse(context, source_species.perp, nc)
        parallel = (
            _inverse(context, source_species.parallel, nc) if a else None
        )
        number_samples = number.reshape(context.N, nc)
        momentum_samples = momentum.reshape(context.N, 3, nc)
        inverse_momentum.append(momentum_samples)
        rho = rho + _as_device(context, source_species.n) * context.p.species[a].charge
        total_number[a] = _scalar(
            context,
            xp.sum(angular_weight[None, :] * xp.real(number_samples)),
        )
        velocity = _as_device(context, rate.u).reshape(context.N, 3, nc)
        work = xp.sum(
            xp.real(xp.conj(velocity) * momentum_samples), axis=1
        )
        speed2 = xp.sum(xp.abs(velocity) ** 2, axis=1)
        kinetic_density = work - (
            0.5
            * context.p.species[a].mass
            * speed2
            * xp.real(number_samples)
        )
        total_kinetic[a] = _scalar(
            context, xp.sum(angular_weight[None, :] * kinetic_density)
        )
        pressure_samples = pressure.reshape(context.N, nc)
        if a:
            electron_perp = _scalar(
                context,
                xp.sum(angular_weight[None, :] * xp.real(pressure_samples)),
            )
            electron_parallel = _scalar(
                context,
                xp.sum(
                    angular_weight[None, :]
                    * xp.real(parallel.reshape(context.N, nc))
                    / 2.0
                ),
            )
        else:
            ion_internal = _scalar(
                context,
                xp.sum(
                    angular_weight[None, :]
                    * xp.real(pressure_samples)
                    / (context.p.species[a].gamma - 1.0)
                ),
            )
    field = electrostatic(rho, stage.tag, context)
    mb_weights = _as_device(context, context.op.MB)
    magnetic = _scalar(
        context,
        TAU
        * xp.sum(
            xp.tile(mb_weights, _size(source.B) // mb)
            * xp.real(
                xp.conj(_as_device(context, midpoint.B))
                * _as_device(context, source.B)
            )
        )
        / context.p.mu0,
    )
    m1_weights = _as_device(context, context.op.M1)
    longitudinal_energy = _scalar(
        context,
        TAU
        * context.p.epsilon0
        * xp.sum(
            xp.tile(m1_weights, _size(field.EL) // ne)
            * xp.real(
                xp.conj(_as_device(context, stage.cache.EL))
                * _as_device(context, field.EL)
            )
        ),
    )
    topology = _scalar(
        context,
        xp.sum(
            _as_device(context, context.op.topology_flux_row).reshape(-1)
            * _as_device(context, source.B)[
                int(context.p.K) * mb : (int(context.p.K) + 1) * mb
            ]
        ),
    )
    nz = len(context.geometry.z_faces.keys)
    physical_flux = _scalar(
        context,
        xp.sum(
            _as_device(context, context.geometry.av)
            * xp.real(
                _as_device(context, source.B)[
                    int(context.p.K) * mb
                    + nz : int(context.p.K) * mb
                    + nz
                    + _size(context.geometry.av)
                ]
            )
        ),
    )
    total_energy = (
        total_kinetic[0]
        + total_kinetic[1]
        + ion_internal
        + electron_perp
        + electron_parallel
        + magnetic
        + longitudinal_energy
    )
    electric = _sample_apply(
        context,
        context.op.H,
        _inverse(
            context,
            stage.cache.EL,
            ne,
            True,
            context.p.ref.electric,
        ),
    ).reshape(context.N, 3, nc)
    electric_rate = _sample_apply(
        context,
        context.op.H,
        _inverse(
            context,
            field.EL,
            ne,
            True,
            context.p.ref.electric / context.p.ref.time,
        ),
    ).reshape(context.N, 3, nc)
    magnetic_rate = _inverse(
        context,
        _sample_apply(
            context,
            context.op.QB,
            source.B,
            2 * int(context.p.K) + 1,
        ),
        3 * nc,
        True,
        context.p.ref.magnetic / context.p.ref.time,
    ).reshape(context.N, 3, nc)
    stage_magnetic = _as_device(context, stage.BN).reshape(context.N, 3, nc)
    samples = max(context.N, 3)
    momentum_value = [0.0, 0.0, 0.0]
    angular = 0.0
    source_momentum = inverse_momentum[0] + inverse_momentum[1]
    rate = [
        xp.real(source_momentum[:, 0])
        + context.p.epsilon0
        * (
            xp.real(electric_rate[:, 1] * stage_magnetic[:, 2]
                    - electric_rate[:, 2] * stage_magnetic[:, 1]
                    + electric[:, 1] * magnetic_rate[:, 2]
                    - electric[:, 2] * magnetic_rate[:, 1])
        ),
        xp.real(source_momentum[:, 1])
        + context.p.epsilon0
        * (
            xp.real(electric_rate[:, 2] * stage_magnetic[:, 0]
                    - electric_rate[:, 0] * stage_magnetic[:, 2]
                    + electric[:, 2] * magnetic_rate[:, 0]
                    - electric[:, 0] * magnetic_rate[:, 2])
        ),
        xp.real(source_momentum[:, 2])
        + context.p.epsilon0
        * (
            xp.real(electric_rate[:, 0] * stage_magnetic[:, 1]
                    - electric_rate[:, 1] * stage_magnetic[:, 0]
                    + electric[:, 0] * magnetic_rate[:, 1]
                    - electric[:, 1] * magnetic_rate[:, 0])
        ),
    ]
    radius = _as_device(context, context.geometry.cells.R).reshape(nc)
    for q in range(samples):
        sample = 0 if context.N == 1 else q
        phi = TAU * q / samples
        weights = TAU * m0 / samples
        cos_phi, sin_phi = math.cos(phi), math.sin(phi)
        momentum_value[0] += _scalar(
            context,
            xp.sum(weights * (rate[0][sample] * cos_phi - rate[1][sample] * sin_phi)),
        )
        momentum_value[1] += _scalar(
            context,
            xp.sum(weights * (rate[0][sample] * sin_phi + rate[1][sample] * cos_phi)),
        )
        momentum_value[2] += _scalar(
            context, xp.sum(weights * rate[2][sample])
        )
        angular += _scalar(
            context, xp.sum(weights * radius * rate[1][sample])
        )
    totals = SimpleNamespace(
        number=total_number,
        kinetic=total_kinetic,
        ion_internal=ion_internal,
        electron_perp=electron_perp,
        electron_parallel=electron_parallel,
        magnetic=magnetic,
        longitudinal=longitudinal_energy,
        energy=total_energy,
        topology=topology,
        physical_flux=physical_flux,
    )
    return SimpleNamespace(
        totals=totals,
        momentum=SimpleNamespace(value=momentum_value, angular=angular),
        longitudinal_rate=field.EL,
    )


def boundary_ledger(state, result, context):
    if int(getattr(context.p, "electron_inertia", 0)) == 0:
        cached = getattr(result, "boundary", None)
        if cached is not None:
            return cached
        from .massless_checks import boundary_ledger as massless_boundary

        return massless_boundary(state, result, context)
    xp = _xp(context)
    geometry = context.geometry
    nc = int(context.op.Nc)
    nr = len(geometry.r_faces.keys)
    nz = len(geometry.z_faces.keys)
    nv = len(geometry.vertices.keys)
    ne = _size(context.op.M1)
    nb = _size(context.op.MB)
    bs = _inverse(context, state.B, nb)
    el = _inverse(
        context, result.cache.EL, ne, True, context.p.ref.electric
    )
    et = _inverse(
        context, result.cache.ET, ne, True, context.p.ref.electric
    )
    walls = geometry.walls
    traces = []
    trace_data = []
    number = [0.0, 0.0]
    kinetic = [0.0, 0.0]
    internal_perp = [0.0, 0.0]
    internal_parallel = [0.0, 0.0]
    field_energy = 0.0
    total_energy = 0.0
    bs_samples = _as_device(context, bs).reshape(context.N, nb)
    el_samples = _as_device(context, el).reshape(context.N, ne)
    et_samples = _as_device(context, et).reshape(context.N, ne)
    i_matrix = _as_device(context, context.op.I)
    for wall in walls:
        radial = wall.component == "r_face"
        edge = wall.face if radial else nr + nc + wall.face
        direction = 0 if radial else 2
        sigma = wall.sign
        bvectors = [xp.zeros(context.N, dtype=xp.float64) for _ in range(3)]
        bvectors[1] = 0.5 * (
            xp.real(bs_samples[:, nz + wall.vertices[0]])
            + xp.real(bs_samples[:, nz + wall.vertices[1]])
        )
        if radial:
            bvectors[2] = xp.real(
                bs_samples[:, nz + nv + wall.face]
            )
        else:
            bvectors[0] = xp.real(bs_samples[:, wall.face])
        bmag = xp.sqrt(sum(value * value for value in bvectors))
        if not bool(xp.all(bmag > context.p.B_min)):
            raise Failure(
                "WallMagnetization",
                "wall magnetic magnitude below declared floor",
            )
        row = i_matrix[edge].reshape(3, nc)
        normal_row = xp.real(row[direction])
        trace_data.append(
            SimpleNamespace(
                wall=wall,
                direction=direction,
                sigma=sigma,
                B=bvectors,
                bmag=bmag,
                normal_row=normal_row,
            )
        )
        area = wall.area_per_radian * TAU / context.N
        for a in range(2):
            fluid = result.fluid[a]
            species = context.p.species[a]
            density = xp.real(
                _as_device(context, fluid.n).reshape(
                    context.N, nc
                ) @ normal_row
            )
            if not bool(xp.all(density > species.face_min)):
                raise Failure(
                    "WallDensity", "wall trace below density floor"
                )
            momentum = _as_device(context, fluid.mom).reshape(
                context.N, 3, nc
            )
            velocity = xp.stack(
                [
                    xp.real(momentum[:, d, :] @ normal_row)
                    / (species.mass * density)
                    for d in range(3)
                ],
                axis=1,
            )
            perp = xp.real(
                _as_device(context, fluid.perp).reshape(
                    context.N, nc
                ) @ normal_row
            )
            parallel = (
                xp.real(
                    _as_device(context, fluid.parallel).reshape(
                        context.N, nc
                    ) @ normal_row
                )
                if a
                else perp
            )
            flux = sigma * xp.real(
                _as_device(context, fluid.flux).reshape(context.N, ne)[
                    :, edge
                ]
            )
            un = sigma * velocity[:, direction]
            kinetic_density = (
                species.mass * density * xp.sum(velocity**2, axis=1) / 2.0
            )
            internal = (
                perp + parallel / 2
                if a
                else perp / (species.gamma - 1.0)
            )
            pressure_work = perp * un
            if a:
                ub = sum(
                    velocity[:, d] * bvectors[d] / bmag
                    for d in range(3)
                )
                pressure_work += (
                    (parallel - perp)
                    * ub
                    * sigma
                    * bvectors[direction]
                    / bmag
                )
            pressure_flux = sigma * xp.real(
                _as_device(context, fluid.pressure_flux).reshape(
                    context.N, ne
                )[:, edge]
            )
            number[a] += _scalar(context, xp.sum(area * flux))
            kinetic[a] += _scalar(
                context, xp.sum(area * kinetic_density * un)
            )
            internal_perp[a] += _scalar(
                context,
                xp.sum(
                    area
                    * pressure_flux
                    / (1.0 if a else species.gamma - 1.0)
                ),
            )
            if a:
                parallel_flux = sigma * xp.real(
                    _as_device(context, fluid.parallel_flux).reshape(
                        context.N, ne
                    )[:, edge]
                )
                internal_parallel[a] += _scalar(
                    context, xp.sum(area * parallel_flux / 2.0)
                )
            total_energy += _scalar(
                context,
                xp.sum(
                    area
                    * ((kinetic_density + internal) * un + pressure_work)
                ),
            )
            appended = trace_data[-1]
            if not hasattr(appended, "species"):
                appended.species = []
            appended.species.append(
                SimpleNamespace(
                    density=density,
                    velocity=velocity,
                    perp=perp,
                    parallel=parallel,
                    normal_flux=flux,
                )
            )
        el_normal = xp.real(el_samples[:, edge])
        et_normal = xp.real(et_samples[:, edge])
        # The prescribed staircase trace has only the normal electric
        # component, so its normal E-cross-B contribution is identically zero.
        field = 0.0
        field_energy += field
        total_energy += field
        trace_data[-1].EL_normal = el_normal
        trace_data[-1].ET_normal = et_normal
    for sample in range(context.N):
        for data in trace_data:
            sigma = data.sigma
            trace = SimpleNamespace(
                B=[
                    value[sample]
                    for value in data.B
                ],
                EL=[0.0, 0.0, 0.0],
                ET=[0.0, 0.0, 0.0],
                velocity=[[0.0] * 3 for _ in range(2)],
                density=[0.0, 0.0],
                normal_flux=[0.0, 0.0],
                perp=[0.0, 0.0],
                parallel=[0.0, 0.0],
            )
            trace.EL[data.direction] = data.EL_normal[sample]
            trace.ET[data.direction] = data.ET_normal[sample]
            for a, species_data in enumerate(data.species):
                trace.density[a] = species_data.density[sample]
                trace.perp[a] = species_data.perp[sample]
                trace.parallel[a] = species_data.parallel[sample]
                trace.normal_flux[a] = species_data.normal_flux[sample]
                trace.velocity[a] = [
                    species_data.velocity[sample, d]
                    for d in range(3)
                ]
            traces.append(trace)
    return SimpleNamespace(
        traces=traces,
        number=number,
        kinetic=kinetic,
        internal_perp=internal_perp,
        internal_parallel=internal_parallel,
        field_energy=field_energy,
        total_energy=total_energy,
        trace_arrays=trace_data,
    )


def momentum(state, result, context):
    if int(getattr(context.p, "electron_inertia", 0)) == 0:
        cached = getattr(result, "momentum", None)
        if cached is not None:
            return cached
        from .massless_checks import momentum as massless_momentum

        return massless_momentum(state, result, context)
    xp = _xp(context)
    geometry = context.geometry
    nc = int(context.op.Nc)
    el_node = _sample_apply(
        context,
        context.op.H,
        _inverse(
            context,
            result.cache.EL,
            _size(context.op.M1),
            True,
            context.p.ref.electric,
        ),
    ).reshape(context.N, 3, nc)
    samples = max(context.N, 3)
    out = SimpleNamespace(
        value=[0.0, 0.0, 0.0],
        outward=[0.0, 0.0, 0.0],
        wall_reaction=[0.0, 0.0, 0.0],
        angular=0.0,
        outward_torque=0.0,
        wall_torque=0.0,
        angular_samples=samples,
    )
    m0 = _as_device(context, context.op.M0)
    magnetic = _as_device(context, result.BN).reshape(context.N, 3, nc)
    total_momentum = (
        _as_device(context, result.fluid[0].mom).reshape(context.N, 3, nc)
        + _as_device(context, result.fluid[1].mom).reshape(context.N, 3, nc)
    )
    total_momentum[:, 0] += context.p.epsilon0 * (
        el_node[:, 1] * magnetic[:, 2]
        - el_node[:, 2] * magnetic[:, 1]
    )
    total_momentum[:, 1] += context.p.epsilon0 * (
        el_node[:, 2] * magnetic[:, 0]
        - el_node[:, 0] * magnetic[:, 2]
    )
    total_momentum[:, 2] += context.p.epsilon0 * (
        el_node[:, 0] * magnetic[:, 1]
        - el_node[:, 1] * magnetic[:, 0]
    )
    wall_reaction = (
        _as_device(context, result.terms[0].reaction).reshape(
            context.N, 3, nc
        )
        + _as_device(context, result.terms[1].reaction).reshape(
            context.N, 3, nc
        )
    )
    radius = _as_device(context, geometry.cells.R).reshape(nc)
    wall_tractions = []
    for data in result.boundary.trace_arrays:
        wall = data.wall
        direction = data.direction
        sigma = wall.sign
        b = data.B
        el = [xp.zeros_like(data.EL_normal) for _ in range(3)]
        et = [xp.zeros_like(data.ET_normal) for _ in range(3)]
        el[direction] = data.EL_normal
        et[direction] = data.ET_normal
        density = [species.density for species in data.species]
        velocity = [
            [species.velocity[:, d] for d in range(3)]
            for species in data.species
        ]
        perp = [species.perp for species in data.species]
        parallel = [species.parallel for species in data.species]
        b2 = sum(value * value for value in b)
        el2 = sum(value * value for value in el)
        elt = sum(el[d] * et[d] for d in range(3))
        traction = []
        for d in range(3):
            unit = 1.0 if d == direction else 0.0
            stress = (
                b[d] * b[direction] - 0.5 * b2 * unit
            ) / context.p.mu0 + context.p.epsilon0 * (
                el[d] * el[direction]
                - 0.5 * el2 * unit
                + el[d] * et[direction]
                + et[d] * el[direction]
                - elt * unit
            )
            component = -sigma * stress
            for a in range(2):
                component += (
                    context.p.species[a].mass
                    * density[a]
                    * velocity[a][d]
                    * sigma
                    * velocity[a][direction]
                    + sigma
                    * (
                        perp[a] * unit
                        + (parallel[a] - perp[a])
                        * b[d]
                        * b[direction]
                        / b2
                    )
                )
            traction.append(component)
        wall_tractions.append((wall, direction, traction))
    for q in range(samples):
        sample = 0 if context.N == 1 else q
        phi = TAU * q / samples
        cos_phi, sin_phi = math.cos(phi), math.sin(phi)
        weights = TAU * m0 / samples
        vector = xp.real(total_momentum[sample])
        reaction = xp.real(wall_reaction[sample])
        cart = (
            vector[0] * cos_phi - vector[1] * sin_phi,
            vector[0] * sin_phi + vector[1] * cos_phi,
            vector[2],
        )
        reaction_cart = (
            reaction[0] * cos_phi - reaction[1] * sin_phi,
            reaction[0] * sin_phi + reaction[1] * cos_phi,
            reaction[2],
        )
        for d in range(3):
            out.value[d] += _scalar(
                context, xp.sum(weights * cart[d])
            )
            out.wall_reaction[d] += _scalar(
                context, xp.sum(weights * reaction_cart[d])
            )
        out.angular += _scalar(
            context, xp.sum(weights * radius * vector[1])
        )
        out.wall_torque += _scalar(
            context, xp.sum(weights * radius * reaction[1])
        )
        for wall, direction, traction in wall_tractions:
            cart = (
                traction[0] * cos_phi - traction[1] * sin_phi,
                traction[0] * sin_phi + traction[1] * cos_phi,
                traction[2],
            )
            weight = TAU * wall.area_per_radian / samples
            for d in range(3):
                out.outward[d] += _scalar(
                    context, weight * cart[d][sample]
                )
            surface_radius = (
                geometry.r_faces.R[wall.face]
                if direction == 0
                else geometry.z_faces.R[wall.face]
            )
            out.outward_torque += _scalar(
                context,
                weight * float(surface_radius) * traction[1][sample],
            )
    return out


def _minimum(out, name, values, floor, unit, population=""):
    record = _record(
        name=name, tag=out_tag(out), unit=unit, population=population
    )
    record.threshold = float(floor)
    record.reference = float(floor)
    record.selected = _size(values)
    record.raw = math.inf
    record.available = bool(_size(values))
    record.passed = False
    if record.available:
        record.raw = float(values.min().real)
        record.passed = math.isfinite(record.raw) and record.raw > floor
        record.worst = int(values.argmin())
    out.append(record)


def out_tag(out):
    return out[-1].tag if out else SimpleNamespace()


def _truth(out, name, ok, tag):
    record = _record(name=name, tag=tag)
    record.raw = 0.0 if ok else 1.0
    record.reference = 1.0
    record.threshold = 0.0
    record.passed = bool(ok)
    out.append(record)


def _physical(
    out,
    tag,
    context,
    values,
    entities,
    reference,
    component,
    unit,
    population="",
):
    record = _record(
        name="hermitian_rhs_component",
        tag=tag,
        unit=unit,
        component=component,
        population=population,
    )
    record.available = _finite(context, values)
    record.reference = float(reference)
    record.selected = _size(values)
    if record.available:
        record.raw = _hermitian_defect(context, values, entities)
        record.threshold = 512.0 * sys.float_info.epsilon * max(
            _max_abs(context, values), reference
        )
    record.passed = record.available and record.raw <= record.threshold
    out.append(record)
    return record.passed


def check_stage(state, result, context):
    if int(getattr(context.p, "electron_inertia", 0)) == 0:
        from .massless_checks import check_stage as massless_check

        return massless_check(state, result, context)
    xp = _xp(context)
    out = result.checks
    tag = result.tag
    nc = int(context.op.Nc)
    finite = _finite(
        context,
        state.B,
        result.rhs.B,
        result.cache.E,
        result.cache.phi,
        result.current,
    )
    hermitian = True
    for a in range(2):
        rhs = result.rhs.fluid[a]
        values = state.fluid[a]
        finite = finite and _finite(
            context,
            values.n,
            values.mom,
            values.perp,
            values.parallel,
            rhs.n,
            rhs.mom,
            rhs.perp,
            rhs.parallel,
        )
        population = context.p.species[a].id
        density_ref = (
            context.p.ref.ne if a else context.p.ref.ni
        )
        hermitian &= _physical(
            out,
            tag,
            context,
            rhs.n,
            nc,
            density_ref / context.p.ref.time,
            "n",
            "1/(m^3*s)",
            population,
        )
        hermitian &= _physical(
            out,
            tag,
            context,
            rhs.mom,
            3 * nc,
            context.p.species[a].mass
            * density_ref
            * context.p.ref.velocity
            / context.p.ref.time,
            "momentum",
            "kg/(m^2*s^2)",
            population,
        )
        hermitian &= _physical(
            out,
            tag,
            context,
            rhs.perp,
            nc,
            context.p.ref.pressure / context.p.ref.time,
            "perp",
            "Pa/s",
            population,
        )
        if a:
            hermitian &= _physical(
                out,
                tag,
                context,
                rhs.parallel,
                nc,
                context.p.ref.pressure / context.p.ref.time,
                "parallel",
                "Pa/s",
                population,
            )
    hermitian &= _physical(
        out,
        tag,
        context,
        result.rhs.B,
        _size(context.op.MB),
        context.p.ref.magnetic / context.p.ref.time,
        "B",
        "T/s",
    )
    _truth(out, "finite", finite, tag)
    _truth(out, "hermitian_rhs", hermitian, tag)
    ownership = (
        context.p.species[0].id != context.p.species[1].id
        and getattr(result, "source_id", 0) != 0
        and getattr(result.cache, "tag", None) == tag
        and getattr(result.fluid[0].wall, "tag", None) == tag
        and getattr(result.fluid[1].wall, "tag", None) == tag
        and state.step == tag.step
        and state.generation == tag.generation
        and state.time == tag.time
    )
    _truth(out, "source_identity", ownership, tag)
    memory = int(getattr(context.p, "memory_limit", 0))
    _truth(
        out,
        "memory_bound",
        memory > 0 and core.estimated_peak_bytes(context) <= memory,
        tag,
    )
    bn = _as_device(context, result.BN)
    bn_samples = bn.reshape(context.N, 3, nc)
    magnetic = xp.sqrt(xp.sum(xp.real(bn_samples) ** 2, axis=1)).reshape(-1)
    _minimum(out, "minimum_B", magnetic, context.p.B_min, "T")

    def remember(name, residual, weights, unit, population="", samples=False):
        measure(
            out,
            name,
            residual,
            weights,
            unit,
            tag,
            context,
            population,
            samples,
        )
        result.residuals[name + (f".{population}" if population else "")] = residual

    remember(
        "poisson_linear",
        result.cache.poisson_residual,
        context.op.M0,
        "V/m^2",
    )
    remember(
        "darwin_dynamic",
        result.residuals["darwin_dynamic"],
        context.op.M1,
        "A/(m^2*s)",
    )
    remember(
        "longitudinal_projection",
        result.residuals["longitudinal_projection"],
        context.op.M0,
        "A/m^3",
    )
    remember(
        "gauss",
        _sub(
            context,
            _mode_apply(context, context.op.DJ, result.cache.EL),
            _mul(context, result.rho, 1.0 / context.p.epsilon0),
        ),
        context.op.M0,
        "V/m^2",
    )
    remember(
        "transverse_divergence",
        _mode_apply(context, context.op.DJ, result.cache.ET),
        context.op.M0,
        "V/m^2",
    )
    remember(
        "current_divB",
        _mode_apply(context, context.op.DB, state.B),
        context.geometry.wv,
        "T/m",
    )
    remember(
        "current_ampere",
        result.ampere,
        context.op.M1,
        "T/m",
    )
    remember(
        "CT_increment",
        _mode_apply(context, context.op.DB, result.rhs.B),
        context.geometry.wv,
        "T/(m*s)",
    )
    actual = xp.zeros(_size(result.current), dtype=xp.complex128)
    t_full = _as_device(context, context.op.T_full)
    for a in range(2):
        fluid = result.fluid[a]
        terms = result.terms[a]
        species = context.p.species[a]
        ledger = result.ledger[a]
        population = species.id
        _minimum(out, "minimum_n", _as_device(context, fluid.n).real,
                 species.n_min, "1/m^3", population)
        _minimum(out, "minimum_face_n", _as_device(context, fluid.nf).real,
                 species.face_min, "1/m^3", population)
        _minimum(out, "minimum_perp", _as_device(context, fluid.perp).real,
                 species.p_min, "Pa", population)
        if a:
            _minimum(
                out,
                "minimum_parallel",
                _as_device(context, fluid.parallel).real,
                species.parallel_min,
                "Pa",
                population,
            )
        remember(
            "stage_continuity",
            _add(context, result.rhs.fluid[a].n, fluid.divergence_hat),
            context.op.M0,
            "1/(m^3*s)",
            population,
        )
        state_wall = _sample_apply(
            context,
            t_full,
            _inverse(context, state.fluid[a].mom, 3 * nc),
        )
        remember(
            "wall_state",
            state_wall,
            xp.ones(int(context.op.T_full.shape[0]), dtype=xp.float64),
            "kg/(m^2*s)",
            population,
            True,
        )
        remember(
            "wall_rhs",
            _sample_apply(context, t_full, terms.force),
            xp.ones(int(context.op.T_full.shape[0]), dtype=xp.float64),
            "kg/(m^2*s^2)",
            population,
            True,
        )
        remember(
            "wall_multiplier",
            terms.multiplier_residual,
            xp.ones(int(context.op.T.shape[0]), dtype=xp.float64),
            "kg/(m^2*s^2)",
            population,
            True,
        )
        walls = context.geometry.walls
        wall_n = xp.zeros(context.N * len(walls), dtype=xp.complex128)
        wall_perp = xp.zeros_like(wall_n)
        wall_parallel = xp.zeros_like(wall_n)
        for sample in range(context.N):
            for wall_index, wall in enumerate(walls):
                edge = (
                    wall.face
                    if wall.component == "r_face"
                    else len(context.geometry.r_faces.keys) + nc + wall.face
                )
                index = sample * len(walls) + wall_index
                mode_index = sample * _size(context.op.M1) + edge
                wall_n[index] = wall.sign * _as_device(context, fluid.flux)[mode_index]
                wall_perp[index] = (
                    wall.sign
                    * _as_device(context, fluid.pressure_flux)[mode_index]
                )
                if a:
                    wall_parallel[index] = (
                        wall.sign
                        * _as_device(context, fluid.parallel_flux)[mode_index]
                    )
        wall_weights = xp.ones(len(walls), dtype=xp.float64)
        remember(
            "wall_number_flux",
            wall_n,
            wall_weights,
            "1/(m^2*s)",
            population,
            True,
        )
        remember(
            "wall_perp_flux",
            wall_perp,
            wall_weights,
            "Pa*m/s",
            population,
            True,
        )
        if a:
            remember(
                "wall_parallel_flux",
                wall_parallel,
                wall_weights,
                "Pa*m/s",
                population,
                True,
            )
        scalar_check(
            out,
            "electric_pairing",
            ledger.electric_node - ledger.electric_edge,
            "W",
            tag,
            context,
            population,
        )
        scalar_check(
            out,
            "pressure_pairing",
            ledger.pressure_kinetic + ledger.pressure_internal,
            "W",
            tag,
            context,
            population,
        )
        for name in (
            "magnetic_work",
            "wall_work",
            "convection_pairing",
        ):
            scalar_check(
                out,
                name,
                getattr(ledger, name.replace("_work", "").replace("_pairing", ""), 0.0),
                "W",
                tag,
                context,
                population,
            )
        scalar_check(
            out,
            "spectral_transfer_defect",
            ledger.spectral_kinetic
            + ledger.spectral_perp
            + ledger.spectral_parallel,
            "W",
            tag,
            context,
            population,
        )
        nu = _as_device(context, fluid.u).reshape(context.N, 3, nc)
        densities = _as_device(context, fluid.n).reshape(context.N, nc)
        velocity_flux = _sample_apply(
            context,
            context.op.I,
            (nu * densities[:, None, :]).reshape(-1),
        )
        nf = _as_device(context, fluid.nf).reshape(-1)
        vu = velocity_flux / nf
        from .fluid import divergence

        theta = divergence(vu, context)
        remember(
            "theta_identity",
            _sub(context, fluid.theta, theta),
            context.op.M0,
            "1/s",
            population,
            True,
        )
        if a:
            tensor = _as_device(context, fluid.tensor).reshape(context.N, 9, nc)
            gradient = _as_device(context, fluid.gradient).reshape(
                context.N, 9, nc
            )
            pq = xp.sum(tensor * gradient, axis=1).reshape(-1)
            residual = (
                _as_device(context, fluid.perp)
                * (_as_device(context, fluid.strain) - _as_device(context, fluid.theta))
                - _as_device(context, fluid.parallel)
                * _as_device(context, fluid.strain)
                + pq
            )
            remember(
                "cgl_source",
                residual,
                context.op.M0,
                "Pa/s",
                population,
                True,
            )
        fluxdot = _sample_apply(
            context,
            context.op.I,
            result.rhs.fluid[a].mom,
            2 * int(context.p.K) + 1,
        )
        actual += fluxdot * (species.charge / species.mass)
    expected = _add(
        context,
        result.fJ0,
        _as_device(context, result.AE) @ _as_device(context, result.cache.E),
    )
    remember(
        "current_response",
        _sub(context, actual, expected),
        context.op.M1,
        "A/(m^2*s)",
    )
    gate_records(out, stage_requirements(context, tag))


def _normalise_quadrature(context, quadrature, midpoint_state, midpoint):
    if quadrature is None:
        return [(midpoint_state, midpoint, float(context.p.dt))]
    out = []
    for item in quadrature:
        if isinstance(item, dict):
            state, stage, weight = item["state"], item["stage"], item["weight"]
        elif hasattr(item, "stage") and hasattr(item, "weight"):
            state, stage, weight = item.state, item.stage, item.weight
        else:
            state, stage, weight = item
        weight = float(weight)
        if not math.isfinite(weight):
            raise Failure("Time", "nonfinite quadrature weight")
        out.append((state, stage, weight))
    if not out:
        raise Failure("Time", "empty quadrature")
    return out


def _weighted_boundary(context, quadrature):
    keys = ("number", "kinetic", "internal_perp", "internal_parallel")
    out = {key: [0.0, 0.0] for key in keys}
    out["field_energy"] = 0.0
    out["total_energy"] = 0.0
    for _, stage, weight in quadrature:
        for key in keys:
            for a in range(2):
                out[key][a] += weight * float(getattr(stage.boundary, key)[a])
        out["field_energy"] += weight * stage.boundary.field_energy
        out["total_energy"] += weight * stage.boundary.total_energy
    return SimpleNamespace(**out)


def _weighted_ledger(stage, weight):
    values = {}
    for name in (
        "electric_node",
        "electric_edge",
        "pressure_kinetic",
        "pressure_internal",
        "magnetic",
        "wall",
        "convection",
        "perp_rate",
        "parallel_rate",
        "spectral_kinetic",
        "spectral_perp",
        "spectral_parallel",
    ):
        values[name] = weight * float(getattr(stage.ledger, name))
    return SimpleNamespace(**values)


def _weighted_momentum(context, quadrature):
    out = SimpleNamespace(
        value=[0.0, 0.0, 0.0],
        outward=[0.0, 0.0, 0.0],
        wall_reaction=[0.0, 0.0, 0.0],
        angular=0.0,
        outward_torque=0.0,
        wall_torque=0.0,
        angular_samples=max(context.N, 3),
    )
    for _, stage, weight in quadrature:
        for name in ("value", "outward", "wall_reaction"):
            for d in range(3):
                out.__dict__[name][d] += weight * float(
                    getattr(stage.momentum, name)[d]
                )
        out.angular += weight * stage.momentum.angular
        out.outward_torque += weight * stage.momentum.outward_torque
        out.wall_torque += weight * stage.momentum.wall_torque
    return out


def _integrated_power_to_rate(integrated_energy, dt):
    """Convert an accepted interval energy (J) to its average power (W)."""
    dt = float(dt)
    value = float(integrated_energy)
    if not math.isfinite(dt) or dt <= 0.0 or not math.isfinite(value):
        raise Failure("DiagnosticPower", "invalid integrated field power")
    return value / dt


def check_step(
    left,
    base,
    midpoint_state,
    midpoint,
    right,
    end,
    cumulative,
    context,
    prescribed_source=None,
    *,
    quadrature=None,
):
    """Build and hard-gate all accepted-interval identities."""
    if int(getattr(context.p, "electron_inertia", 0)) == 0:
        from .massless_checks import check_step as massless_check_step

        return massless_check_step(
            left,
            base,
            midpoint_state,
            midpoint,
            right,
            end,
            cumulative,
            context,
            prescribed_source,
            quadrature=quadrature,
        )
    xp = _xp(context)
    quadrature = _normalise_quadrature(
        context, quadrature, midpoint_state, midpoint
    )
    dt = float(context.p.dt)
    if abs(sum(item[2] for item in quadrature) - dt) > (
        64 * sys.float_info.epsilon * max(abs(dt), 1.0)
    ):
        raise Failure("Time", "quadrature weights do not span dt")
    interval_tag = SimpleNamespace(**vars(end.tag))
    interval_tag.stage = "interval"
    out = []
    state_total_left = base.totals
    state_total_right = end.totals
    source_rates = None
    if prescribed_source is not None:
        source_stages = list(prescribed_source)
        source_rates = _weighted_source_rates(
            context, quadrature, source_stages
        )
    else:
        source_stages = None
        source_rates = SimpleNamespace(
            totals=SimpleNamespace(
                number=[0.0, 0.0],
                kinetic=[0.0, 0.0],
                ion_internal=0.0,
                electron_perp=0.0,
                electron_parallel=0.0,
                magnetic=0.0,
                longitudinal=0.0,
                energy=0.0,
                topology=0.0,
            ),
            momentum=SimpleNamespace(
                value=[0.0, 0.0, 0.0], angular=0.0
            ),
            longitudinal_rate=xp.zeros_like(
                _as_device(context, midpoint.cache.EL)
            ),
        )

    interval = SimpleNamespace(
        source_id=getattr(midpoint, "source_id", 0),
        trial=midpoint.tag.trial,
        left_step=left.step,
        right_step=right.step,
        weight=dt,
        left_time=left.time,
        right_time=right.time,
        number_flux=[None, None],
        current=[None, None],
        boundary_charge_integral=None,
        quadrature=[
            SimpleNamespace(
                source_id=getattr(stage, "source_id", 0),
                weight=weight,
                stage=stage.tag.stage,
            )
            for _, stage, weight in quadrature
        ],
        accepted=False,
    )

    def scalar(name, value, unit, population=""):
        scalar_check(out, name, value, unit, interval_tag, context, population)

    def measure_interval(name, residual, weights, unit, population=""):
        measure(
            out,
            name,
            residual,
            weights,
            unit,
            interval_tag,
            context,
            population,
        )

    charge_change = xp.zeros(_size(base.rho), dtype=xp.complex128)
    flux_div = xp.zeros_like(charge_change)
    integrated_divergence = None
    integrated_flux = None
    integrated_number_flux = None
    boundary_charge_integral = [0.0, 0.0]
    for a in range(2):
        species = context.p.species[a]
        integrated_flux = sum(
            weight * _as_device(context, stage.fluid[a].flux_hat)
            for _, stage, weight in quadrature
        )
        integrated_divergence = sum(
            weight * _as_device(context, stage.fluid[a].divergence_hat)
            for _, stage, weight in quadrature
        )
        integrated_number_flux = integrated_flux.copy()
        interval.number_flux[a] = integrated_number_flux
        interval.current[a] = integrated_number_flux * species.charge
        ndiff = _sub(context, right.fluid[a].n, left.fluid[a].n) / dt
        source_n = None
        if prescribed_source is not None:
            source_n = sum(
                weight * _as_device(context, source.fluid[a].n)
                for (_, _, weight), source in zip(
                    quadrature, source_stages
                )
            ) / dt
            ndiff = ndiff - source_n
        measure_interval(
            "species_continuity",
            ndiff + integrated_divergence / dt,
            context.op.M0,
            "1/(m^3*s)",
            species.id,
        )
        charge_change += species.charge * ndiff
        flux_div += species.charge * (integrated_divergence / dt)
        boundary_number = sum(
            weight * stage.boundary.number[a] for _, stage, weight in quadrature
        )
        boundary_charge_integral[a] = species.charge * boundary_number
        scalar(
            "total_number_balance",
            state_total_right.number[a]
            - state_total_left.number[a]
            + boundary_number
            - source_rates.totals.number[a],
            "1",
            species.id,
        )
        boundary_kinetic = sum(
            weight * stage.boundary.kinetic[a] for _, stage, weight in quadrature
        )
        ledger = _sum_ledgers(quadrature, a)
        scalar(
            "kinetic_energy",
            state_total_right.kinetic[a]
            - state_total_left.kinetic[a]
            + boundary_kinetic
            - (
                ledger.electric_edge
                + ledger.pressure_kinetic
                + ledger.wall
                + ledger.magnetic
                + source_rates.totals.kinetic[a]
            ),
            "J",
            species.id,
        )
    interval.boundary_charge_integral = boundary_charge_integral
    measure_interval(
        "RC_flux", charge_change + flux_div, context.op.M0, "C/(m^3*s)"
    )
    integrated_current = sum(
        weight * _as_device(context, stage.current)
        for _, stage, weight in quadrature
    )
    average_current = integrated_current / dt
    dj_j = _mode_apply(context, context.op.DJ, average_current)
    measure_interval("RC_mom", charge_change + dj_j, context.op.M0, "C/(m^3*s)")
    measure_interval("deltaC", dj_j - flux_div, context.op.M0, "C/(m^3*s)")
    el_rate = _sub(context, end.cache.EL, base.cache.EL) / dt
    if prescribed_source is not None:
        el_rate = el_rate - source_rates.longitudinal_rate / dt
    from .fields import longitudinal

    measure_interval(
        "RL",
        longitudinal(average_current, context)
        + el_rate * context.p.epsilon0,
        context.op.M1,
        "A/m^2",
    )
    bchange = _sub(context, right.B, left.B)
    if prescribed_source is not None:
        bchange = bchange - sum(
            weight * _as_device(context, source.B)
            for (_, _, weight), source in zip(quadrature, source_stages)
        )
    ce = sum(
        weight * _mode_apply(context, context.op.C, stage.cache.ET)
        for _, stage, weight in quadrature
    )
    measure_interval(
        "CT_update", bchange / dt + ce / dt, context.op.MB, "T/s"
    )
    measure_interval(
        "divB_update",
        _mode_apply(context, context.op.DB, bchange),
        context.geometry.wv,
        "T/m",
    )
    measure_interval(
        "long_term_divB_change",
        _mode_apply(context, context.op.DB, right.B) - cumulative.baseline.divB,
        context.geometry.wv,
        "T/m",
    )
    topology_row = _as_device(
        context, context.op.topology_flux_row
    ).reshape(-1)
    midpoint = int(context.p.K) * _size(context.op.MB)
    stop = midpoint + _size(context.op.MB)
    topology_change = _scalar(
        context,
        xp.sum(
            topology_row
            * (_as_device(context, right.B)[midpoint:stop]
               - _as_device(context, left.B)[midpoint:stop])
        ),
    )
    scalar(
        "topology_increment",
        topology_change - source_rates.totals.topology,
        "Wb",
    )
    charge = 0.0
    for a in range(2):
        charge += context.p.species[a].charge * (
            state_total_right.number[a] - state_total_left.number[a]
            + sum(
                weight * stage.boundary.number[a]
                for _, stage, weight in quadrature
            )
            - source_rates.totals.number[a]
        )
    scalar("total_charge_balance", charge, "C")
    boundary = _weighted_boundary(context, quadrature)
    scalar(
        "ion_internal",
        state_total_right.ion_internal
        - state_total_left.ion_internal
        + boundary.internal_perp[0]
        - (
            _sum_ledgers(quadrature, 0).perp_rate
            + source_rates.totals.ion_internal
        ),
        "J",
    )
    scalar(
        "electron_perp",
        state_total_right.electron_perp
        - state_total_left.electron_perp
        + boundary.internal_perp[1]
        - (
            _sum_ledgers(quadrature, 1).perp_rate
            + source_rates.totals.electron_perp
        ),
        "J",
    )
    scalar(
        "electron_parallel",
        state_total_right.electron_parallel
        - state_total_left.electron_parallel
        + boundary.internal_parallel[1]
        - (
            _sum_ledgers(quadrature, 1).parallel_rate
            + source_rates.totals.electron_parallel
        ),
        "J",
    )
    pb = 0.0
    pl = 0.0
    for state, stage, weight in quadrature:
        mb = _as_device(context, context.op.MB)
        pb += weight * _scalar(
            context,
            TAU
            * xp.sum(
                xp.tile(mb, _size(state.B) // _size(mb))
                * xp.real(
                    xp.conj(_as_device(context, state.B))
                    * _as_device(context, stage.rhs.B)
                )
            )
            / context.p.mu0,
        )
        m1 = _as_device(context, context.op.M1)
        pl += weight * _scalar(
            context,
            TAU
            * context.p.epsilon0
            * xp.sum(
                xp.tile(m1, _size(stage.cache.EL) // _size(m1))
                * xp.real(
                    xp.conj(_as_device(context, stage.cache.EL))
                    * _as_device(context, stage.ELdot)
                )
            ),
        )
    scalar(
        "field_energy_B",
        state_total_right.magnetic
        - state_total_left.magnetic
        - pb
        - source_rates.totals.magnetic,
        "J",
    )
    scalar(
        "field_energy_L",
        state_total_right.longitudinal
        - state_total_left.longitudinal
        - pl
        - source_rates.totals.longitudinal,
        "J",
    )
    scalar(
        "field_exchange",
        _integrated_power_to_rate(
            pb
            + pl
            + _sum_ledgers(quadrature, 0).electric_edge
            + _sum_ledgers(quadrature, 1).electric_edge
            + boundary.field_energy,
            dt,
        ),
        "W",
    )
    scalar(
        "total_energy",
        state_total_right.energy
        - state_total_left.energy
        + boundary.total_energy
        - source_rates.totals.energy,
        "J",
    )
    integrated_momentum = _weighted_momentum(context, quadrature)
    for d, axis in enumerate("xyz"):
        scalar(
            f"momentum_{axis}",
            end.momentum.value[d]
            - base.momentum.value[d]
            + integrated_momentum.outward[d]
            - source_rates.momentum.value[d],
            "kg*m/s",
        )
    scalar(
        "angular_momentum",
        end.momentum.angular
        - base.momentum.angular
        + integrated_momentum.outward_torque
        - source_rates.momentum.angular,
        "kg*m^2/s",
    )
    result = SimpleNamespace(
        checks=out,
        totals=state_total_right,
        momentum=end.momentum,
        interval=interval,
    )
    gate(
        out,
        {
            "species_continuity",
            "RC_flux",
            "RC_mom",
            "deltaC",
            "RL",
            "CT_update",
            "divB_update",
            "long_term_divB_change",
            "topology_increment",
            "total_charge_balance",
            "total_number_balance",
            "kinetic_energy",
            "ion_internal",
            "electron_perp",
            "electron_parallel",
            "field_energy_B",
            "field_energy_L",
            "field_exchange",
            "total_energy",
            "momentum_x",
            "momentum_y",
            "momentum_z",
            "angular_momentum",
        },
    )
    return result


def _sum_ledgers(quadrature, species):
    values = {}
    for name in (
        "electric_node",
        "electric_edge",
        "pressure_kinetic",
        "pressure_internal",
        "magnetic",
        "wall",
        "convection",
        "perp_rate",
        "parallel_rate",
        "spectral_kinetic",
        "spectral_perp",
        "spectral_parallel",
    ):
        values[name] = sum(
            weight * float(getattr(stage.ledger[species], name))
            for _, stage, weight in quadrature
        )
    return SimpleNamespace(**values)


def _weighted_source_rates(context, quadrature, sources):
    """Integrate the optional test-only prescribed-source diagnostic rates."""
    if len(sources) != len(quadrature):
        raise Failure("ManufacturedSource", "source stage count mismatch")
    xp = _xp(context)
    totals = SimpleNamespace(
        number=[0.0, 0.0],
        kinetic=[0.0, 0.0],
        ion_internal=0.0,
        electron_perp=0.0,
        electron_parallel=0.0,
        magnetic=0.0,
        longitudinal=0.0,
        energy=0.0,
        topology=0.0,
        physical_flux=0.0,
    )
    momentum = SimpleNamespace(value=[0.0, 0.0, 0.0], angular=0.0)
    longitudinal = None
    for (state, stage, weight), source in zip(quadrature, sources):
        check_source(source, context)
        rate = source_rates(source, state, stage, context)
        if longitudinal is None:
            longitudinal = (
                _as_device(context, rate.longitudinal_rate) * weight
            )
        else:
            longitudinal += (
                _as_device(context, rate.longitudinal_rate) * weight
            )
        for a in range(2):
            totals.number[a] += weight * rate.totals.number[a]
            totals.kinetic[a] += weight * rate.totals.kinetic[a]
        for name in (
            "ion_internal",
            "electron_perp",
            "electron_parallel",
            "magnetic",
            "longitudinal",
            "energy",
            "topology",
            "physical_flux",
        ):
            setattr(
                totals,
                name,
                getattr(totals, name) + weight * getattr(rate.totals, name),
            )
        for d in range(3):
            momentum.value[d] += weight * rate.momentum.value[d]
        momentum.angular += weight * rate.momentum.angular
    return SimpleNamespace(
        totals=totals,
        momentum=momentum,
        longitudinal_rate=longitudinal
        if longitudinal is not None
        else xp.zeros(
            (2 * int(context.p.K) + 1) * _size(context.op.M1),
            dtype=xp.complex128,
        ),
    )
