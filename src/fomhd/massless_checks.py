"""Read-only diagnostics for the massless quasineutral electron branch."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np

from . import core, diagnostics
from .box_boundary import wall_magnetic_trace
from .core import forward, inverse, mode_apply, sample_apply
from .fluid import power_ledger
from .massless import MODEL_NAME
from .types import Failure


def _xp(context):
    return context.compute.xp


def _modes(context):
    return 2 * context.p.K + 1


def _integrate_cells(values, components, context):
    xp = _xp(context)
    nc = context.op.Nc
    if values.size != context.N * components * nc:
        raise Failure("Shape", "cell diagnostic shape mismatch")
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    return (
        xp.sum(
            xp.asarray(values).reshape(context.N, components, nc)
            * m0[None, None, :]
        )
        * (2.0 * math.pi / context.N)
    )


def _integrate_edges(values, context):
    xp = _xp(context)
    if values.size != context.N * context.op.Ne:
        raise Failure("Shape", "edge diagnostic shape mismatch")
    m1 = xp.real(xp.asarray(context.op.M1)).astype(xp.float64)
    return (
        xp.sum(
            xp.asarray(values).reshape(context.N, context.op.Ne)
            * m1[None, :]
        )
        * (2.0 * math.pi / context.N)
    )


def _physical_real(value, name):
    value = complex(value)
    if abs(value.imag) > (
        4096.0 * sys.float_info.epsilon * max(abs(value), 1.0e-300)
    ):
        raise Failure("Hermitian", f"{name} has a nonzero imaginary physical value")
    return value.real


def _bound_real(value, name):
    """Convert a physical bound and reject invalid reduced scalars."""
    value = complex(value)
    if not (math.isfinite(value.real) and math.isfinite(value.imag)) or value.real < 0.0:
        raise Failure(
            "WallWork",
            f"{name} is not a finite nonnegative error bound: {value!r}",
        )
    return _physical_real(value, name)


def applicability():
    """Return explicit model-applicability metadata for JSON consumers."""
    return SimpleNamespace(
        model=MODEL_NAME,
        electron_inertia="not_applicable",
        electron_kinetic_energy="not_applicable",
        electron_kinetic_transport="not_applicable",
        electron_mechanical_stress="not_applicable",
        finite_charge_electric_stress="not_applicable",
        longitudinal_electric_energy="not_applicable",
        finite_charge_gauss="not_applicable",
        spectral_kinetic_transfer="not_applicable",
    )


def _applicability():
    return applicability()


def state_totals(state, result, context):
    """Return the massless total energy, excluding omitted electron DOFs."""
    xp = _xp(context)
    nc = context.op.Nc
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    cell_weight = (2.0 * math.pi / context.N) * m0
    number = [0.0, 0.0]
    kinetic = [0.0, 0.0]
    for species in range(2):
        fluid = result.fluid[species]
        density = xp.real(xp.asarray(fluid.n)).reshape(context.N, nc)
        number[species] = float(
            (xp.sum(cell_weight[None, :] * density)).item()
        )
    ion = result.fluid[0]
    ion_density = xp.real(xp.asarray(ion.n)).reshape(context.N, nc)
    ion_velocity = xp.asarray(ion.u).reshape(context.N, 3, nc)
    kinetic[0] = float(
        (
            xp.sum(
                cell_weight[None, :]
                * 0.5
                * context.p.species[0].mass
                * ion_density
                * xp.sum(xp.abs(ion_velocity) ** 2, axis=1)
            )
        ).item()
    )
    ion_internal = float(
        (
            xp.sum(
                cell_weight[None, :]
                * xp.real(xp.asarray(ion.perp)).reshape(context.N, nc)
                / (context.p.species[0].gamma - 1.0)
            )
        ).item()
    )
    electron_perp = float(
        (
            xp.sum(
                cell_weight[None, :]
                * xp.real(xp.asarray(result.fluid[1].perp)).reshape(
                    context.N, nc
                )
            )
        ).item()
    )
    electron_parallel = float(
        (
            xp.sum(
                cell_weight[None, :]
                * xp.real(xp.asarray(result.fluid[1].parallel)).reshape(
                    context.N, nc
                )
                / 2.0
            )
        ).item()
    )
    mb = xp.real(xp.asarray(context.op.MB)).astype(xp.float64)
    magnetic = float(
        (
            xp.sum(
                2.0
                * math.pi
                * xp.tile(mb, state.B.size // mb.size)
                * xp.abs(xp.asarray(state.B)) ** 2
                / (2.0 * context.p.mu0)
            )
        ).item()
    )
    b0_start = context.p.K * mb.size
    b0 = xp.asarray(state.B)[b0_start : b0_start + mb.size]
    topology_row = xp.asarray(context.op.topology_flux_row).reshape(-1)
    topology_value = xp.sum(topology_row * b0).item()
    topology = float(complex(topology_value).real)
    nz = len(context.geometry.z_faces.keys)
    physical_flux = float(
        xp.sum(
            xp.asarray(context.geometry.av)
            * xp.real(b0[nz : nz + context.geometry.av.size])
        ).item()
    )
    energy = (
        kinetic[0]
        + ion_internal
        + electron_perp
        + electron_parallel
        + magnetic
    )
    return SimpleNamespace(
        number=number,
        kinetic=kinetic,
        ion_internal=ion_internal,
        electron_perp=electron_perp,
        electron_parallel=electron_parallel,
        magnetic=magnetic,
        longitudinal=0.0,
        energy=energy,
        topology=topology,
        physical_flux=physical_flux,
        applicability=_applicability(),
    )


def _electron_ledger(stage, context):
    xp = _xp(context)
    nc = context.op.Nc
    fluid = stage.fluid[1]
    electron = context.p.species[1]
    electric = inverse(
        stage.cache.E,
        context.op.Ne,
        context,
        True,
        context.p.ref.electric,
    )
    edge_electric = (
        electron.charge
        * xp.conj(fluid.flux.reshape(context.N, context.op.Ne))
        * electric.reshape(context.N, context.op.Ne)
    )
    from .massless import _constraint_project

    e = -float(electron.charge)
    pressure_source = (
        xp.asarray(fluid.pressure_force).reshape(context.N, 3, nc) / e
    )
    pressure_flux = _constraint_project(
        pressure_source.reshape(-1), fluid.n, context
    ).reshape(context.N, 3, nc)
    pressure_flux /= fluid.n.reshape(context.N, 1, nc)
    pressure_flux = pressure_flux.reshape(-1)
    pressure_electric_field = sample_apply(
        context.op.I,
        pressure_flux,
        context.N,
        context.compute,
    )
    pressure_electric_edge = (
        electron.charge
        * xp.conj(fluid.flux.reshape(context.N, context.op.Ne))
        * pressure_electric_field.reshape(context.N, context.op.Ne)
    )
    fluid.pressure_electric_field = pressure_electric_field.reshape(-1)
    node_electric = sample_apply(
        context.op.H,
        electric,
        context.N,
        context.compute,
    ).reshape(context.N, 3, nc)
    node_electric *= (
        electron.charge
        * xp.real(xp.asarray(fluid.n)).reshape(context.N, 1, nc)
        * xp.asarray(fluid.u).reshape(context.N, 3, nc)
    )
    velocity = xp.asarray(fluid.u).reshape(context.N, 3, nc)
    pressure_kinetic = xp.conj(velocity) * xp.asarray(
        fluid.pressure_force
    ).reshape(context.N, 3, nc)
    magnetic = xp.conj(velocity) * xp.asarray(
        fluid.magnetic_force
    ).reshape(context.N, 3, nc)
    perp = xp.real(xp.asarray(fluid.perp)).reshape(context.N, nc)
    parallel = xp.real(xp.asarray(fluid.parallel)).reshape(context.N, nc)
    perp_back = inverse(
        stage.rhs.fluid[1].perp,
        nc,
        context,
        True,
        context.p.ref.pressure / context.p.ref.time,
    ).reshape(context.N, nc)
    parallel_back = inverse(
        stage.rhs.fluid[1].parallel,
        nc,
        context,
        True,
        context.p.ref.pressure / context.p.ref.time,
    ).reshape(context.N, nc)
    pressure_full = (
        xp.asarray(fluid.pdot_perp).reshape(context.N, nc)
        + 0.5
        * xp.asarray(fluid.pdot_parallel).reshape(context.N, nc)
    )
    pressure_retained = perp_back + 0.5 * parallel_back
    node_electric_work = _physical_real(
        _integrate_cells(node_electric.reshape(-1), 3, context),
        "electron compatibility node electric work",
    )
    pressure_full_integral = _physical_real(
        _integrate_cells(pressure_full.reshape(-1), 1, context),
        "full padded pressure energy rate",
    )
    pressure_retained_integral = _physical_real(
        _integrate_cells(pressure_retained.reshape(-1), 1, context),
        "retained pressure energy rate",
    )
    edge_electric_work = _physical_real(
        _integrate_edges(edge_electric.reshape(-1), context),
        "electron edge electric power",
    )
    return SimpleNamespace(
        electric_node=0.0,
        node_electric_work=node_electric_work,
        electric_edge=edge_electric_work,
        pressure_electric_edge=_physical_real(
            _integrate_edges(
                pressure_electric_edge.reshape(-1), context
            ),
            "electron pressure-only electric power",
        ),
        pressure_kinetic=_physical_real(
            _integrate_cells(pressure_kinetic.reshape(-1), 3, context),
            "electron pressure kinetic work",
        ),
        pressure_internal=pressure_retained_integral,
        pressure_internal_full=pressure_full_integral,
        pressure_truncation=pressure_retained_integral - pressure_full_integral,
        magnetic=_physical_real(
            _integrate_cells(magnetic.reshape(-1), 3, context),
            "electron magnetic work",
        ),
        magnetic_scale=_physical_real(
            _integrate_cells(xp.abs(magnetic).reshape(-1), 3, context),
            "electron magnetic work scale",
        ),
        wall=_physical_real(
            _integrate_edges(
                (
                    electron.charge
                    * xp.conj(fluid.flux.reshape(context.N, context.op.Ne))
                    * inverse(
                        stage.cache.wall_reaction,
                        context.op.Ne,
                        context,
                        True,
                        context.p.ref.electric,
                    ).reshape(context.N, context.op.Ne)
                ).reshape(-1),
                context,
            ),
            "electron wall reaction work",
        ),
        convection=0.0,
        perp_rate=_physical_real(
            _integrate_cells(perp_back.reshape(-1), 1, context),
            "electron perpendicular pressure source",
        ),
        parallel_rate=_physical_real(
            0.5 * _integrate_cells(parallel_back.reshape(-1), 1, context),
            "electron parallel pressure source",
        ),
        spectral_kinetic=0.0,
        node_edge_reconstruction_difference=(
            node_electric_work - edge_electric_work
        ),
        spectral_perp=_physical_real(
            _integrate_cells(
                (
                    perp_back
                    - xp.asarray(fluid.pdot_perp).reshape(context.N, nc)
                ).reshape(-1),
                1,
                context,
            ),
            "electron perpendicular spectral transfer",
        ),
        spectral_parallel=_physical_real(
            _integrate_cells(
                (
                    0.5
                    * (
                        parallel_back
                        - xp.asarray(fluid.pdot_parallel).reshape(
                            context.N, nc
                        )
                    )
                ).reshape(-1),
                1,
                context,
            ),
            "electron parallel spectral transfer",
        ),
        applicability=_applicability(),
    )


def _wall_trace_data(state, stage, context):
    xp = _xp(context)
    nc = context.op.Nc
    nr = len(context.geometry.r_faces.keys)
    ne = context.op.Ne
    nz = len(context.geometry.z_faces.keys)
    nv = len(context.geometry.vertices.keys)
    electric = inverse(
        stage.cache.E,
        ne,
        context,
        True,
        context.p.ref.electric,
    ).reshape(context.N, ne)
    magnetic = inverse(state.B, context.op.Nb, context, True).reshape(
        context.N, context.op.Nb
    )
    traces = []
    physical_wall_B = (
        wall_magnetic_trace(state.B, context)
        if getattr(context.op, "box_wall", None) is not None else None
    )
    for wall_id, wall in enumerate(context.geometry.walls):
        radial = wall.component == "r_face"
        edge = wall.face if radial else nr + nc + wall.face
        direction = 0 if radial else 2
        sigma = wall.sign
        normal_row = xp.asarray(context.op.I[edge]).reshape(3, nc)[direction]
        area = wall.area_per_radian * 2.0 * math.pi / context.N
        bvectors = [
            xp.zeros(context.N, dtype=xp.float64) for _ in range(3)
        ]
        bvectors[1] = 0.5 * (
            xp.real(magnetic[:, nz + wall.vertices[0]])
            + xp.real(magnetic[:, nz + wall.vertices[1]])
        )
        if radial:
            bvectors[2] = xp.real(magnetic[:, nz + nv + wall.face])
        else:
            bvectors[0] = xp.real(magnetic[:, wall.face])
        if physical_wall_B is not None:
            bvectors = [physical_wall_B[:, d, wall_id] for d in range(3)]
        density = []
        velocity = []
        perp = []
        parallel = []
        for species in range(2):
            fluid = stage.fluid[species]
            density_trace = xp.real(
                xp.asarray(fluid.n).reshape(context.N, nc)
                @ normal_row
            )
            if not bool(xp.all(density_trace > context.p.species[species].face_min)):
                raise Failure("WallDensity", "wall trace below density floor")
            if species == 0:
                momentum = xp.asarray(fluid.mom).reshape(
                    context.N, 3, nc
                )
            else:
                momentum = (
                    xp.real(xp.asarray(fluid.n)).reshape(
                        context.N, 1, nc
                    )
                    * xp.asarray(fluid.u).reshape(context.N, 3, nc)
                )
            momentum_to_number_velocity = (
                context.p.species[species].mass if species == 0 else 1.0
            )
            velocity_trace = xp.stack(
                [
                    xp.real(momentum[:, component, :] @ normal_row)
                    / (momentum_to_number_velocity * density_trace)
                    for component in range(3)
                ],
                axis=1,
            )
            perp_trace = xp.real(
                xp.asarray(fluid.perp).reshape(context.N, nc)
                @ normal_row
            )
            parallel_trace = (
                xp.real(
                    xp.asarray(fluid.parallel).reshape(context.N, nc)
                    @ normal_row
                )
                if species
                else perp_trace
            )
            density.append(density_trace)
            velocity.append(velocity_trace)
            perp.append(perp_trace)
            parallel.append(parallel_trace)
        traces.append(
            SimpleNamespace(
                wall=wall,
                edge=edge,
                direction=direction,
                sigma=sigma,
                area=area,
                B=bvectors,
                density=density,
                velocity=velocity,
                perp=perp,
                parallel=parallel,
                electric=electric[:, edge],
            )
        )
    return traces


def _boundary(state, stage, context):
    """Integrate actual per-wall traces; electron kinetic transport is absent."""
    xp = _xp(context)
    nc = context.op.Nc
    ne = context.op.Ne
    result = SimpleNamespace(
        traces=[],
        number=[0.0, 0.0],
        kinetic=[0.0, 0.0],
        internal_perp=[0.0, 0.0],
        internal_parallel=[0.0, 0.0],
        field_energy=0.0,
        electron_pressure_work=0.0,
        total_energy=0.0,
        applicability=_applicability(),
    )
    box = getattr(context.op, "box_wall", None) is not None
    trace_data = _wall_trace_data(state, stage, context)
    for sample in range(context.N):
        for data in trace_data:
            wall = data.wall
            edge = data.edge
            direction = data.direction
            area = data.area
            trace = SimpleNamespace(
                B=[
                    float(xp.real(value[sample]).item())
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
            trace.ET[direction] = wall.sign * float(
                data.electric[sample].real
            )
            trace.EL = [0.0, 0.0, 0.0]
            for species in range(2):
                fluid = stage.fluid[species]
                density = float(data.density[species][sample].item())
                normal_flux = wall.sign * float(
                    xp.real(xp.asarray(fluid.flux)).reshape(
                        context.N, ne
                    )[sample, edge]
                )
                pressure_flux = wall.sign * float(
                    xp.real(xp.asarray(fluid.pressure_flux)).reshape(
                        context.N, ne
                    )[sample, edge]
                )
                perp = float(data.perp[species][sample].item())
                parallel = float(data.parallel[species][sample].item())
                velocity = data.velocity[species][sample]
                un = wall.sign * float(velocity[direction].real)
                kinetic_density = (
                    0.5
                    * context.p.species[species].mass
                    * density
                    * float(xp.sum(xp.abs(velocity) ** 2).item())
                    if species == 0
                    else 0.0
                )
                trace.density[species] = density
                trace.normal_flux[species] = normal_flux
                trace.perp[species] = perp
                trace.parallel[species] = parallel
                trace.velocity[species] = [float(v.real) for v in velocity]
                result.number[species] += area * normal_flux
                result.kinetic[species] += area * kinetic_density * un
                result.internal_perp[species] += (
                    area
                    * pressure_flux
                    * (
                        1.0
                        if species
                        else 1.0 / (context.p.species[0].gamma - 1.0)
                    )
                )
                if species:
                    parallel_flux = wall.sign * float(
                        xp.real(xp.asarray(fluid.parallel_flux)).reshape(
                            context.N, ne
                        )[sample, edge]
                    )
                    result.internal_parallel[species] += (
                        area * 0.5 * parallel_flux
                    )
                internal_density = (
                    perp + 0.5 * parallel
                    if species
                    else perp / (context.p.species[0].gamma - 1.0)
                )
                pressure_work = perp * un
                if species:
                    direction_b = float(
                        xp.real(data.B[direction][sample]).item()
                    )
                    parallel_b = float(
                        sum(
                            data.B[d][sample]
                            * velocity[d]
                            for d in range(3)
                        ).real.item()
                    )
                    bmag = float(
                        xp.sqrt(
                            xp.sum(
                                xp.stack(data.B)[:, sample] ** 2
                            )
                        ).item()
                    )
                    pressure_work += (
                        (parallel - perp)
                        * parallel_b
                        * wall.sign
                        * direction_b
                        / (bmag * bmag)
                    )
                    if box:
                        result.electron_pressure_work += area * pressure_work
                result.total_energy += (
                    area
                    * (
                        (kinetic_density + internal_density) * un
                        + pressure_work
                    )
                )
            result.traces.append(trace)
    result._trace_data = trace_data
    return result


def momentum(state, result, context):
    """Return Cartesian ion momentum and the approved no-displacement traction."""
    xp = _xp(context)
    nc = context.op.Nc
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    cell_radius = xp.asarray(context.geometry.cells.R).reshape(nc)
    ion = xp.asarray(result.fluid[0].mom).reshape(context.N, 3, nc)
    reaction = xp.asarray(result.terms[0].reaction).reshape(
        context.N, 3, nc
    )
    ion_rate = inverse(
        result.rhs.fluid[0].mom,
        3 * nc,
        context,
        True,
        context.p.species[0].mass
        * context.p.ref.ni
        * context.p.ref.velocity
        / context.p.ref.time,
    ).reshape(context.N, 3, nc)
    samples = max(context.N, 3)
    value = np.zeros(3)
    pdot = np.zeros(3)
    reaction_xyz = np.zeros(3)
    traction_xyz = np.zeros(3)
    angular = 0.0
    angular_rate = 0.0
    wall_torque = 0.0
    outward_torque = 0.0
    trace_data = result.boundary._trace_data

    def traction_at(data, sample):
        direction = data.direction
        sigma = data.sigma
        b = [
            float(xp.real(component[sample]).item())
            for component in data.B
        ]
        b2 = sum(component * component for component in b)
        bmag = math.sqrt(b2)
        b_normal = b[direction]
        direction_b = b_normal / bmag
        velocity = data.velocity[0][sample]
        un = sigma * float(velocity[direction].real)
        density = float(data.density[0][sample].item())
        ion_perp = float(data.perp[0][sample].item())
        electron_perp = float(data.perp[1][sample].item())
        electron_parallel = float(data.parallel[1][sample].item())
        traction = []
        for component in range(3):
            unit = 1.0 if component == direction else 0.0
            magnetic = sigma * (
                0.5 * b2 * unit - b[component] * b_normal
            ) / context.p.mu0
            ion_ram = (
                context.p.species[0].mass
                * density
                * float(velocity[component].real)
                * un
            )
            ion_pressure = ion_perp * sigma * unit
            electron_pressure = (
                electron_perp * sigma * unit
                + (electron_parallel - electron_perp)
                * (b[component] / bmag)
                * direction_b
                * sigma
            )
            traction.append(
                magnetic + ion_ram + ion_pressure + electron_pressure
            )
        return traction

    for q in range(samples):
        sample = 0 if context.N == 1 else q
        angle = 2.0 * math.pi * q / samples
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        weights = 2.0 * math.pi * m0 / samples
        for target, source in (
            (value, ion[sample]),
            (pdot, ion_rate[sample]),
            (reaction_xyz, reaction[sample]),
        ):
            cartesian = (
                source[0] * cos_angle - source[1] * sin_angle,
                source[0] * sin_angle + source[1] * cos_angle,
                source[2],
            )
            for component in range(3):
                target[component] += _physical_real(
                    xp.sum(weights * cartesian[component]),
                    "ion momentum integral",
                )
        angular += _physical_real(
            xp.sum(weights * cell_radius * ion[sample, 1, :]),
            "ion angular momentum integral",
        )
        angular_rate += _physical_real(
            xp.sum(weights * cell_radius * ion_rate[sample, 1, :]),
            "ion angular momentum rate",
        )
        wall_torque += _physical_real(
            xp.sum(weights * cell_radius * reaction[sample, 1, :]),
            "ion projection torque",
        )
        for data in trace_data:
            traction = traction_at(data, sample)
            cartesian = (
                traction[0] * cos_angle - traction[1] * sin_angle,
                traction[0] * sin_angle + traction[1] * cos_angle,
                traction[2],
            )
            wall_weight = (
                2.0 * math.pi * data.wall.area_per_radian / samples
            )
            surface_radius = (
                float(context.geometry.r_faces.R[data.wall.face])
                if data.wall.component == "r_face"
                else float(context.geometry.z_faces.R[data.wall.face])
            )
            for component in range(3):
                traction_xyz[component] += wall_weight * cartesian[component]
            outward_torque += (
                wall_weight * surface_radius * traction[1]
            )
    balance_defect = pdot + traction_xyz
    return SimpleNamespace(
        value=value.tolist(),
        pdot=pdot.tolist(),
        outward=traction_xyz.tolist(),
        wall_reaction=reaction_xyz.tolist(),
        angular=angular,
        angular_rate=angular_rate,
        angular_balance_defect=angular_rate + outward_torque,
        outward_torque=outward_torque,
        wall_torque=wall_torque,
        angular_samples=max(context.N, 3),
        balance_defect=balance_defect.tolist(),
        applicability=SimpleNamespace(
            electron_momentum="not_applicable",
            electron_kinetic_energy="not_applicable",
            electron_mechanical_stress="not_applicable",
            longitudinal_electric_stress="not_applicable",
            boundary_traction=(
                "mi ni ui ui + pi I + Pe + |B|^2 I/(2 mu0) "
                "- B B/mu0 applied to the actual wall normal"
            ),
        ),
    )


def _gamma_operation(count, epsilon):
    if (
        not math.isfinite(count)
        or not math.isfinite(epsilon)
        or count < 0
        or epsilon <= 0.0
        or count * epsilon >= 1.0
    ):
        raise Failure("WallWork", "invalid floating-point operation count")
    return count * epsilon / (1.0 - count * epsilon)


def _wall_work_metrics(state, result, context):
    """Retained-mode work with producer envelopes for a backward-error gate."""
    xp = _xp(context)
    nc = context.op.Nc
    ne = context.op.Ne
    nb = context.op.Nb
    modes = _modes(context)
    eps = float(xp.finfo(xp.float64).eps)

    def gamma(count):
        return _gamma_operation(count, eps)

    def max_abs(value):
        return (
            float(xp.max(xp.abs(value)).item())
            if value.size
            else 0.0
        )

    reaction_hat = xp.asarray(
        result.cache.wall_reaction, dtype=xp.complex128
    ).reshape(modes, ne)
    reaction_samples = inverse(
        result.cache.wall_reaction,
        ne,
        context,
        True,
        context.p.ref.electric,
    ).reshape(context.N, ne)
    m1 = xp.real(xp.asarray(context.op.M1)).astype(xp.float64)
    abs_i = xp.abs(xp.asarray(context.op.I, dtype=xp.complex128))
    abs_cdag = xp.abs(xp.asarray(context.op.Cdag, dtype=xp.complex128))
    abs_t_full = xp.abs(
        xp.asarray(context.op.T_full, dtype=xp.complex128)
    )
    t_full = xp.asarray(context.op.T_full, dtype=xp.complex128)
    ion_momentum = xp.asarray(state.fluid[0].mom).reshape(modes, 3 * nc)
    magnetic = xp.asarray(state.B).reshape(modes, nb)
    ion_charge = abs(float(context.p.species[0].charge))
    ion_mass = float(context.p.species[0].mass)
    electron_charge = float(context.p.species[1].charge)
    qi = float(context.p.species[0].charge)
    e = -electron_charge

    a_ion = xp.empty((modes, ne), dtype=xp.float64)
    a_current = xp.empty((modes, ne), dtype=xp.float64)
    d_ion_direct = xp.empty((modes, ne), dtype=xp.float64)
    d_current_direct = xp.empty((modes, ne), dtype=xp.float64)
    b_ion = xp.zeros((modes, ne), dtype=xp.float64)
    b_current = xp.zeros((modes, ne), dtype=xp.float64)
    momentum_allowance = 0.0
    current_allowance = 0.0
    extraction = [
        xp.asarray(value, dtype=xp.complex128)
        for value in result.cache.wall_correction.extraction
    ]
    if len(extraction) != modes:
        raise Failure("WallWork", "wall extraction mode count mismatch")
    current_hat = xp.asarray(result.current).reshape(modes, ne)
    current_reference = (
        4096.0
        * eps
        * nb
        * max(
            (
                max_abs(abs_cdag[stored])
                * max_abs(magnetic[stored])
                / context.p.mu0
            )
            for stored in range(modes)
        )
    )
    if not math.isfinite(current_reference) or current_reference < 0.0:
        raise Failure("WallWork", "invalid current compatibility allowance")
    current_allowance = current_reference
    momentum_reference = (
        ion_mass * context.p.ref.ni * context.p.ref.velocity
    )
    for stored in range(modes):
        a_ion[stored] = (
            ion_charge
            / ion_mass
            * (abs_i @ xp.abs(ion_momentum[stored]))
        )
        a_current[stored] = (
            abs_cdag[stored] @ xp.abs(magnetic[stored])
        ) / context.p.mu0
        d_ion_direct[stored] = (
            gamma(8 * 3 * nc + 4) * a_ion[stored]
        )
        d_current_direct[stored] = (
            gamma(8 * nb + 2) * a_current[stored]
        )
        momentum_limit = (
            2048.0
            * eps
            * max(
                max_abs(ion_momentum[stored]),
                momentum_reference,
                1.0e-300,
            )
        )
        momentum_allowance = max(momentum_allowance, momentum_limit)
        wall_momentum = t_full @ ion_momentum[stored]
        if max_abs(wall_momentum) > momentum_limit:
            raise Failure(
                "WallWorkInput",
                "ion momentum no longer satisfies its checked wall allowance",
            )
        trace_roundoff = gamma(8 * 3 * nc) * (
            abs_t_full @ xp.abs(ion_momentum[stored])
        )
        edge_allowance = (
            ion_charge
            / ion_mass
            * (
                extraction[stored].T.conj()
                @ (momentum_limit + trace_roundoff)
            )
        )
        b_ion[stored] = xp.abs(edge_allowance)
        wall_current = extraction[stored] @ current_hat[stored]
        if max_abs(wall_current) > current_reference:
            raise Failure(
                "WallWorkInput",
                "current no longer satisfies its checked wall allowance",
            )
        current_edge_allowance = (
            extraction[stored].T.conj()
            @ xp.full(
                (extraction[stored].shape[0],),
                current_reference,
                dtype=xp.float64,
            )
        )
        b_current[stored] = xp.abs(current_edge_allowance)
    a_electron = a_ion + a_current
    d_electron_direct = (
        (1.0 + gamma(10)) * (d_ion_direct + d_current_direct)
        + gamma(10) * a_electron
    )
    b_electron = (1.0 + gamma(10)) * (b_ion + b_current)
    d_ion = d_ion_direct + b_ion
    d_current = d_current_direct + b_current
    d_electron = d_electron_direct + b_electron

    retained_flux = {
        "ion": qi * xp.asarray(result.fluid[0].flux_hat),
        "electron": electron_charge * xp.asarray(result.fluid[1].flux_hat),
        "current": xp.asarray(result.current),
    }
    sample_flux = {
        "ion": result.fluid[0].flux,
        "electron": result.fluid[1].flux,
        "current": inverse(result.current, ne, context),
    }
    charge = {
        "ion": qi,
        "electron": electron_charge,
        "current": 1.0,
    }
    direct_envelopes = {
        "ion": d_ion_direct,
        "electron": d_electron_direct,
        "current": d_current_direct,
    }
    compatibility_envelopes = {
        "ion": b_ion,
        "electron": b_electron,
        "current": b_current,
    }
    for values in (
        a_ion,
        a_current,
        d_ion_direct,
        d_current_direct,
        b_ion,
        b_current,
        d_electron_direct,
        b_electron,
        d_ion,
        d_current,
        d_electron,
    ):
        if not bool(xp.all(xp.isfinite(values) & (values >= 0.0)).item()):
            raise Failure("WallWork", "producer bound is nonfinite or negative")
    mode_weight = m1[None, :]
    output = {}
    for name in ("ion", "electron", "current"):
        flux_hat = retained_flux[name].reshape(modes, ne)
        work = 2.0 * math.pi * xp.sum(
            mode_weight
            * xp.real(xp.conj(flux_hat) * reaction_hat)
        )
        absolute_scale = 2.0 * math.pi * xp.sum(
            mode_weight
            * xp.abs(flux_hat)
            * xp.abs(reaction_hat)
        )
        upstream_bound = 2.0 * math.pi * xp.sum(
            mode_weight
            * xp.abs(reaction_hat)
            * direct_envelopes[name]
        )
        compatibility_bound = 2.0 * math.pi * xp.sum(
            mode_weight
            * xp.abs(reaction_hat)
            * compatibility_envelopes[name]
        )
        sum_bound = gamma(10 * modes * ne + 8) * absolute_scale
        arithmetic_bound = (
            (1.0 + gamma(2))
            * (upstream_bound + compatibility_bound + sum_bound)
        )
        sample_work = _integrate_edges(
            (
                charge[name]
                * xp.conj(
                    xp.asarray(sample_flux[name]).reshape(context.N, ne)
                )
                * reaction_samples
            ).reshape(-1),
            context,
        )
        output[name] = SimpleNamespace(
            work=_physical_real(work, f"{name} retained wall work"),
            sample_work=_physical_real(
                sample_work, f"{name} sample wall work"
            ),
            transform_difference=_physical_real(
                sample_work - work, f"{name} sample/modal work difference"
            ),
            absolute_scale=_bound_real(
                absolute_scale, f"{name} retained absolute work scale"
            ),
            upstream_bound=_bound_real(
                upstream_bound, f"{name} producer arithmetic bound"
            ),
            compatibility_bound=_bound_real(
                compatibility_bound,
                f"{name} checked-input compatibility bound",
            ),
            arithmetic_bound=_bound_real(
                arithmetic_bound, f"{name} retained work arithmetic bound"
            ),
            momentum_allowance=momentum_allowance,
            current_allowance=current_allowance,
        )
    output["species"] = SimpleNamespace(
        work=output["ion"].work + output["electron"].work,
        sample_work=(
            output["ion"].sample_work + output["electron"].sample_work
        ),
        transform_difference=(
            output["ion"].transform_difference
            + output["electron"].transform_difference
        ),
        absolute_scale=_bound_real(
            output["ion"].absolute_scale + output["electron"].absolute_scale,
            "species retained absolute work scale",
        ),
        upstream_bound=_bound_real(
            (1.0 + gamma(2))
            * (
                output["ion"].upstream_bound
                + output["electron"].upstream_bound
            ),
            "species producer arithmetic bound",
        ),
        compatibility_bound=_bound_real(
            (1.0 + gamma(2))
            * (
                output["ion"].compatibility_bound
                + output["electron"].compatibility_bound
            ),
            "species checked-input compatibility bound",
        ),
        arithmetic_bound=_bound_real(
            (1.0 + gamma(2))
            * (
                output["ion"].arithmetic_bound
                + output["electron"].arithmetic_bound
            )
            + gamma(2)
            * (
                abs(output["ion"].work)
                + abs(output["electron"].work)
            ),
            "species retained work arithmetic bound",
        ),
        momentum_allowance=momentum_allowance,
        current_allowance=current_allowance,
    )
    return output


def _stage_residuals(state, result, context):
    xp = _xp(context)
    nc = context.op.Nc
    ne = context.op.Ne
    qi = float(context.p.species[0].charge)
    qe = float(context.p.species[1].charge)
    e = -qe
    residuals = {}
    residuals["quasineutrality"] = xp.asarray(result.rho).copy()
    residuals["current_definition"] = (
        xp.asarray(result.current)
        - mode_apply(context.op.Cdag, state.B, context.compute) / context.p.mu0
    )
    residuals["electron_flux_definition"] = (
        xp.asarray(result.fluid[1].flux_hat)
        - (qi * xp.asarray(result.fluid[0].flux_hat) - result.current) / e
    )
    residuals["electron_density_compatibility"] = (
        xp.asarray(result.rhs.fluid[1].n)
        - (qi / e) * xp.asarray(result.rhs.fluid[0].n)
    )
    residuals["electron_momentum_compatibility"] = xp.asarray(
        result.rhs.fluid[1].mom
    ).copy()
    residuals["electron_flux_continuity"] = (
        xp.asarray(result.rhs.fluid[1].n)
        + mode_apply(
            context.op.DJ,
            result.fluid[1].flux_hat,
            context.compute,
        )
    )
    residuals["djj"] = mode_apply(
        context.op.DJ, result.current, context.compute
    )
    residuals["ct_divergence"] = mode_apply(
        context.op.DB, result.rhs.B, context.compute
    )
    wall_fluxes = {}
    for name, flux in (
        ("ion", result.fluid[0].flux),
        ("electron", result.fluid[1].flux),
        ("current", inverse(result.current, ne, context)),
    ):
        wall_fluxes[name] = _wall_flux_samples(flux, context)
    residuals["wall_field_original_rows"] = xp.concatenate(
        [
            record.full_residual
            for record in result.cache.wall_correction.modes
        ]
    )
    reaction = xp.asarray(result.cache.wall_reaction).copy()
    wall_metrics = _wall_work_metrics(state, result, context)
    for name in ("ion", "electron", "current"):
        metric = wall_metrics[name]
        residuals[f"wall_field_reaction_scale_{name}"] = xp.asarray(
            metric.absolute_scale
        ).reshape(1)
        residuals[f"wall_field_reaction_sample_work_{name}"] = xp.asarray(
            metric.sample_work
        ).reshape(1)
        residuals[f"wall_field_reaction_transform_difference_{name}"] = (
            xp.asarray(metric.transform_difference).reshape(1)
        )
        residuals[f"wall_field_reaction_upstream_bound_{name}"] = (
            xp.asarray(metric.upstream_bound).reshape(1)
        )
        residuals[f"wall_field_reaction_compatibility_bound_{name}"] = (
            xp.asarray(metric.compatibility_bound).reshape(1)
        )
        residuals[f"wall_field_reaction_arithmetic_bound_{name}"] = (
            xp.asarray(metric.arithmetic_bound).reshape(1)
        )
        residuals[f"wall_field_reaction_work_{name}"] = xp.asarray(
            metric.work
        ).reshape(1)
    species_metric = wall_metrics["species"]
    residuals["wall_field_reaction_work"] = xp.asarray(
        species_metric.work
    ).reshape(1)
    residuals["wall_field_reaction_scale"] = xp.asarray(
        species_metric.absolute_scale
    ).reshape(1)
    residuals["wall_field_reaction_sample_work"] = xp.asarray(
        species_metric.sample_work
    ).reshape(1)
    residuals["wall_field_reaction_transform_difference"] = xp.asarray(
        species_metric.transform_difference
    ).reshape(1)
    residuals["wall_field_reaction_upstream_bound"] = xp.asarray(
        species_metric.upstream_bound
    ).reshape(1)
    residuals["wall_field_reaction_compatibility_bound"] = xp.asarray(
        species_metric.compatibility_bound
    ).reshape(1)
    residuals["wall_field_reaction_arithmetic_bound"] = xp.asarray(
        species_metric.arithmetic_bound
    ).reshape(1)
    residuals["wall_field_momentum_allowance"] = xp.asarray(
        species_metric.momentum_allowance
    ).reshape(1)
    residuals["wall_field_current_allowance"] = xp.asarray(
        species_metric.current_allowance
    ).reshape(1)
    residuals["drift_ohm_residual"] = (
        xp.asarray(result.cache.E)
        - xp.asarray(result.E0)
        - reaction
    )
    perp_expected = (
        -_sample_divergence_transport(
            result.fluid[1].pressure_flux,
            _modes(context),
            context,
        )
        + xp.asarray(result.fluid[1].perp).reshape(context.N, nc)
        * (
            xp.asarray(result.fluid[1].strain).reshape(context.N, nc)
            - xp.asarray(result.fluid[1].theta).reshape(context.N, nc)
        )
    )
    parallel_expected = (
        -_sample_divergence_transport(
            result.fluid[1].parallel_flux,
            _modes(context),
            context,
        )
        - 2.0
        * xp.asarray(result.fluid[1].parallel).reshape(context.N, nc)
        * xp.asarray(result.fluid[1].strain).reshape(context.N, nc)
    )
    residuals["electron_cgl_perp"] = (
        xp.asarray(result.fluid[1].pdot_perp).reshape(context.N, nc)
        - perp_expected
    ).reshape(-1)
    residuals["electron_cgl_parallel"] = (
        xp.asarray(result.fluid[1].pdot_parallel).reshape(context.N, nc)
        - parallel_expected
    ).reshape(-1)
    residuals["electron_pressure_adjoint"] = (
        xp.asarray(result.ledger[1].pressure_kinetic)
        + xp.asarray(result.ledger[1].pressure_internal_full)
        + xp.asarray(result.boundary.electron_pressure_work)
    ).reshape(1)
    residuals["electron_electric_pairing"] = xp.asarray(
        result.ledger[1].pressure_internal
        - result.ledger[1].electric_edge
        + result.boundary.electron_pressure_work
    ).reshape(1)
    residuals["electron_pressure_only_pairing"] = xp.asarray(
        result.ledger[1].pressure_kinetic
        + result.ledger[1].pressure_electric_edge
    ).reshape(1)
    residuals["electron_pressure_truncation"] = xp.asarray(
        result.ledger[1].pressure_truncation
    ).reshape(1)
    residuals["theta_consistency"] = xp.asarray(
        result.fluid[1].theta
    ) - xp.asarray(result.fluid[1].theta_consistency)
    return residuals


def _wall_flux_samples(flux, context):
    xp = _xp(context)
    nc = context.op.Nc
    nr = len(context.geometry.r_faces.keys)
    samples = xp.asarray(flux).reshape(context.N, context.op.Ne)
    result = xp.empty(
        (context.N, len(context.geometry.walls)), dtype=xp.complex128
    )
    for index, wall in enumerate(context.geometry.walls):
        edge = (
            wall.face
            if wall.component == "r_face"
            else nr + nc + wall.face
        )
        result[:, index] = wall.sign * samples[:, edge]
    return result.reshape(-1)


def _sample_divergence_transport(flux, entities, context):
    from .fluid import _sampled_divergence

    if context.N * context.op.Ne != xp_size(flux):
        raise Failure("Shape", "pressure transport shape mismatch")
    return _sampled_divergence(flux, context).reshape(context.N, context.op.Nc)


def xp_size(value):
    return int(value.size)


def populate_stage(state, result, context):
    """Populate all diagnostic-owned stage records without mutating state."""
    result.ledger[0] = power_ledger(result, 0, context)
    result.ledger[0].applicability = _applicability()
    result.ledger[1] = _electron_ledger(result, context)
    result.boundary = _boundary(state, result, context)
    result.totals = state_totals(state, result, context)
    result.momentum = momentum(state, result, context)
    result.residuals.update(_stage_residuals(state, result, context))


def _truth(out, name, ok, tag, reason=""):
    record = diagnostics._record(name=name, tag=tag)
    record.raw = 0.0 if ok else 1.0
    record.reference = 1.0
    record.threshold = 0.0
    record.passed = bool(ok)
    record.available = True
    if reason:
        record.reason = reason
    out.append(record)


def _scaled_scalar(
    out, name, raw, scale, unit, tag, context, population=""
):
    record = diagnostics._record(
        name=name, tag=tag, unit=unit, population=population
    )
    record.raw = float(raw)
    record.reference = float(scale)
    roundoff_threshold = (
        4096.0 * sys.float_info.epsilon * float(scale)
        if scale > 0.0
        else 0.0
    )
    record.threshold = roundoff_threshold
    record.available = math.isfinite(record.raw) and math.isfinite(scale)
    configured_threshold = 0.0
    try:
        configured = diagnostics.budget(
            context,
            diagnostics._budget_key(name, population, "value"),
        )
        record.hard = bool(configured.hard)
        configured_threshold = (
            float(configured.atol)
            + float(configured.rtol) * float(configured.reference)
        )
    except (AttributeError, Failure):
        record.hard = True
    record.roundoff_threshold = record.threshold
    record.configured_threshold = configured_threshold
    record.threshold = max(record.threshold, configured_threshold)
    record.passed = record.available and abs(record.raw) <= record.threshold
    out.append(record)


def _convergence_scalar(
    out, name, raw, unit, tag, reference=0.0, reason=""
):
    record = diagnostics._record(name=name, tag=tag, unit=unit)
    record.raw = float(raw)
    record.reference = float(reference)
    record.threshold = (
        4096.0 * sys.float_info.epsilon * abs(float(reference))
        if reference
        else 0.0
    )
    record.available = math.isfinite(record.raw)
    record.passed = record.available and abs(record.raw) <= record.threshold
    record.hard = False
    record.reason = reason
    out.append(record)


def _unavailable(out, name, tag, reason):
    record = diagnostics._record(name=name, tag=tag)
    record.available = False
    record.hard = False
    record.passed = False
    record.reason = reason
    out.append(record)


def _minimum(out, name, values, floor, unit, population=""):
    record = diagnostics._record(
        name=name,
        tag=out_tag(out),
        unit=unit,
        population=population,
    )
    record.reference = float(floor)
    record.threshold = float(floor)
    record.available = values.size > 0
    record.raw = (
        float(_xp_dummy(values).min().real) if record.available else math.inf
    )
    record.passed = record.available and record.raw > floor
    out.append(record)


def _xp_dummy(values):
    return values


def out_tag(out):
    return out[-1].tag if out else SimpleNamespace()


def check_stage(state, result, context):
    """Hard-gate algebraic branch identities and report convergence defects."""
    xp = _xp(context)
    nc = context.op.Nc
    ne = context.op.Ne
    out = result.checks
    tag = result.tag
    finite = all(
        bool(xp.all(xp.isfinite(value)).item())
        for value in (
            state.B,
            result.rhs.B,
            result.cache.E,
            result.current,
            result.rhs.fluid[0].n,
            result.rhs.fluid[0].mom,
            result.rhs.fluid[0].perp,
            result.rhs.fluid[1].n,
            result.rhs.fluid[1].mom,
            result.rhs.fluid[1].perp,
            result.rhs.fluid[1].parallel,
        )
    )
    _truth(out, "finite", finite, tag)
    _truth(
        out,
        "model_massless_quasineutral",
        result.model == MODEL_NAME
        and int(getattr(context.p, "electron_inertia", 0)) == 0
        and context.p.species[1].mass > 0.0,
        tag,
    )
    ownership = (
        result.source_id != 0
        and result.cache.tag == tag
        and result.fluid[0].wall.tag == tag
        and result.fluid[1].wall.tag == tag
        and state.step == tag.step
        and state.generation == tag.generation
        and state.time == tag.time
    )
    _truth(out, "source_identity", ownership, tag)
    _minimum(
        out,
        "minimum_B",
        xp.sqrt(
            xp.sum(
                xp.real(xp.asarray(result.BN)).reshape(context.N, 3, nc) ** 2,
                axis=1,
            )
        ).reshape(-1),
        context.p.B_min,
        "T",
    )
    for species in range(2):
        population = context.p.species[species].id
        _minimum(
            out,
            "minimum_density",
            xp.real(xp.asarray(result.fluid[species].n)),
            context.p.species[species].n_min,
            "1/m^3",
            population,
        )
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    m1 = xp.real(xp.asarray(context.op.M1)).astype(xp.float64)
    wall_weights = xp.ones(
        max(len(context.geometry.walls), 0), dtype=xp.float64
    )

    def remember(name, residual, weights, unit, population="", samples=False):
        diagnostics.measure(
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

    remember(
        "quasineutrality",
        result.residuals["quasineutrality"],
        m0,
        "C/m^3",
    )
    remember(
        "electron_density_compatibility",
        result.residuals["electron_density_compatibility"],
        m0,
        "1/(m^3*s)",
    )
    remember(
        "electron_momentum_compatibility",
        result.residuals["electron_momentum_compatibility"],
        xp.tile(m0, 3),
        "kg/(m^2*s^2)",
    )
    remember(
        "current_definition",
        result.residuals["current_definition"],
        m1,
        "A/m^2",
    )
    remember(
        "electron_flux_definition",
        result.residuals["electron_flux_definition"],
        m1,
        "1/(m^2*s)",
    )
    remember(
        "electron_flux_continuity",
        result.residuals["electron_flux_continuity"],
        m0,
        "1/(m^3*s)",
    )
    remember("djj", result.residuals["djj"], m0, "A/m^3")
    remember(
        "ct_divergence",
        result.residuals["ct_divergence"],
        context.geometry.wv,
        "T/(m*s)",
    )
    remember(
        "wall_current",
        _wall_flux_samples(inverse(result.current, ne, context), context),
        wall_weights,
        "A/m^2",
        samples=True,
    )
    for species, name in ((0, "ion"), (1, "electron")):
        remember(
            f"wall_{name}_flux",
            _wall_flux_samples(result.fluid[species].flux, context),
            wall_weights,
            "1/(m^2*s)",
            samples=True,
        )
    remember(
        "wall_field_original_rows",
        result.residuals["wall_field_original_rows"],
        xp.ones(
            result.residuals["wall_field_original_rows"].size,
            dtype=xp.float64,
        ),
        "V/m^3",
        samples=True,
    )
    remember(
        "drift_ohm_residual",
        result.residuals["drift_ohm_residual"],
        m1,
        "V/m",
    )
    remember(
        "electron_cgl_perp",
        result.residuals["electron_cgl_perp"],
        m0,
        "Pa/s",
        samples=True,
    )
    remember(
        "electron_cgl_parallel",
        result.residuals["electron_cgl_parallel"],
        m0,
        "Pa/s",
        samples=True,
    )
    theta_start = len(out)
    remember(
        "theta_consistency",
        result.residuals["theta_consistency"],
        m0,
        "1/s",
        samples=True,
    )
    for record in out[theta_start:]:
        record.hard = False
        record.reason = (
            "convergence diagnostic: full padded divergence of Fe/nf "
            "versus derived theta"
        )

    pressure_tensor = xp.asarray(result.fluid[1].tensor).reshape(
        context.N, 3, 3, nc
    )
    velocity_gradient = xp.asarray(result.fluid[1].gradient).reshape(
        context.N, 3, 3, nc
    )
    pressure_scale = abs(
        _integrate_cells(
            xp.sum(
                xp.abs(pressure_tensor * velocity_gradient), axis=(1, 2)
            ).reshape(-1),
            1,
            context,
        )
    )
    _scaled_scalar(
        out,
        "electron_pressure_adjoint",
        result.residuals["electron_pressure_adjoint"][0].real,
        pressure_scale,
        "W",
        tag,
        context,
        context.p.species[1].id,
    )
    _scaled_scalar(
        out,
        "electron_magnetic_work",
        result.ledger[1].magnetic,
        max(result.ledger[1].magnetic_scale, 1.0e-300),
        "W",
        tag,
        context,
        context.p.species[1].id,
    )
    _scaled_scalar(
        out,
        "pressure_pairing",
        result.residuals["electron_pressure_adjoint"][0].real,
        pressure_scale,
        "W",
        tag,
        context,
        context.p.species[1].id,
    )
    electron_energy = inverse(
        result.cache.E,
        ne,
        context,
        True,
        context.p.ref.electric,
    )
    electric_scale = abs(
        _integrate_edges(
            xp.abs(
                context.p.species[1].charge
                * xp.asarray(result.fluid[1].flux)
                * xp.conj(electron_energy)
            ).reshape(-1),
            context,
        )
    )
    pressure_electric_scale = abs(
        _integrate_edges(
            xp.abs(
                context.p.species[1].charge
                * xp.asarray(result.fluid[1].flux)
                * xp.asarray(result.fluid[1].pressure_electric_field)
            ).reshape(-1),
            context,
        )
    )
    _scaled_scalar(
        out,
        "electron_electric_pairing",
        result.residuals["electron_electric_pairing"][0].real,
        max(pressure_scale, abs(electric_scale)),
        "W",
        tag,
        context,
        context.p.species[1].id,
    )
    _scaled_scalar(
        out,
        "electric_pairing",
        result.residuals["electron_electric_pairing"][0].real,
        max(pressure_scale, abs(electric_scale)),
        "W",
        tag,
        context,
        context.p.species[1].id,
    )
    _scaled_scalar(
        out,
        "electron_pressure_only_pairing",
        result.residuals["electron_pressure_only_pairing"][0].real,
        max(pressure_scale, abs(pressure_electric_scale)),
        "W",
        tag,
        context,
        context.p.species[1].id,
    )
    pressure_truncation_record = diagnostics._record(
        name="electron_pressure_truncation",
        tag=tag,
        unit="W",
        population=context.p.species[1].id,
    )
    pressure_truncation_record.raw = float(
        result.residuals["electron_pressure_truncation"][0].real
    )
    pressure_truncation_record.reference = pressure_scale
    pressure_truncation_record.threshold = (
        4096.0 * sys.float_info.epsilon * pressure_scale
    )
    pressure_truncation_record.available = True
    pressure_truncation_record.passed = (
        abs(pressure_truncation_record.raw)
        <= pressure_truncation_record.threshold
    )
    pressure_truncation_record.hard = False
    pressure_truncation_record.reason = (
        "full padded pressure source versus retained Fourier RHS"
    )
    out.append(pressure_truncation_record)
    for name in ("ion", "electron", "current"):
        sample_scale = _bound_real(
            result.residuals[f"wall_field_reaction_scale_{name}"][0],
            f"{name} absolute scale",
        )
        _scaled_scalar(
            out,
            f"wall_field_reaction_work_{name}",
            result.residuals[
                f"wall_field_reaction_work_{name}"
            ][0].real,
            sample_scale,
            "W",
            tag,
            context,
        )
        record = out[-1]
        record.absolute_final_terms = record.reference
        record.modal_work = record.raw
        record.sample_work = float(
            result.residuals[
                f"wall_field_reaction_sample_work_{name}"
            ][0].real
        )
        record.transform_difference = float(
            result.residuals[
                f"wall_field_reaction_transform_difference_{name}"
            ][0].real
        )
        record.upstream_bound = _bound_real(
            result.residuals[
                f"wall_field_reaction_upstream_bound_{name}"
            ][0],
            f"{name} upstream bound",
        )
        record.compatibility_bound = _bound_real(
            result.residuals[
                f"wall_field_reaction_compatibility_bound_{name}"
            ][0],
            f"{name} compatibility bound",
        )
        record.arithmetic_bound = _bound_real(
            result.residuals[
                f"wall_field_reaction_arithmetic_bound_{name}"
            ][0],
            f"{name} arithmetic bound",
        )
        record.threshold = max(
            record.configured_threshold, record.arithmetic_bound
        )
        record.passed = record.available and (
            abs(record.raw) <= record.threshold
        )
        record.reason = (
            "retained Parseval work; arithmetic_bound includes direct "
            "producer, checked-input compatibility, and bilinear-sum error"
        )
    wall_work_scale = _bound_real(
        result.residuals["wall_field_reaction_scale"][0],
        "species absolute scale",
    )
    _scaled_scalar(
        out,
        "wall_field_reaction_work",
        result.residuals["wall_field_reaction_work"][0].real,
        wall_work_scale,
        "W",
        tag,
        context,
    )
    record = out[-1]
    record.absolute_final_terms = record.reference
    record.modal_work = record.raw
    record.sample_work = float(
        result.residuals["wall_field_reaction_sample_work"][0].real
    )
    record.transform_difference = float(
        result.residuals[
            "wall_field_reaction_transform_difference"
        ][0].real
    )
    record.upstream_bound = _bound_real(
        result.residuals["wall_field_reaction_upstream_bound"][0],
        "species upstream bound",
    )
    record.compatibility_bound = _bound_real(
        result.residuals[
            "wall_field_reaction_compatibility_bound"
        ][0],
        "species compatibility bound",
    )
    record.arithmetic_bound = _bound_real(
        result.residuals["wall_field_reaction_arithmetic_bound"][0],
        "species arithmetic bound",
    )
    record.threshold = max(
        record.configured_threshold, record.arithmetic_bound
    )
    record.passed = record.available and abs(record.raw) <= record.threshold
    record.reason = (
        "aggregate retained Parseval work with component and "
        "species-addition bounds"
    )
    _truth(
        out,
        "memory_bound",
        int(context.p.memory_limit) > 0
        and core.estimated_peak_bytes(context)
        <= int(context.p.memory_limit),
        tag,
    )
    hermitian = all(
        diagnostics._hermitian_defect(context, value, entities)
        <= 4096.0
        * sys.float_info.epsilon
        * max(float(xp.max(xp.abs(value))), 1.0e-300)
        for value, entities in (
            (result.rhs.fluid[0].n, nc),
            (result.rhs.fluid[0].mom, 3 * nc),
            (result.rhs.fluid[0].perp, nc),
            (result.rhs.fluid[1].n, nc),
            (result.rhs.fluid[1].mom, 3 * nc),
            (result.rhs.fluid[1].perp, nc),
            (result.rhs.fluid[1].parallel, nc),
            (result.rhs.B, context.op.MB.size),
        )
    )
    _truth(out, "hermitian_rhs", hermitian, tag)
    for axis, component in enumerate(range(3)):
        _convergence_scalar(
            out,
            f"momentum_balance_{'xyz'[axis]}",
            result.momentum.balance_defect[component],
            "kg*m/s^2",
            tag,
            max(
                abs(result.momentum.pdot[component]),
                abs(result.momentum.outward[component]),
                1.0e-300,
            ),
            "Pdot plus actual wall traction; spatial closure not assumed",
        )
    _convergence_scalar(
        out,
        "angular_momentum_balance",
        result.momentum.angular_balance_defect,
        "kg*m^2/s^2",
        tag,
        max(
            abs(result.momentum.angular_rate),
            abs(result.momentum.outward_torque),
            1.0e-300,
        ),
        "angular Pdot plus actual wall traction torque",
    )
    for axis, component in enumerate(range(3)):
        _convergence_scalar(
            out,
            f"momentum_projection_reaction_{'xyz'[axis]}",
            result.momentum.wall_reaction[component],
            "kg*m/s^2",
            tag,
            (
                context.p.species[0].mass
                * context.p.ref.ni
                * context.p.ref.velocity
                / context.p.ref.time
                * context.p.ref.length**3
            ),
            "ion normal-momentum projection reaction; not physical traction",
        )
    _truth(
        out,
        "wall_correction_rank_checked",
        all(record.rank == len(record.selected_rows) for record in result.cache.wall_correction.modes),
        tag,
    )
    _unavailable(
        out,
        "electron_inertia",
        tag,
        "not_applicable: electron mass is metadata only",
    )
    _unavailable(
        out,
        "electron_kinetic_energy",
        tag,
        "not_applicable: electron velocity is constrained, not an energy DOF",
    )
    _unavailable(
        out,
        "electron_kinetic_transport",
        tag,
        "not_applicable: boundary transport contains ion kinetic energy only",
    )
    _unavailable(
        out,
        "spectral_kinetic_transfer",
        tag,
        "not_applicable: no electron kinetic-energy channel exists",
    )
    _unavailable(
        out,
        "electron_mechanical_stress",
        tag,
        "not_applicable: electron momentum is a compatibility output",
    )
    _unavailable(
        out,
        "electric_maxwell_stress",
        tag,
        "not_applicable: displacement current and longitudinal field are omitted",
    )
    _unavailable(
        out,
        "finite_charge_gauss",
        tag,
        "not_applicable: exact quasineutrality replaces finite-charge Gauss",
    )
    _unavailable(
        out,
        "longitudinal_electric_energy",
        tag,
        "not_applicable: longitudinal displacement current is omitted",
    )
    diagnostics.gate(out)


def stage_requirements(context, tag):
    """Return the massless branch's failure-evidence check shapes."""
    records = [
        diagnostics._record(name="finite", tag=tag),
        diagnostics._record(name="source_identity", tag=tag),
        diagnostics._record(name="model_massless_quasineutral", tag=tag),
    ]
    units = {
        "quasineutrality": "C/m^3",
        "electron_density_compatibility": "1/(m^3*s)",
        "electron_momentum_compatibility": "kg/(m^2*s^2)",
        "current_definition": "A/m^2",
        "electron_flux_definition": "1/(m^2*s)",
        "electron_flux_continuity": "1/(m^3*s)",
        "djj": "A/m^3",
        "ct_divergence": "T/(m*s)",
        "wall_current": "A/m^2",
        "wall_ion_flux": "1/(m^2*s)",
        "wall_electron_flux": "1/(m^2*s)",
        "wall_field_original_rows": "V/m^3",
        "wall_field_reaction_work": "W",
        "drift_ohm_residual": "V/m",
        "electron_cgl_perp": "Pa/s",
        "electron_cgl_parallel": "Pa/s",
        "electron_pressure_adjoint": "W",
        "electron_electric_pairing": "W",
        "electron_pressure_only_pairing": "W",
        "electron_pressure_truncation": "W",
        "electron_magnetic_work": "W",
        "wall_field_reaction_work_ion": "W",
        "wall_field_reaction_work_electron": "W",
        "wall_field_reaction_work_current": "W",
        "momentum_balance_x": "kg*m/s^2",
        "momentum_balance_y": "kg*m/s^2",
        "momentum_balance_z": "kg*m/s^2",
        "angular_momentum_balance": "kg*m^2/s^2",
        "momentum_projection_reaction_x": "kg*m/s^2",
        "momentum_projection_reaction_y": "kg*m/s^2",
        "momentum_projection_reaction_z": "kg*m/s^2",
        "pressure_pairing": "W",
        "electric_pairing": "W",
        "theta_consistency": "1/s",
    }
    for name, unit in units.items():
        records.append(diagnostics._record(name=name, tag=tag, unit=unit))
    return records


def _quadrature(context, quadrature, midpoint_state, midpoint):
    if quadrature is None:
        return [(midpoint_state, midpoint, float(context.p.dt))]
    return [(state, stage, float(weight)) for state, stage, weight in quadrature]


def _sum_ledger(quadrature, species, name):
    return sum(
        weight * float(getattr(stage.ledger[species], name))
        for _, stage, weight in quadrature
    )


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
    """Check one accepted explicit interval without finite-mass field logic."""
    if prescribed_source is not None:
        raise Failure(
            "ManufacturedSource",
            "massless independent electron n/momentum sources are unsupported",
        )
    xp = _xp(context)
    quadrature = _quadrature(context, quadrature, midpoint_state, midpoint)
    dt = float(context.p.dt)
    if abs(sum(weight for _, _, weight in quadrature) - dt) > (
        64.0 * sys.float_info.epsilon * max(abs(dt), 1.0e-300)
    ):
        raise Failure("Time", "quadrature weights do not span dt")
    out = []
    interval_tag = SimpleNamespace(**vars(end.tag))
    interval_tag.stage = "interval"
    nc = context.op.Nc
    ne = context.op.Ne
    m0 = xp.real(xp.asarray(context.op.M0)).astype(xp.float64)
    interval = SimpleNamespace(
        source_id=end.source_id,
        trial=end.tag.trial,
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
                source_id=stage.source_id,
                weight=weight,
                stage=stage.tag.stage,
            )
            for _, stage, weight in quadrature
        ],
        accepted=False,
        applicability=_applicability(),
    )
    charge_change = xp.zeros(_modes(context) * nc, dtype=xp.complex128)
    boundary_number = [0.0, 0.0]
    boundary_kinetic = [0.0, 0.0]
    boundary_internal_perp = [0.0, 0.0]
    boundary_internal_parallel = [0.0, 0.0]
    boundary_total_energy = 0.0
    boundary_field_energy = 0.0
    for _, stage, weight in quadrature:
        for species in range(2):
            boundary_number[species] += (
                weight * stage.boundary.number[species]
            )
            boundary_kinetic[species] += (
                weight * stage.boundary.kinetic[species]
            )
            boundary_internal_perp[species] += (
                weight * stage.boundary.internal_perp[species]
            )
            boundary_internal_parallel[species] += (
                weight * stage.boundary.internal_parallel[species]
            )
        boundary_total_energy += weight * stage.boundary.total_energy
        boundary_field_energy += weight * stage.boundary.field_energy

    def scalar(name, value, unit, population=""):
        diagnostics.scalar_check(
            out, name, value, unit, interval_tag, context, population
        )

    for species in range(2):
        integrated_flux = sum(
            weight * xp.asarray(stage.fluid[species].flux_hat)
            for _, stage, weight in quadrature
        )
        integrated_divergence = sum(
            weight * xp.asarray(stage.fluid[species].divergence_hat)
            for _, stage, weight in quadrature
        )
        interval.number_flux[species] = integrated_flux
        interval.current[species] = (
            integrated_flux * context.p.species[species].charge
        )
        ndiff = (xp.asarray(right.fluid[species].n) - xp.asarray(left.fluid[species].n)) / dt
        diagnostics.measure(
            out,
            "species_continuity",
            ndiff + integrated_divergence / dt,
            m0,
            "1/(m^3*s)",
            interval_tag,
            context,
            context.p.species[species].id,
        )
        charge_change += context.p.species[species].charge * ndiff
        scalar(
            "total_number_balance",
            end.totals.number[species]
            - base.totals.number[species]
            + boundary_number[species],
            "1",
            context.p.species[species].id,
        )
        ledger = SimpleNamespace(
            **{
                name: _sum_ledger(quadrature, species, name)
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
                )
            }
        )
        if species == 0:
            scalar(
                "kinetic_energy",
                end.totals.kinetic[0]
                - base.totals.kinetic[0]
                + boundary_kinetic[0]
                - (
                    ledger.electric_edge
                    + ledger.pressure_kinetic
                    + ledger.wall
                    + ledger.magnetic
                ),
                "J",
                context.p.species[0].id,
            )
            scalar(
                "ion_internal",
                end.totals.ion_internal
                - base.totals.ion_internal
                + boundary_internal_perp[0]
                - ledger.perp_rate,
                "J",
            )
        else:
            scalar(
                "electron_perp",
                end.totals.electron_perp
                - base.totals.electron_perp
                + boundary_internal_perp[1]
                - ledger.perp_rate,
                "J",
            )
            scalar(
                "electron_parallel",
                end.totals.electron_parallel
                - base.totals.electron_parallel
                + boundary_internal_parallel[1]
                - ledger.parallel_rate,
                "J",
            )
    average_current = sum(
        weight * xp.asarray(stage.current)
        for _, stage, weight in quadrature
    ) / dt
    dj_current = mode_apply(context.op.DJ, average_current, context.compute)
    diagnostics.measure(
        out,
        "djj_interval",
        dj_current,
        m0,
        "A/m^3",
        interval_tag,
        context,
    )
    wall_current = _wall_flux_samples(
        inverse(average_current, ne, context), context
    )
    wall_weights = xp.ones(
        len(context.geometry.walls), dtype=xp.float64
    )
    diagnostics.measure(
        out,
        "wall_current_interval",
        wall_current,
        wall_weights,
        "A/m^2",
        interval_tag,
        context,
        samples=True,
    )
    for species in range(2):
        average_flux = sum(
            weight * xp.asarray(stage.fluid[species].flux)
            for _, stage, weight in quadrature
        ) / dt
        diagnostics.measure(
            out,
            f"wall_{'ion' if species == 0 else 'electron'}_flux_interval",
            _wall_flux_samples(average_flux, context),
            wall_weights,
            "1/(m^2*s)",
            interval_tag,
            context,
            context.p.species[species].id,
            samples=True,
        )
    diagnostics.measure(
        out,
        "quasineutrality_interval",
        charge_change,
        m0,
        "C/(m^3*s)",
        interval_tag,
        context,
    )
    pb = 0.0
    for state, stage, weight in quadrature:
        mb = xp.real(xp.asarray(context.op.MB)).astype(xp.float64)
        state_b = xp.asarray(state.B)
        rhs_b = xp.asarray(stage.rhs.B)
        power = (
            2.0
            * math.pi
            * xp.sum(
                xp.tile(mb, state_b.size // mb.size)
                * xp.real(xp.conj(state_b) * rhs_b)
            ).item()
            / context.p.mu0
        )
        pb += weight * float(complex(power).real)
    scalar(
        "field_energy_B_interval",
        end.totals.magnetic - base.totals.magnetic - pb,
        "J",
    )
    field_exchange = (
        pb
        + _sum_ledger(quadrature, 0, "electric_edge")
        + _sum_ledger(quadrature, 1, "electric_edge")
        + boundary_field_energy
    ) / dt
    scalar("field_exchange_interval", field_exchange, "W")
    scalar(
        "total_energy_interval",
        end.totals.energy
        - base.totals.energy
        + boundary_total_energy,
        "J",
    )
    total_charge = sum(
        context.p.species[species].charge
        * (
            end.totals.number[species]
            - base.totals.number[species]
            + boundary_number[species]
        )
        for species in range(2)
    )
    scalar("total_charge_balance_interval", total_charge, "C")
    for axis, d in zip(("x", "y", "z"), range(3)):
        integrated_traction = sum(
            weight * stage.momentum.outward[d]
            for _, stage, weight in quadrature
        )
        defect = (
            end.momentum.value[d]
            - base.momentum.value[d]
            + integrated_traction
        )
        _convergence_scalar(
            out,
            f"momentum_balance_interval_{axis}",
            defect,
            "kg*m/s",
            interval_tag,
            max(
                abs(end.momentum.value[d] - base.momentum.value[d]),
                abs(integrated_traction),
                1.0e-300,
            ),
            "endpoint momentum change plus integrated wall traction",
        )
    angular_traction = sum(
        weight * stage.momentum.outward_torque
        for _, stage, weight in quadrature
    )
    _convergence_scalar(
        out,
        "angular_momentum_interval",
        end.momentum.angular
        - base.momentum.angular
        + angular_traction,
        "kg*m^2/s",
        interval_tag,
        max(
            abs(end.momentum.angular - base.momentum.angular),
            abs(angular_traction),
            1.0e-300,
        ),
        "endpoint angular momentum plus integrated wall torque",
    )
    bchange = xp.asarray(right.B) - xp.asarray(left.B)
    ce = sum(
        weight * mode_apply(context.op.C, stage.cache.E, context.compute)
        for _, stage, weight in quadrature
    )
    diagnostics.measure(
        out,
        "ct_update",
        bchange / dt + ce / dt,
        context.op.MB,
        "T/s",
        interval_tag,
        context,
    )
    diagnostics.measure(
        out,
        "divB_update",
        mode_apply(context.op.DB, bchange, context.compute),
        context.geometry.wv,
        "T/m",
        interval_tag,
        context,
    )
    for _, stage, _ in quadrature:
        diagnostics.gate(stage.checks)
    result = SimpleNamespace(
        checks=out,
        totals=end.totals,
        momentum=end.momentum,
        interval=interval,
        applicability=_applicability(),
    )
    diagnostics.gate(out)
    return result
