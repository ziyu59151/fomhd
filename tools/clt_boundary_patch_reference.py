"""Bounded true-intercept triangle closure in cylindrical R-Z coordinates.

Geometry and fields use SI units and physical ``(R, Z)`` coordinates. Measures
and line integrals are per radian where the axisymmetric azimuth is implicit.
The module is a local CPU oracle only; it does not assemble a global solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

_Array = np.ndarray


def _readonly(value) -> _Array:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _quadrature_order(order: int) -> int:
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("quadrature order must be an integer >= 2")
    return order


def _gauss_unit(order: int) -> tuple[_Array, _Array]:
    order = _quadrature_order(order)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


@dataclass(frozen=True, slots=True)
class TriangleGeometry:
    """Immutable SI triangle ``O=(r0,0), A=(r0+a,0), B=(r0,b)``."""

    r0: float
    a: float
    b: float

    length_unit: ClassVar[str] = "m"
    measure_unit: ClassVar[str] = "m^3/rad"

    def __post_init__(self) -> None:
        for name in ("r0", "a", "b"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))

    @property
    def area(self) -> float:
        """Area in m^2."""
        return 0.5 * self.a * self.b

    @property
    def volume(self) -> float:
        """Axisymmetric volume per radian, in m^3/rad."""
        return 0.5 * self.r0 * self.a * self.b + self.a**2 * self.b / 6.0

    @property
    def wall_length(self) -> float:
        """Physical wall length in m."""
        return math.hypot(self.a, self.b)

    @property
    def wall_normal(self) -> _Array:
        """Outward unit wall normal in the R-Z plane."""
        return _readonly(np.array((self.b, self.a)) / self.wall_length)

    @property
    def wall_tangent(self) -> _Array:
        """Unit tangent oriented A to B."""
        return _readonly(np.array((-self.a, self.b)) / self.wall_length)

    def signed_normal_distance(self, r, z):
        """Return the signed distance from the physical wall; interior is <0."""
        return (
            self.b * (np.asarray(r) - self.r0)
            + self.a * np.asarray(z)
            - self.a * self.b
        ) / self.wall_length

    def wall_tangent_coordinate(self, r, z):
        """Return projection from A onto the A-to-B wall tangent, in m."""
        return (
            -self.a * (np.asarray(r) - (self.r0 + self.a)) + self.b * np.asarray(z)
        ) / self.wall_length


def _field_value(field, r, z, name: str) -> _Array:
    r_array, z_array = np.broadcast_arrays(
        np.asarray(r, dtype=np.float64), np.asarray(z, dtype=np.float64)
    )
    try:
        value = np.asarray(field(r_array, z_array), dtype=np.float64)
    except Exception as error:
        raise ValueError(f"{name} callable failed") from error
    if value.shape[-1:] != (2,):
        raise ValueError(f"{name} callable must return a trailing (R,Z) axis")
    try:
        result = np.broadcast_to(value, (*r_array.shape, 2))
    except ValueError as error:
        raise ValueError(f"{name} callable returned an incompatible shape") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} callable returned non-finite data")
    return result


def _scalar_value(field, r, z, name: str) -> _Array:
    r_array, z_array = np.broadcast_arrays(
        np.asarray(r, dtype=np.float64), np.asarray(z, dtype=np.float64)
    )
    try:
        value = np.asarray(field(r_array, z_array), dtype=np.float64)
    except Exception as error:
        raise ValueError(f"{name} callable failed") from error
    try:
        result = np.broadcast_to(value, r_array.shape)
    except ValueError as error:
        raise ValueError(f"{name} callable returned an incompatible shape") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} callable returned non-finite data")
    return result


@dataclass(frozen=True, slots=True)
class ElectricTrace:
    """Physical edge integrals of one poloidal vector field."""

    e_r: float
    e_z: float
    wall_line_integral: float
    oa_line_integral: float
    ob_line_integral: float
    quadrature_order: int
    area: float

    @property
    def axis_coefficients(self) -> _Array:
        return _readonly((self.e_r, self.e_z))

    @property
    def coefficients(self) -> _Array:
        return _readonly((self.e_r, self.e_z, self.wall_line_integral))

    @property
    def circulation(self) -> float:
        """CCW circulation for the basis convention, in field*length."""
        return self.oa_line_integral - self.ob_line_integral + self.wall_line_integral

    @property
    def curl_phi(self) -> float:
        """Physical toroidal curl component from Stokes' theorem."""
        return -self.circulation / self.area


def sample_electric(
    geometry: TriangleGeometry, field, *, order: int = 8
) -> ElectricTrace:
    """Sample a physical ``(R,Z)`` field along OA, OB, and A-to-B.

    ``e_r`` and ``e_z`` are axis-component line averages. ``wall_line_integral``
    is the A-to-B tangential line integral, including its physical length.
    Nonzero wall moments are supported for manufactured controls; the physical
    ideal-wall condition ``E_t=0`` is a separate production contract.
    """
    nodes, weights = _gauss_unit(order)
    edge_r = {
        "oa": geometry.r0 + geometry.a * nodes,
        "ob": np.full_like(nodes, geometry.r0),
        "ab": geometry.r0 + geometry.a * (1.0 - nodes),
    }
    edge_z = {
        "oa": np.zeros_like(nodes),
        "ob": geometry.b * nodes,
        "ab": geometry.b * nodes,
    }
    sampled = {
        edge: _field_value(field, edge_r[edge], edge_z[edge], "field")
        for edge in ("oa", "ob", "ab")
    }
    oa_integral = geometry.a * float(np.dot(weights, sampled["oa"][:, 0]))
    ob_integral = geometry.b * float(np.dot(weights, sampled["ob"][:, 1]))
    wall_integral = float(
        np.dot(
            weights,
            -geometry.a * sampled["ab"][:, 0] + geometry.b * sampled["ab"][:, 1],
        )
    )
    return ElectricTrace(
        e_r=oa_integral / geometry.a,
        e_z=ob_integral / geometry.b,
        wall_line_integral=wall_integral,
        oa_line_integral=oa_integral,
        ob_line_integral=ob_integral,
        quadrature_order=_quadrature_order(order),
        area=geometry.area,
    )


@dataclass(frozen=True, slots=True)
class VertexSamples:
    """Scalar physical samples at O, A, and B."""

    o: float
    a: float
    b: float


def sample_potential(geometry: TriangleGeometry, potential) -> VertexSamples:
    """Read a physical potential callable at the three shared vertices."""
    r = np.array((geometry.r0, geometry.r0 + geometry.a, geometry.r0))
    z = np.array((0.0, 0.0, geometry.b))
    value = _scalar_value(potential, r, z, "potential")
    return VertexSamples(*(float(item) for item in value))


def p1_gradient(geometry: TriangleGeometry, samples: VertexSamples) -> _Array:
    """Gradient of the unique affine field through the three vertex samples."""
    return _readonly(
        (
            (samples.a - samples.o) / geometry.a,
            (samples.b - samples.o) / geometry.b,
        )
    )


def basis(geometry: TriangleGeometry, r, z) -> _Array:
    """Evaluate ``[wR, wZ, wL]`` with trailing basis and component axes."""
    r_array, z_array = np.broadcast_arrays(
        np.asarray(r, dtype=np.float64), np.asarray(z, dtype=np.float64)
    )
    x = r_array - geometry.r0
    value = np.empty((*r_array.shape, 3, 2), dtype=np.float64)
    value[..., 0, 0] = 1.0 - z_array / geometry.b
    value[..., 0, 1] = x / geometry.b
    value[..., 1, 0] = z_array / geometry.a
    value[..., 1, 1] = 1.0 - x / geometry.a
    value[..., 2, 0] = -z_array / (geometry.a * geometry.b)
    value[..., 2, 1] = x / (geometry.a * geometry.b)
    return value


def reconstruct_field(geometry: TriangleGeometry, coefficients, r, z) -> _Array:
    """Reconstruct a physical R-Z field from two axis or three lifted DOFs."""
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    if coefficient_array.shape[-1:] == (2,):
        coefficient_array = np.concatenate(
            (coefficient_array, np.zeros((*coefficient_array.shape[:-1], 1))),
            axis=-1,
        )
    if coefficient_array.shape[-1:] != (3,):
        raise ValueError("coefficients must end in 2 axis or 3 lifted values")
    r_array, z_array = np.broadcast_arrays(
        np.asarray(r, dtype=np.float64), np.asarray(z, dtype=np.float64)
    )
    try:
        coefficient_values = np.broadcast_to(coefficient_array, (*r_array.shape, 3))
    except ValueError as error:
        raise ValueError("coefficients are incompatible with sample shape") from error
    return np.einsum(
        "...c,...cp->...p",
        coefficient_values,
        basis(geometry, r_array, z_array),
    )


@dataclass(frozen=True, slots=True)
class Metric:
    """Small local cylindrical Gram matrix."""

    matrix: _Array
    includes_wall: bool
    quadrature_order: int

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("metric matrix must be square")
        object.__setattr__(self, "matrix", _readonly(matrix))

    @property
    def is_spd(self) -> bool:
        if not np.all(np.isfinite(self.matrix)):
            return False
        if not np.allclose(self.matrix, self.matrix.T, rtol=0.0, atol=1e-14):
            return False
        return bool(np.min(np.linalg.eigvalsh(self.matrix)) > 0.0)

    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self.matrix))

    def apply(self, value) -> _Array:
        array = np.asarray(value, dtype=np.float64)
        if array.shape[-1:] != (self.matrix.shape[0],):
            raise ValueError("metric action has the wrong trailing dimension")
        return array @ self.matrix.T

    def adjoint(self, value) -> _Array:
        array = np.asarray(value, dtype=np.float64)
        if array.shape[-1:] != (self.matrix.shape[0],):
            raise ValueError("metric adjoint has the wrong trailing dimension")
        return array @ self.matrix

    def solve(self, value) -> _Array:
        array = np.asarray(value, dtype=np.float64)
        count = self.matrix.shape[0]
        if array.shape[-1:] != (count,):
            raise ValueError("metric solve has the wrong trailing dimension")
        flat = array.reshape(-1, count).T
        result = np.linalg.solve(self.matrix, flat).T
        return result.reshape(array.shape)

    def pair(self, left, right):
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
        count = self.matrix.shape[0]
        if left_array.shape[-1:] != (count,) or right_array.shape[-1:] != (count,):
            raise ValueError("metric pairing has the wrong trailing dimension")
        return np.einsum("...i,ij,...j->...", left_array, self.matrix, right_array)


def homogeneous_metric(geometry: TriangleGeometry) -> Metric:
    """Exact homogeneous 2-by-2 electric Gram in m^3/rad."""
    a, b, r0 = geometry.a, geometry.b, geometry.r0
    m11 = r0 * (a * b / 4.0 + a**3 / (12.0 * b)) + a**2 * b / 10.0 + a**4 / (20.0 * b)
    m22 = r0 * (a * b / 4.0 + b**3 / (12.0 * a)) + a**2 * b / 20.0 + b**3 / 60.0
    m12 = r0 * (a**2 + b**2) / 12.0 + a * b**2 / 40.0 + a**3 / 30.0
    return Metric(np.array(((m11, m12), (m12, m22))), False, 0)


def _volume_points(
    geometry: TriangleGeometry, order: int
) -> tuple[_Array, _Array, _Array]:
    nodes, weights = _gauss_unit(order)
    s, t = np.meshgrid(nodes, nodes, indexing="ij")
    ws, wt = np.meshgrid(weights, weights, indexing="ij")
    r = geometry.r0 + geometry.a * s
    z = geometry.b * (1.0 - s) * t
    weight = geometry.a * geometry.b * (1.0 - s) * r * ws * wt
    return r, z, weight


def lifted_metric(geometry: TriangleGeometry, *, order: int = 8) -> Metric:
    """Numerically integrated 3-by-3 Gram including the wall line lift."""
    order = _quadrature_order(order)
    r, z, weight = _volume_points(geometry, order)
    values = basis(geometry, r, z)
    gram = np.einsum(
        "ij,ijpc,ijqc->pq",
        weight,
        values,
        values,
        optimize=True,
    )
    return Metric(gram, True, order)


def normal_lumped_metric(geometry: TriangleGeometry) -> _Array:
    """Return ``D_ii=(M*v)_i/v_i`` with ``v=(b,a)``."""
    matrix = homogeneous_metric(geometry).matrix
    trace = np.array((geometry.b, geometry.a))
    return _readonly(np.diag((matrix @ trace) / trace))


@dataclass(frozen=True, slots=True)
class PowerPair:
    """Power per radian from coefficient and direct physical-field samples."""

    electric: ElectricTrace
    current: ElectricTrace
    power: float
    field_power: float
    reconstruction_power_error: float
    includes_wall: bool


def paired_power(
    geometry: TriangleGeometry,
    electric,
    current,
    *,
    order: int = 8,
    include_wall_moment: bool = True,
) -> PowerPair:
    """Pair two independently sampled physical fields through the lifted Gram."""
    order = _quadrature_order(order)
    e_trace = sample_electric(geometry, electric, order=order)
    j_trace = sample_electric(geometry, current, order=order)
    e_coefficients = np.array(e_trace.coefficients)
    j_coefficients = np.array(j_trace.coefficients)
    if not include_wall_moment:
        e_coefficients[2] = 0.0
        j_coefficients[2] = 0.0
    metric = lifted_metric(geometry, order=order)
    power = float(metric.pair(e_coefficients, j_coefficients))

    r, z, weight = _volume_points(geometry, order)
    e_value = _field_value(electric, r, z, "electric")
    j_value = _field_value(current, r, z, "current")
    field_power = float(np.sum(weight * np.sum(e_value * j_value, axis=-1)))
    return PowerPair(
        electric=e_trace,
        current=j_trace,
        power=power,
        field_power=field_power,
        reconstruction_power_error=abs(power - field_power),
        includes_wall=include_wall_moment,
    )


@dataclass(frozen=True, slots=True)
class Divergence:
    """Independent boundary-flux estimate of cylindrical divergence."""

    axis_oa_flux: float
    axis_ob_flux: float
    wall_flux: float
    total_flux: float
    volume: float
    average_divergence: float


def magnetic_divergence(
    geometry: TriangleGeometry, field, *, order: int = 8
) -> Divergence:
    """Integrate the physical B normal flux over all three edges."""
    nodes, weights = _gauss_unit(order)
    edges = (
        (
            geometry.r0 + geometry.a * nodes,
            np.zeros_like(nodes),
            np.array((0.0, -1.0)),
            geometry.a,
        ),
        (
            np.full_like(nodes, geometry.r0),
            geometry.b * nodes,
            np.array((-1.0, 0.0)),
            geometry.b,
        ),
        (
            geometry.r0 + geometry.a * (1.0 - nodes),
            geometry.b * nodes,
            np.array(geometry.wall_normal),
            geometry.wall_length,
        ),
    )
    fluxes = []
    for r, z, normal, length in edges:
        value = _field_value(field, r, z, "magnetic field")
        fluxes.append(float(np.dot(weights, (value @ normal) * r * length)))
    total = float(sum(fluxes))
    return Divergence(
        axis_oa_flux=fluxes[0],
        axis_ob_flux=fluxes[1],
        wall_flux=fluxes[2],
        total_flux=total,
        volume=geometry.volume,
        average_divergence=total / geometry.volume,
    )


@dataclass(frozen=True, slots=True)
class AffinePressure:
    """Affine pressure reconstructed from the three shared vertex samples."""

    o: float
    r0: float
    gradient: _Array

    def __post_init__(self) -> None:
        gradient = np.asarray(self.gradient, dtype=np.float64)
        if gradient.shape != (2,):
            raise ValueError("affine pressure gradient must have shape (2,)")
        object.__setattr__(self, "gradient", _readonly(gradient))

    def value(self, r, z) -> _Array:
        r_array, z_array = np.broadcast_arrays(
            np.asarray(r, dtype=np.float64), np.asarray(z, dtype=np.float64)
        )
        return (
            self.o + self.gradient[0] * (r_array - self.r0) + self.gradient[1] * z_array
        )


def sample_pressure(geometry: TriangleGeometry, pressure) -> VertexSamples:
    """Sample a pressure callable at the shared triangle vertices."""
    return sample_potential(geometry, pressure)


def affine_pressure(
    geometry: TriangleGeometry, samples: VertexSamples
) -> AffinePressure:
    """Construct the affine pressure and its constant physical gradient."""
    return AffinePressure(
        o=samples.o,
        r0=geometry.r0,
        gradient=p1_gradient(geometry, samples),
    )


def pressure_gradient_force(
    geometry: TriangleGeometry,
    pressure: AffinePressure,
    *,
    order: int = 8,
) -> _Array:
    """Pressure force on the fluid, ``-integral grad(p) R dR dZ``, in N/rad."""
    _, _, weight = _volume_points(geometry, order)
    return -pressure.gradient * float(np.sum(weight))


def _pressure_on_wall(
    geometry: TriangleGeometry, pressure: AffinePressure, order: int
) -> tuple[_Array, _Array, _Array]:
    nodes, weights = _gauss_unit(order)
    r = geometry.r0 + geometry.a * (1.0 - nodes)
    z = geometry.b * nodes
    return r, pressure.value(r, z), weights


def pressure_wall_reaction(
    geometry: TriangleGeometry,
    pressure: AffinePressure,
    *,
    order: int = 8,
) -> _Array:
    """Fluid force on wall, ``+integral p n R ds`` on AB, in N/rad.

    The wall reaction on the fluid is the negative of this vector.
    """
    r, value, weights = _pressure_on_wall(geometry, pressure, _quadrature_order(order))
    scale = geometry.wall_length * float(np.dot(weights, value * r))
    return _readonly(scale * geometry.wall_normal)


@dataclass(frozen=True, slots=True)
class PressureGradientIdentity:
    """R-weighted ``+grad(p)`` identity and fluid-on-boundary reactions."""

    gradient_pairing: float
    pressure_divergence: float
    boundary_pairing: float
    defect: float
    wall_reaction: _Array
    support_reaction: _Array


def pressure_gradient_identity(
    geometry: TriangleGeometry,
    pressure: AffinePressure,
    velocity,
    divergence,
    *,
    order: int = 8,
) -> PressureGradientIdentity:
    """Check ``integral grad(p).u + integral p div(u) = boundary``.

    This intentionally retains the positive gradient pairing. It is not the
    force on the fluid; use :func:`pressure_gradient_force` for that quantity.
    """
    order = _quadrature_order(order)
    r, z, weight = _volume_points(geometry, order)
    velocity_value = _field_value(velocity, r, z, "velocity")
    divergence_value = _scalar_value(divergence, r, z, "divergence")
    pressure_value = pressure.value(r, z)
    gradient_pairing = float(
        np.sum(
            weight
            * (
                pressure.gradient[0] * velocity_value[..., 0]
                + pressure.gradient[1] * velocity_value[..., 1]
            )
        )
    )
    pressure_divergence = float(np.sum(weight * pressure_value * divergence_value))

    nodes, weights = _gauss_unit(order)
    edges = (
        (
            geometry.r0 + geometry.a * nodes,
            np.zeros_like(nodes),
            np.array((0.0, -1.0)),
            geometry.a,
        ),
        (
            np.full_like(nodes, geometry.r0),
            geometry.b * nodes,
            np.array((-1.0, 0.0)),
            geometry.b,
        ),
        (
            geometry.r0 + geometry.a * (1.0 - nodes),
            geometry.b * nodes,
            np.array(geometry.wall_normal),
            geometry.wall_length,
        ),
    )
    boundary = 0.0
    reactions = []
    for r_edge, z_edge, normal, length in edges:
        edge_velocity = _field_value(velocity, r_edge, z_edge, "velocity")
        edge_pressure = pressure.value(r_edge, z_edge)
        edge_weight = weights * r_edge * length
        boundary += float(
            np.sum(edge_weight * edge_pressure * (edge_velocity @ normal))
        )
        reactions.append(float(np.sum(edge_weight * edge_pressure)) * normal)
    support_reaction = reactions[0] + reactions[1]
    return PressureGradientIdentity(
        gradient_pairing=gradient_pairing,
        pressure_divergence=pressure_divergence,
        boundary_pairing=boundary,
        defect=gradient_pairing + pressure_divergence - boundary,
        wall_reaction=_readonly(reactions[2]),
        support_reaction=_readonly(support_reaction),
    )
