"""Fixed-ion RK4 field windows with batched residuals and affine tree correction.

The arrays are SI complex128 on the caller's NumPy/CuPy backend. A leading
dimension indexes independent microsteps, not accepted physical transactions.
The caller owns the stream and inputs; all returned storage is trial-local.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .types import Failure


def operator_prefix(power: Callable, offsets: Any, start: Any, *, xp: Any) -> Any:
    """All endpoints for a constant reference action, with a true up/down tree.

    ``power(n, batch)`` must apply the reference n-step propagation to an
    independent vector batch. Its cost is deliberately not assumed constant.
    Only tail leaves are padded, so they cannot affect any real prefix.
    """
    if offsets.ndim < 2 or offsets.shape[0] < 1 or offsets.shape[1:] != start.shape:
        raise Failure("Shape", "invalid field-prefix extents")
    if offsets.dtype != xp.complex128 or start.dtype != xp.complex128:
        raise Failure("Backend", "field-prefix storage must be complex128")
    count = offsets.shape[0]
    padded = 1 << (count-1).bit_length()
    leaves = xp.zeros((padded,) + start.shape, dtype=xp.complex128)
    leaves[:count] = offsets
    levels = [leaves]
    length = 1
    while levels[-1].shape[0] > 1:
        children = levels[-1]
        levels.append(power(length, children[::2]) + children[1::2])
        length *= 2
    left = start[None].copy()
    for children in reversed(levels[:-1]):
        length //= 2
        next_left = xp.empty_like(children)
        next_left[::2] = left
        next_left[1::2] = power(length, left) + children[::2]
        left = next_left
    return (power(1, left[:count]) + offsets).copy()


class PolynomialReference:
    """Truncated powers of a frozen RK4 propagator without a field matrix.

    Store scalar coefficients of R4(hL)^n, truncated at max_degree. Horner
    evaluation has degree-dependent serial derivative depth; every derivative
    action (which may itself use two physical RHS batches) is counted.
    """

    def __init__(self, action: Callable, h: float, *, xp: Any, max_degree: int = 30):
        if not math.isfinite(h) or h <= 0 or not 1 <= max_degree <= 30:
            raise ValueError("invalid frozen-polynomial controls")
        self.action, self.h, self.xp = action, h, xp
        self.max_degree = max_degree
        self.coefficients: dict[int, np.ndarray] = {}
        self.action_calls = 0
        self.action_vectors = 0
        self.maximum_degree = 0

    def _coefficients(self, length: int) -> np.ndarray:
        if length < 0:
            raise ValueError("negative segment length")
        if length not in self.coefficients:
            result = np.array([1.])
            base = np.array([1., 1., .5, 1/6, 1/24])
            remaining = length
            while remaining:
                if remaining & 1:
                    result = np.convolve(result, base)[:self.max_degree+1]
                remaining >>= 1
                if remaining:
                    base = np.convolve(base, base)[:self.max_degree+1]
            self.coefficients[length] = result
        return self.coefficients[length]

    def power(self, length: int, vectors: Any) -> Any:
        coefficients = self._coefficients(length)
        self.maximum_degree = max(self.maximum_degree, len(coefficients)-1)
        result = coefficients[-1]*vectors
        for coefficient in coefficients[-2::-1]:
            result = self.h*self.action(result) + coefficient*vectors
            self.action_calls += 1
            self.action_vectors += vectors.shape[0]
        return result


class PeriodicHallReference:
    """Uniform frozen Hall symbol; the real nonlinear RHS remains separate.

    Limited to the one-dimensional periodic fixture (K=0), with Z blocks
    (pperp, pparallel, Bx, By, Bz). The two magnetic components are rotated by
    the exact power of their RK4 polynomial, using one FFT over spatial cells.
    No production RZ/variable-coefficient diagonalization is implied.
    """

    def __init__(self, points: int, k: float, b0: float, n0: float,
                 mu0: float, charge: float, h: float, *, xp: Any):
        self.points, self.xp = points, xp
        frequency = xp.fft.fftfreq(points)*points*k
        x = h*frequency**2*b0/(mu0*charge*n0)
        self.real = 1-x*x/2+x**4/24
        self.imag = x-x**3/6
        self.powers: dict[int, tuple[Any, Any]] = {}
        self.action_calls = 0
        self.action_vectors = 0

    def power(self, length: int, vectors: Any) -> Any:
        xp, n = self.xp, self.points
        if vectors.shape[1:] != (1, 5*n):
            raise Failure("Shape", "periodic Hall reference requires K=0 and Nb=3Nc")
        if length not in self.powers:
            plus = (self.real+1j*self.imag)**length
            minus = (self.real-1j*self.imag)**length
            self.powers[length] = ((plus+minus)/2, (plus-minus)/(2j))
        cosine, sine = self.powers[length]
        values = vectors.reshape(-1, 5, n)
        transformed = xp.fft.fft(values, axis=-1)
        bx, by = transformed[:, 2].copy(), transformed[:, 3].copy()
        transformed[:, 2] = cosine*bx-sine*by
        transformed[:, 3] = sine*bx+cosine*by
        self.action_calls += 1
        self.action_vectors += len(values)
        return xp.fft.ifft(transformed, axis=-1).reshape(vectors.shape)


class FrozenJVP:
    """Centered, real-physical directional derivative at a fixed P/Z version.

    Directions are normalized per independent vector using physical scales.
    The caller supplies admissible Hermitian, compatible magnetic directions.
    No automatic complex-step or explicitly stored Jacobian is used.
    """

    def __init__(self, rhs: Callable, p: Any, z: Any, scales: Any, *, xp: Any,
                 eta: float = 1e-5):
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError("invalid directional difference scale")
        self.rhs, self.p, self.z = rhs, p.copy(), z.copy()
        self.scales, self.xp, self.eta = scales.copy(), xp, eta
        self.calls = 0
        self.rhs_vectors = 0

    def __call__(self, vectors: Any) -> Any:
        xp = self.xp
        axes = tuple(range(1, vectors.ndim))
        size = xp.max(xp.abs(vectors/self.scales), axis=axes)
        # Do not amplify a tiny cochain's constraint-roundoff component.
        # Sub-roundoff directional responses remain approximate, never exact JVPs.
        step = xp.minimum(1.,self.eta/xp.maximum(size, xp.finfo(xp.float64).tiny))
        step = xp.where(size > 0, step, 1.)
        weights = step.reshape((-1,) + (1,)*(vectors.ndim-1))
        p = xp.broadcast_to(self.p, (len(vectors),)+self.p.shape)
        plus = self.rhs(p, self.z+weights*vectors)
        minus = self.rhs(p, self.z-weights*vectors)
        self.calls += 1
        self.rhs_vectors += 2*len(vectors)
        return (plus-minus)/(2*weights)


@dataclass
class FieldWindowResult:
    """Trial-local full field waveform and measured algorithm work."""

    nodes: Any
    stages: Any
    rates: Any
    history: list[dict[str, Any]]
    iterations: int
    residual_evaluations: int
    rhs_batches: int
    rhs_vectors: int
    stage_electric: Any = None
    stage_current: Any = None
    stage_velocity: Any = None


def rk4_batch(rhs: Callable, path: Any, left: Any, h: float, *, xp: Any):
    """Four causal stages, with every microstep exposed in one numerical batch."""
    y1 = left
    k1 = rhs(path[:, 0], y1)
    y2 = left+h*k1/2
    k2 = rhs(path[:, 1], y2)
    y3 = left+h*k2/2
    k3 = rhs(path[:, 2], y3)
    y4 = left+h*k3
    k4 = rhs(path[:, 3], y4)
    return (left+h*(k1+2*k2+2*k3+k4)/6,
            xp.stack((y1,y2,y3,y4), axis=1),
            xp.stack((k1,k2,k3,k4), axis=1))


def solve_window(rhs: Callable, path: Any, start: Any, h: float, reference: Any,
                 *, xp: Any, scales: Any, tolerance: float,
                 max_iterations: int = 8, initial: Any = None,
                 workspace_limit_bytes: int = 8 << 30) -> FieldWindowResult:
    """Solve the original fixed-ion RK4 relations using old-window residuals.

    Only convergence scalars cross to the host. There is no causal whole-window
    initialization/replay and no physical state/ledger commit in this function.
    """
    if (not math.isfinite(h) or h <= 0 or not math.isfinite(tolerance)
            or tolerance <= 0 or type(max_iterations) is not int
            or max_iterations < 1):
        raise ValueError("invalid field-window controls")
    if (path.ndim != start.ndim+2 or path.shape[1] != 4 or len(path) < 1
            or path.dtype != xp.complex128 or start.dtype != xp.complex128
            or scales.shape != start.shape):
        raise Failure("Shape", "invalid field-window layout or precision")
    if not bool(xp.all(xp.isfinite(scales) & (scales > 0)).item()):
        raise Failure("Reference", "field-window scales must be finite and positive")
    count = len(path)
    memory_plan=32*count*start.nbytes+path.nbytes+scales.nbytes
    if memory_plan>workspace_limit_bytes:
        raise Failure("MemoryBudget",f"field-window history plan {memory_plan} exceeds {workspace_limit_bytes} bytes")
    rhs_batches, rhs_vectors, evaluations = 0, 0, 0
    history: list[dict[str, Any]] = []
    latest={"nodes":None}
    stage_records=[]

    def enrich(failure):
        failure.history=list(history)
        if latest["nodes"] is not None:
            failure.nodes=latest["nodes"]
        failure.residual_evaluations=evaluations
        failure.rhs_batches,failure.rhs_vectors=rhs_batches,rhs_vectors
        return failure

    def counted_rhs(p,z):
        nonlocal rhs_batches,rhs_vectors
        rhs_batches+=1
        rhs_vectors+=len(z)
        try:
            value=rhs(p,z)
            raw=getattr(rhs,"last",None)
            if raw is not None and all(hasattr(raw,key) for key in ("electric","current","velocity")):
                stage_records.append((raw.electric,raw.current,raw.velocity))
            return value
        except Failure as failure:
            raise enrich(failure)

    if initial is None:
        slope = counted_rhs(path[:1, 0], start[None])[0]
        times = xp.arange(1,count+1,dtype=xp.float64).reshape((-1,)+(1,)*start.ndim)
        window = start+times*h*slope
    else:
        if initial.shape != (count,)+start.shape:
            raise Failure("Shape", "invalid initial field waveform")
        window = initial.copy()
    latest["nodes"]=window

    def evaluate(nodes):
        nonlocal rhs_batches, rhs_vectors, evaluations
        left = xp.concatenate((start[None], nodes[:-1]), axis=0)
        stage_records.clear()
        evaluations += 1
        end, stages, rates = rk4_batch(counted_rhs,path,left,h,xp=xp)
        defect = nodes-end
        norm = float(xp.max(xp.abs(defect)/scales).item())
        if not math.isfinite(norm):
            raise Failure("Finite", "non-finite real field-window residual")
        return norm, defect, stages, rates, tuple(stage_records)

    norm, defect, stages, rates, records = evaluate(window)
    history.append({"iteration":0,"trial":0,"residual":norm,"lambda":0.})
    for iteration in range(max_iterations+1):
        if norm <= tolerance:
            # The last RK stage is not the accepted field endpoint. Validate
            # that endpoint too, against the supplied path's right-stage P.
            # This one charged RHS is neither a serial replay nor an ion commit.
            counted_rhs(path[-1:,3],window[-1:])
            result=FieldWindowResult(window,stages,rates,history,iteration,
                                     evaluations,rhs_batches,rhs_vectors)
            if records:
                result.stage_electric=xp.stack([x[0] for x in records],axis=1)
                result.stage_current=xp.stack([x[1] for x in records],axis=1)
                result.stage_velocity=xp.stack([x[2] for x in records],axis=1)
            return result
        if iteration == max_iterations:
            break
        try:
            delta = operator_prefix(reference.power,-defect,xp.zeros_like(start),xp=xp)
        except Failure as failure:
            raise enrich(failure)
        selected = False
        for trial, weight in enumerate((1.,.5,.25),1):
            candidate = window+weight*delta
            latest["nodes"]=candidate
            try:
                candidate_norm, candidate_defect, candidate_stages, candidate_rates,candidate_records = evaluate(candidate)
            except Failure as failure:
                history.append({"iteration":iteration+1,"trial":trial,
                                "residual":None,"lambda":weight,"failure":failure.code})
                if failure.code not in {"Domain","Magnetization","MagneticDomain",
                                        "InitialCompatibility","WallCurrent","WallState"}:
                    failure.history,failure.nodes=history,window
                    failure.residual_evaluations=evaluations
                    failure.rhs_batches,failure.rhs_vectors=rhs_batches,rhs_vectors
                    raise
                continue
            history.append({"iteration":iteration+1,"trial":trial,
                            "residual":candidate_norm,"lambda":weight})
            if candidate_norm < norm or candidate_norm <= tolerance:
                window, norm, defect = candidate, candidate_norm, candidate_defect
                stages, rates = candidate_stages, candidate_rates
                records=candidate_records
                selected = True
                break
            latest["nodes"]=window
        if not selected:
            break
    failure = Failure("FieldWindowConvergence", "field-window iteration did not meet its original residual tolerance")
    failure.history, failure.nodes = history, window
    failure.residual_evaluations = evaluations
    failure.rhs_batches, failure.rhs_vectors = rhs_batches, rhs_vectors
    raise failure
