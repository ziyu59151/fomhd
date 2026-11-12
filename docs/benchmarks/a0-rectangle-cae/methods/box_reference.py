"""Independent constrained cold Hall Ritz spectrum for a toroidal vacuum box."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy.linalg import eigh, null_space

from tools.cold_hall_reference import E, MP, MU0, _cross_bphi, _jets, _whiten


@dataclass(frozen=True)
class Box:
    order: int = 6
    mode: int = 1
    R0: float = 3.0
    a: float = 0.6
    B0: float = 2.0
    density: float = 2e20
    mass: float = 2*MP
    rank_rtol: float = 1e-11


def terms(box):
    """Boundary-adapted Chebyshev products in the physical component basis."""
    dim = box.order + 4
    basis = [
        Chebyshev.basis(i).convert(kind=Polynomial)
        for i in range(box.order + 1)
    ]
    bubble = Polynomial([1, 0, -1])
    velocity, potential = [], []
    for degree in range(box.order + 1):
        for i in range(degree + 1):
            j = degree - i
            for component in (0, 1, 2):
                v = np.zeros((3, dim, dim))
                a = np.zeros_like(v)
                for array, bx, bz in (
                    (v, component == 0, component == 2),
                    (a, component != 0, component != 2),
                ):
                    x = basis[i] * bubble if bx else basis[i]
                    z = basis[j] * bubble if bz else basis[j]
                    array[component, :len(x.coef), :len(z.coef)] = np.outer(x.coef,z.coef)
                velocity.append((v, 0))
                # Covariant scaling of A_phi makes the representation regular in R.
                potential.append((a, int(component == 1)))
    return velocity, potential


def gauge_complement(box):
    """Remove analytically represented gradients before mass whitening."""
    triples = [(i,degree-i,c) for degree in range(box.order+1)
               for i in range(degree+1) for c in range(3)]
    index = {key: k for k,key in enumerate(triples)}
    bubble = Chebyshev([.5,0,-.5])
    directions = []
    for degree in range(box.order):
        for i in range(degree+1):
            j = degree-i
            column = np.zeros(len(triples),complex)
            for k,value in enumerate((Chebyshev.basis(i)*bubble).deriv().coef / box.a):
                column[index[(k,j,0)]] = value
            for k,value in enumerate((Chebyshev.basis(j)*bubble).deriv().coef / box.a):
                column[index[(i,k,2)]] = value
            column[index[(i,j,1)]] = 1j*box.mode
            directions.append(column)
    return null_space(np.asarray(directions).conj(), rcond=box.rank_rtol)


def fields(box, R, Z):
    """Analytic cylindrical potential, B=curl A and J=curl B/mu0."""
    R, Z = np.broadcast_arrays(np.asarray(R), np.asarray(Z))
    R, Z = R.ravel(), Z.ravel()
    vt, at = terms(box)
    v = _jets(box, vt, R, Z)
    a = _jets(box, at, R, Z)
    rr = R[:, None]
    ik = 1j * box.mode
    av, ar, az = a['v'], a['r'], a['z']
    b, br, bz = np.zeros_like(av), np.zeros_like(av), np.zeros_like(av)
    b[:, 0] = ik * av[:, 2] / rr - az[:, 1]
    b[:, 1] = az[:, 0] - ar[:, 2]
    b[:, 2] = ar[:, 1] + av[:, 1] / rr - ik * av[:, 0] / rr
    br[:, 0] = ik * (ar[:, 2] / rr - av[:, 2] / rr**2) - a['rz'][:, 1]
    bz[:, 0] = ik * az[:, 2] / rr - a['zz'][:, 1]
    br[:, 1] = a['rz'][:, 0] - a['rr'][:, 2]
    bz[:, 1] = a['zz'][:, 0] - a['rz'][:, 2]
    br[:, 2] = a['rr'][:, 1] + ar[:, 1] / rr - av[:, 1] / rr**2 - ik * (ar[:, 0] / rr - av[:, 0] / rr**2)
    bz[:, 2] = a['rz'][:, 1] + az[:, 1] / rr - ik * az[:, 0] / rr
    current = np.zeros_like(b)
    current[:, 0] = (ik * b[:, 2] / rr - bz[:, 1]) / MU0
    current[:, 1] = (bz[:, 0] - br[:, 2]) / MU0
    current[:, 2] = (br[:, 1] + b[:, 1] / rr - ik * b[:, 0] / rr) / MU0
    div_v = v['r'][:, 0] + v['v'][:, 0] / rr + ik * v['v'][:, 1] / rr + v['z'][:, 2]
    return dict(velocity=v['v'], potential=av, b=b, current=current, div_velocity=div_v)


def quadrature(box):
    points, weights = np.polynomial.legendre.leggauss(2*box.order + 9)
    r, z = np.meshgrid(box.R0 + box.a*points, box.a*points)
    w = r * np.outer(weights, weights) * box.a**2
    return r.ravel(), z.ravel(), w.ravel()


def wall_currents(box, coordinates):
    R, Z = np.asarray(coordinates), np.asarray(coordinates)
    return np.vstack((
        fields(box, box.R0-box.a, box.a*Z)['current'][:, 0],
        fields(box, box.R0+box.a, box.a*Z)['current'][:, 0],
        fields(box, box.R0+box.a*R, -box.a)['current'][:, 2],
        fields(box, box.R0+box.a*R, box.a)['current'][:, 2],
    ))


def solve(box, *, hall_multiplier=1.0):
    r, z, w = quadrature(box)
    f = fields(box, r, z)
    coordinates = np.polynomial.legendre.leggauss(2*box.order+11)[0]
    wall = wall_currents(box, coordinates)
    # Multiplying the axial-wall trace by R² gives polynomial rows of the
    # same physical constraint and improves their numerical conditioning.
    rr = box.R0 + box.a*coordinates
    wall *= np.r_[np.ones(2*coordinates.size),rr**2,rr**2][:,None]
    gauge_free = gauge_complement(box)
    wall = wall @ gauge_free
    restrict = gauge_free @ null_space(wall / np.max(abs(wall)), rcond=box.rank_rtol)
    vt, _, nv = _whiten(f['velocity'], w * box.mass * box.density, box.rank_rtol)
    bt, _, nb = _whiten(f['b'] @ restrict, w / MU0, box.rank_rtol)
    at = restrict @ bt
    v = f['velocity'] @ vt
    b = f['b'] @ at
    current = f['current'] @ at
    force = _cross_bphi(current, box.B0*box.R0/r)
    # Explicit quadrature and BLAS products keep the reference inexpensive.
    wf = (force*w[:, None, None]).reshape(-1, nb)
    L = v.reshape(-1, nv).conj().T @ wf
    H = -current.reshape(-1, nb).conj().T @ wf / (E * box.density)
    H *= hall_multiplier
    A = np.block([[np.zeros((nv,nv)), L], [-L.conj().T, H]])
    frequency, eigenvectors = eigh(-1j*A)
    b_modes = b @ eigenvectors[nv:]
    energy = np.einsum('qij,q->j', np.abs(b_modes)**2, w)
    compression = np.einsum('qj,q->j', np.abs(b_modes[:, 1])**2, w)
    ratio = compression / np.maximum(energy, 1e-300)
    independent_wall = wall_currents(box, np.linspace(-.997,.998,3*box.order+11))
    ac = at @ eigenvectors[nv:]
    vc = vt @ eigenvectors[:nv]
    va = box.B0 / np.sqrt(MU0*box.mass*box.density)
    return dict(
        box=box, omega=frequency, v=vc, a=ac, compression=ratio,
        n_v=nv, n_b=nb, n_constraints=gauge_free.shape[1]-restrict.shape[1],
        relative_skew=np.linalg.norm(A+A.conj().T) / np.linalg.norm(A),
        wall_Jn=np.max(abs(independent_wall @ ac), axis=0),
        jnorm=np.max(abs(current @ eigenvectors[nv:]), axis=(0,1)),
        norm_omega=frequency * box.a / va,
    )


def evaluate(result, index, r, z):
    f = fields(result['box'], r, z)
    return dict(
        u=f['velocity'] @ result['v'][:, index],
        A=f['potential'] @ result['a'][:, index],
        B=f['b'] @ result['a'][:, index],
        J=f['current'] @ result['a'][:, index],
        div_u=f['div_velocity'] @ result['v'][:, index],
    )


def save(output, result):
    output = Path(output)
    box = result['box']
    arrays = {name: val for name, val in result.items() if isinstance(val, np.ndarray)}
    np.savez(output.with_suffix('.npz'), **arrays)
    meta = dict(box=asdict(box), **{k: float(result[k]) for k in ('n_v','n_b','n_constraints','relative_skew')})
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')


def load(path):
    path = Path(path)
    with np.load(path) as f:
        values = dict(f)
    values['box'] = Box(**json.loads(path.with_suffix('.json').read_text())['box'])
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--order', type=int, default=6)
    parser.add_argument('--width', type=float, default=.6,
                        help='Half-width of both physical box sides in meters')
    parser.add_argument('--major-radius', type=float, default=3.0)
    parser.add_argument('--density', type=float, default=2e20)
    args = parser.parse_args()
    result = solve(Box(order=args.order,a=args.width,R0=args.major_radius,density=args.density))
    save(args.output, result)
    keep = (result['norm_omega'] > .4) & (result['compression'] > .6)
    for index in np.flatnonzero(keep)[:22]:
        print(index, f"{result['omega'][index]/(2*np.pi*1e6):.8f} MHz",
              f"omega a/vA={result['norm_omega'][index]:.6f}",
              f"parallel={result['compression'][index]:.6f}",
              f"wall={result['wall_Jn'][index]/result['jnorm'][index]:.3g}")
