"""Frozen pressure-free matrix from A0's actual rectangular operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, eig, null_space

from fomhd.geometry import build_aligned_box
from fomhd.operators import OperatorOptions, build_operators

from box_reference import E, MU0, load
from prepare import choose_branch, map_fields


def row_normalized(matrix):
    scale = np.linalg.norm(matrix, axis=1)
    active = scale > 0
    return matrix[active]/scale[active,None]


def frozen(geom, op, box, mode):
    c, a, d = [group[op.K+mode] for group in (op.C,op.Cdag,op.DB)]
    inject, pair, t = op.I, op.H, op.T
    nu = 3*op.Nc
    m0 = np.tile(op.M0,3)
    lift = t.conj().T/m0[:,None]
    q = np.eye(nu) - lift @ np.linalg.solve(t @ lift,t)
    walls = [w.face if w.component == 'r_face' else len(geom.r_faces.R)+op.Nc+w.face
             for w in geom.walls]
    trace = np.eye(op.Ne)[walls]
    electric_lift = trace.T/op.M1[:,None]
    normal_curl_curl = trace @ a @ c
    wall_system = normal_curl_curl @ electric_lift
    rows = op.compute.independent_rows(wall_system,1e-12)
    pw = np.eye(op.Ne) - electric_lift[:, rows] @ np.linalg.solve(
        wall_system[np.ix_(rows,rows)],normal_curl_curl[rows])
    cross = np.zeros((nu,nu),complex)
    n = op.Nc
    background = np.zeros(op.Nb,complex)
    off = len(geom.z_faces.R)
    background[off:off+len(geom.vertices.R)] = box.B0*box.R0/geom.Rbar_v
    b0 = (op.QB @ background)[n:2*n]/box.B0
    cross[:n,2*n:] = -np.diag(b0)
    cross[2*n:,:n] = np.diag(b0)
    # All velocities, fields and electric fields are dimensionless.
    va = box.B0/np.sqrt(MU0*box.mass*box.density)
    di = np.sqrt(box.mass/(MU0*box.density*E**2))
    gyro = E*box.B0/box.mass
    ohm = -pw @ inject @ q @ cross @ q @ pair
    eu, eb = ohm @ inject, -di*ohm @ a
    matrix = np.block([
        [gyro*q@(cross + pair@eu), gyro*q@pair@eb],
        [-va*c@eu, -va*c@eb],
    ])
    basis = block_diag(null_space(row_normalized(t),rcond=1e-12),
                       null_space(row_normalized(np.vstack((d,trace@a))),rcond=1e-12))
    restriction = basis.conj().T@matrix@basis
    eigvals,eigvecs = eig(restriction)
    projected = basis@eigvecs
    residual = float(np.linalg.norm(matrix@basis-basis@restriction)/np.linalg.norm(matrix@basis))
    return eigvals, projected, dict(
        operator_invariance_defect=residual,
        max_wall_reaction_defect=float(np.linalg.norm(normal_curl_curl@pw)/np.linalg.norm(normal_curl_curl)),
        max_real_s_inv=float(np.max(np.abs(eigvals.real))),
        max_imag_rad_s=float(np.max(np.abs(eigvals.imag))),
        max_real_positive_s_inv=float(np.max(eigvals.real)),
        constrained_dim=restriction.shape[0],
        va_m_per_s=float(va), ion_skin_depth_m=float(di),
    )


def run(reference, grid, output):
    box = reference['box']
    nodes = -box.a + (np.arange(grid)+.5)*2*box.a/grid
    marker=[(box.R0-.15,-.1),(box.R0+.15,-.1),(box.R0,.15)]
    geom=build_aligned_box(box.R0+nodes,nodes,marker)
    kwargs={'rank_rtol':1e-12,'memory_cap_bytes':20<<30}
    if hasattr(OperatorOptions,'include_finite_mass'):
        kwargs['include_finite_mass']=False
    op=build_operators(geom,1,kwargs)
    selected_ref = choose_branch(reference)
    b,v,fields,seed = map_fields(reference,selected_ref,geom,op)
    width = op.Nc
    result = dict(grid=[grid,grid],reference_frequency_hz=float(reference['omega'][selected_ref]/(2*np.pi)),
                  reference_mode_index=selected_ref,seed=seed,spectra={})
    for mode in (0,1):
        vals, vecs, metrics = frozen(geom,op,box,mode)
        result['spectra'][mode]=metrics
        np.save(output.with_name(output.name+f'-mode{mode}.npy'),vals)
        if mode != 1:
            continue
        vel = vecs[:3*width]*metrics['va_m_per_s']
        mag = (op.QB@vecs[3*width:])*box.B0
        comp = mag[width:2*width]
        compref = fields['B'][:,1]
        magref = fields['B'].T.reshape(-1)
        weights = op.M0
        m0 = np.tile(weights,3)
        compnorm = np.einsum('ij,i->j',abs(comp)**2,weights)
        magnorm = np.einsum('ij,i->j',abs(mag)**2,m0)
        scores = abs(compref.conj()@(comp*weights[:,None]))/np.sqrt(
            np.maximum(compnorm,1e-300)*np.vdot(compref,weights*compref).real)
        fullmag = abs(magref.conj()@(mag*m0[:,None]))/np.sqrt(
            np.maximum(magnorm,1e-300)*np.vdot(magref,m0*magref).real)
        energy_frac = compnorm/np.maximum(magnorm,1e-300)
        sel = np.flatnonzero((vals.imag > 0) & (energy_frac > .6))
        rank = sorted(sel,key=lambda i:fullmag[i],reverse=True)[:10]
        result['candidates'] = [dict(
            index=int(i),frequency_hz=float(vals[i].imag/(2*np.pi)),
            real_s_inv=float(vals[i].real),compression=float(energy_frac[i]),
            deltaB_parallel_overlap=float(scores[i]),all_B_overlap=float(fullmag[i]),
        ) for i in rank]
        i=rank[0]
        result['best']=result['candidates'][0]
        np.savez(output.with_name(output.name+'-tracked.npz'),R=geom.cells.R,Z=geom.cells.Z,
                 B=vecs[3*width:,i]*box.B0,B_nodal=mag[:,i],u=vel[:,i],lambda_s_inv=vals[i],
                 refB=fields['B'].T.reshape(-1),refu=fields['u'].T.reshape(-1),
                 weights=op.M0)
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--grid',type=int,default=8)
    args=p.parse_args()
    run(load(args.reference),args.grid,args.output)
