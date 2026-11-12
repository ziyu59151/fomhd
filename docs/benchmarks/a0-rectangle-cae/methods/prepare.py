"""Map the independent same-box cold Hall mode to the production RK4 case."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tomllib

import numpy as np

from fomhd import io
from fomhd.backend import Backend
from fomhd.geometry import build_aligned_box
from fomhd.operators import OperatorOptions, apply_t1, build_operators, magnetic_pair
from fomhd.types import FluidState, State
from tools.case_config import render_massless_example
from tools.warp_tokamak_cae import physical_parameters

from box_reference import E, MU0, evaluate, load


def toml_lines(table,prefix=()):
    """Serialize the scalar and nested-table values in the solver case."""
    lines=[]
    if prefix:
        lines.append("["+".".join(json.dumps(key) for key in prefix)+"]")
    for key,value in table.items():
        if not isinstance(value,dict):
            lines.append(json.dumps(key)+" = "+json.dumps(value))
    for key,value in table.items():
        if isinstance(value,dict):
            lines.extend(["",*toml_lines(value,prefix+(key,))])
    return lines


def choose_branch(result):
    indices = np.flatnonzero((result['norm_omega'] > .4) & (result['compression'] > .6))
    return int(indices[0])


def average_potential(result, index, r, z, lo, hi, component):
    points, weights = np.polynomial.legendre.leggauss(8)
    coordinate = (lo[:, None] + hi[:, None])/2 + (hi[:,None]-lo[:,None])*points/2
    radii = coordinate if component == 0 else r[:,None]
    heights = coordinate if component == 2 else z[:,None]
    field = evaluate(result, index, radii, heights)['A'][:, component].reshape(-1,points.size)
    return field @ (weights/2)


def map_fields(result, index, geom, op):
    box = result['box']
    R, Z = geom.cells.R, geom.cells.Z
    h = 2*box.a/len(geom.R)
    re, ze = geom.r_faces, geom.z_faces
    raw_a = np.r_[
        average_potential(result,index,re.R,re.Z,np.maximum(box.R0-box.a,re.R-h/2),
                          np.minimum(box.R0+box.a,re.R+h/2),0),
        evaluate(result,index,R,Z)['A'][:,1],
        average_potential(result,index,ze.R,ze.Z,np.maximum(-box.a,ze.Z-h/2),
                          np.minimum(box.a,ze.Z+h/2),2),
    ]
    mode_slot = op.K + box.mode
    b_raw = op.C[mode_slot] @ apply_t1(op, raw_a)
    nr = len(re.R)
    walls = np.asarray([w.face if w.component == 'r_face' else nr+op.Nc+w.face
                        for w in geom.walls])
    # Minimum magnetic-norm correction on the exact Faraday range. Removing
    # normal Ampère current leaves magnetic divergence and wall caps intact.
    wall_lift = op.C[mode_slot][:,walls]/np.sqrt(op.M1[walls])[None,:]
    wall_rows = np.sqrt(op.M1[walls])[:,None]*op.Cdag[mode_slot][walls]
    lam = np.linalg.lstsq(wall_rows @ wall_lift, -(wall_rows @ b_raw), rcond=1e-12)[0]
    b = b_raw + wall_lift @ lam
    fields = evaluate(result,index,R,Z)
    v_raw = fields['u'].T.reshape(-1)
    mass = np.tile(op.M0,3)
    v = v_raw - np.linalg.lstsq(op.T/np.sqrt(mass)[None,:], op.T@v_raw, rcond=1e-12)[0]/np.sqrt(mass)
    back = Backend('cpu')
    corr_b = math.sqrt(float(magnetic_pair(op,b-b_raw,b-b_raw,back)/magnetic_pair(op,b_raw,b_raw,back)))
    corr_v = math.sqrt(float(np.sum(mass*abs(v-v_raw)**2)/np.sum(mass*abs(v_raw)**2)))
    b_reference = fields['B'].T.reshape(-1)
    b_nodal = op.QB @ b
    def overlap(x, y, weights):
        return float(abs(np.vdot(x,weights*y))/np.sqrt(np.vdot(x,weights*x).real*np.vdot(y,weights*y).real))
    receipt = dict(
        magnetic_projection_relative=corr_b, ion_velocity_projection_relative=corr_v,
        sampled_to_reference_B_overlap=overlap(b_reference,b_nodal,mass),
        sampled_to_reference_Bphi_overlap=overlap(b_reference[op.Nc:2*op.Nc],b_nodal[op.Nc:2*op.Nc],op.M0),
        perturbation_discrete_divB_max=float(np.max(abs(op.DB[mode_slot] @ b))),
        perturbation_discrete_normal_curlB_max=float(np.max(abs(op.Cdag[mode_slot][walls] @ b))),
        perturbation_discrete_normal_velocity_max=float(np.max(abs(op.T @ v))),
    )
    return b,v,fields,receipt


def discrete_seed(path, reference, index, geometry, op):
    box = reference['box']
    with np.load(path) as saved:
        b, v, frequency = saved['B'], saved['u'], complex(saved['lambda_s_inv'].item())
        for name, coords in [('R',geometry.cells.R),('Z',geometry.cells.Z)]:
            np.testing.assert_array_equal(saved[name], coords)
    n = op.Nc
    mode = op.K + box.mode
    independent = evaluate(reference,index,geometry.cells.R,geometry.cells.Z)
    current = op.Cdag[mode] @ b / MU0
    u_e = (v - (op.H @ current)/(E*box.density)).reshape(3,n).T
    fields = dict(
        independent, div_u=op.DJ[mode]@op.I@v, eigen_omega=frequency,
        electron_velocity=u_e,
    )
    bphi = (op.QB@b)[n:2*n]
    bphi_reference = independent['B'][:,1]
    weight = op.M0
    wall_edges = [w.face if w.component == 'r_face' else len(geometry.r_faces.R)+n+w.face
                  for w in geometry.walls]
    receipt = dict(
        discrete_seed_file=str(path.resolve()),
        discrete_seed_eigenfrequency_hz=frequency.imag/(2*np.pi),
        discrete_seed_eigenvalue_real_s_inv=frequency.real,
        reference_Bphi_overlap=float(abs(np.vdot(bphi_reference,weight*bphi)) /
            np.sqrt(np.vdot(bphi,weight*bphi).real*np.vdot(bphi_reference,weight*bphi_reference).real)),
        normal_current_residual=float(np.max(abs(current[wall_edges]))),
        normal_velocity_residual=float(np.max(abs(op.T@v))),
        divergence_residual=float(np.max(abs(op.DB[mode]@b))),
    )
    return b, v, fields, receipt


def write_case(result, index, geometry, op, mapped, directory, dt, steps, amplitude, stride, device, memory):
    directory.mkdir(parents=True, exist_ok=False)
    box = result['box']
    b,v,fields,receipt = mapped
    n = op.Nc
    va = box.B0 / math.sqrt(MU0*box.mass*box.density)
    pressure_total = 1e-5*box.B0**2/(2*MU0)
    p_i = p_e = pressure_total/2
    iomega = fields.get('eigen_omega', 1j*result['omega'][index])
    vc = fields['u']
    u_e = fields.get('electron_velocity', vc - fields['J']/(E*box.density))
    strain = (u_e[:,0] + 1j*box.mode*u_e[:,1])/geometry.cells.R
    div_v = fields['div_u']
    dn = -box.density*div_v/iomega
    dpi = -(5/3)*p_i*div_v/iomega
    dpe = p_e*(strain-2*div_v)/iomega
    dppe = -p_e*(div_v+2*strain)/iomega
    def add_mode(base, delta):
        base = np.asarray(base, np.complex128)
        return np.r_[np.conj(delta*scale)/2,base,delta*scale/2]
    bphi = (op.QB@b)[n:2*n]
    phase = np.exp(-1j*np.angle(bphi[np.argmax(abs(bphi))]))
    norm = max(float(np.max(abs(b)))/box.B0, float(np.max(abs(v)))/va, float(np.max(abs(dn)))/box.density)
    scale = phase*amplitude/norm
    dn_field = add_mode(np.full(n,box.density),dn)
    # Products of two perturbations are deliberately omitted in this linear
    # mode seed. The normal-wall projection remains exact at first order.
    ion = FluidState(n=dn_field.copy(), mom=add_mode(np.zeros(3*n),box.mass*box.density*v),
                     perp=add_mode(np.full(n,p_i),dpi), parallel=np.zeros(0,np.complex128))
    electron = FluidState(n=dn_field.copy(),mom=np.zeros(9*n,np.complex128),
                          perp=add_mode(np.full(n,p_e),dpe), parallel=add_mode(np.full(n,p_e),dppe))
    bg = np.zeros(op.Nb,np.complex128)
    off = len(geometry.z_faces.R)
    bg[off:off+len(geometry.vertices.R)] = box.B0*box.R0/geometry.Rbar_v
    state = State(fluid=(ion,electron),B=add_mode(bg,b),generation=1)
    arrays = io.encode_state(state,geometry)
    arrays['wall_magnetic_flux'] = np.zeros((3,len(geometry.walls),2),np.complex128)
    np.savez(directory/'initial.npz',**arrays)
    polygon=np.asarray([(point.R,point.Z) for point in geometry.lcfs_polygon])
    np.savez(directory/'geometry.npz',R_nodes=geometry.R,Z_nodes=geometry.Z,
             lcfs_R=polygon[:,0],lcfs_Z=polygon[:,1])
    np.savez(directory/'seed.npz', R=geometry.cells.R,Z=geometry.cells.Z,
             weight=op.M0,B_seed=(op.QB@b)*scale,B_reference=fields['B'].T.reshape(-1)*scale,
             u_seed=v*scale,u_reference=vc.T.reshape(-1)*scale,background_B=op.QB@bg)
    doc=tomllib.loads(render_massless_example())
    h=2*box.a/len(geometry.R)
    cells=len(geometry.R)
    doc['case_name']=f'box-cae-n1-g{cells}'
    doc['grid']=dict(R0=float(geometry.R[0]),Z0=float(geometry.Z[0]),hR=h,hZ=h,NR=cells,NZ=cells,K=1,Nphi=5)
    doc['species']['mi']=box.mass
    doc['time'].update(dt=dt,nsteps=steps,dt_guard='error',
                       initial_kind='non_equilibrium' if amplitude else 'equilibrium',integrator='rk4')
    doc['boundary']=dict(kind='fixed_outer_wall_static',potential=0.,version='rectangle_relative_v1')
    p=physical_parameters(1,box.density,box.B0,box.a,pressure_total)
    doc['solver'].update(backend=device,memory_limit_bytes=memory,device_memory_limit_bytes=memory)
    ref=p.ref
    doc['solver']['references']=dict(L_m=ref.length,t_s=ref.time,n_i_m3=ref.ni,n_e_m3=ref.ne,
        u_mps=ref.velocity,E_Vpm=ref.electric,B_T=float(ref.magnetic),J_Apm2=ref.current,
        rho_Cpm3=ref.charge,p_Pa=ref.pressure)
    doc['budgets']['domain'].update(B_min=1e-10)
    doc['budgets']['checks']={key:dict(atol=float(budget.atol),rtol=float(budget.rtol),
        reference=float(budget.reference),unit=budget.unit,norm=key.rsplit('.',1)[1],
        gate='hard' if io._massless_budget_registry(p)[key][2] else 'observe')
        for key,budget in p.budgets.items()}
    doc['diagnostics']['inner_distance_m']=.2*box.a
    doc['io'].update(geometry_file='geometry.npz',initial_file='initial.npz',
        source_manifest_file='source.json',output_dir='run',snapshot_stride=stride,
        checkpoint_stride=steps,write_residual_arrays=False)
    (directory/'case.toml').write_text('\n'.join(toml_lines(doc))+'\n')
    source = dict(
        provider_name='Independent same-box constrained cold Hall Ritz eigenmode',
        provider_version='v1',units='SI',
        conversion_method=('Same-branch frozen solver mode as initial data, independently compared to the same-box Ritz mode' if 'eigen_omega' in fields else 'Analytic vector potential link averages with explicit Faraday-range normal current projection'),
        continuous_residual_status='Vacuum toroidal guide field with uniform pressure and density is an exact static equilibrium',
        reference_mode_index=index,reference_order=box.order,
        reference_frequency_hz=float(result['omega'][index]/(2*np.pi)), total_axis_beta=1e-5,
        source_box_bounds_m=[box.R0-box.a,box.R0+box.a,-box.a,box.a],
        amplitude=amplitude,reference_compression=float(result['compression'][index]),
        background_normal_cap_max_Wb_per_rad=0.0,
        background_current_max_A_per_m2=float(np.max(abs(op.Cdag[op.K] @ bg))/MU0),
        background_divergence_max_T_per_m=float(np.max(abs(op.DB[op.K] @ bg))),
        initial_divergence_max_T_per_m=float(max(np.max(abs(curl @ col))
            for curl,col in zip(op.DB,state.B.reshape(3,-1),strict=True))),
        projection_receipt=receipt,
        input_hashes={key:io.sha256_file(directory/file) for key,file in (
            ('case','case.toml'),('initial','initial.npz'),('geometry','geometry.npz'))},
    )
    (directory/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('reference',type=Path)
    parser.add_argument('--grid',type=int,default=8)
    parser.add_argument('--dt',type=float,default=8e-10)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--memory-gib',type=int,default=20)
    parser.add_argument('--amplitude',type=float,default=1e-4)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--quiet',action='store_true')
    parser.add_argument('--discrete',type=Path)
    parser.add_argument('--pair',action='store_true')
    args=parser.parse_args()
    result=load(args.reference)
    index=choose_branch(result)
    box=result['box']
    x=-box.a+(np.arange(args.grid)+.5)*2*box.a/args.grid
    angle=np.arange(48)*2*np.pi/48
    marker=np.c_[box.R0+.25*box.a*np.cos(angle),.25*box.a*np.sin(angle)]
    memory=args.memory_gib<<30
    geom=build_aligned_box(box.R0+x,x,marker,{'memory_cap_bytes':memory})
    kwargs={'rank_rtol':1e-12,'memory_cap_bytes':memory}
    if hasattr(OperatorOptions,'include_finite_mass'):
        kwargs['include_finite_mass']=False
    op=build_operators(geom,1,kwargs)
    mapped=(discrete_seed(args.discrete,result,index,geom,op) if args.discrete
            else map_fields(result,index,geom,op))
    kinds=[('quiet',0.0),('pulse',args.amplitude)] if args.pair else [
        ('',0.0 if args.quiet else args.amplitude)
    ]
    for suffix,amplitude in kinds:
        directory=args.output.with_name(args.output.name+'-'+suffix) if suffix else args.output
        source=write_case(result,index,geom,op,mapped,directory,args.dt,args.steps,
                          amplitude,args.stride,args.device,memory)
        print(json.dumps(source,indent=2))


if __name__=='__main__':
    main()
