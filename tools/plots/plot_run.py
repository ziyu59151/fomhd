#!/usr/bin/env python3
"""Render native-grid snapshots and diagnostics from one accepted A0 run."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edges(x):
    return np.r_[x[0]-(x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2]


def coefficient(run, path, key, component=None, mode=0):
    return analyze._snapshot_field(run, path, {"key": key, "component": component, "mode": mode})


def real_fourier(coefficients, phase):
    count = coefficients.shape[0]
    if count % 2 != 1:
        raise ValueError("Retained Fourier axis must have odd length")
    modes = np.arange(-(count//2), count//2+1)
    samples = np.einsum("kr,kp->rp", coefficients, np.exp(1j*modes[:,None]*phase))
    scale = max(float(np.max(np.abs(samples))), np.finfo(float).tiny)
    if np.max(np.abs(samples.imag)) > 1e-10*scale:
        raise ValueError("Physical Fourier reconstruction is not real")
    return samples.real


def render(run, output):
    history = run.history
    times = np.array([float(r["time_s"]) for r in history])
    steps = np.array([int(r["step"]) for r in history])
    if len(history) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("An accepted multi-step run is required")
    first = run.directory/"snapshots"/f"state_{steps[0]:010d}.npz"
    last = run.directory/"snapshots"/f"state_{steps[-1]:010d}.npz"
    density, cell, t, step, _ = coefficient(run, last, "n_i")
    coefficient(run, first, "n_i")
    magnetic, vertex, bt, bstep, _ = coefficient(run, last, "B", "phi")
    if step != steps[-1] or t != times[-1] or bt != t or bstep != step:
        raise ValueError("Final snapshot identity does not match history")
    with np.load(run.paths["inputs/geometry.npz"], allow_pickle=False) as geometry:
        pr, pz = geometry["lcfs_R"], geometry["lcfs_Z"]
    kmax = int(run.resolved["grid"]["K"])
    rows = np.flatnonzero(np.any(cell.active, axis=1))
    iz = int(rows[len(rows)//2])
    phi = np.linspace(0, 2*np.pi, 145)
    nc, pc = [], []
    for mode in range(-kmax, kmax+1):
        n, _, _, _, _ = coefficient(run, last, "n_i", mode=mode)
        p, _, _, _, _ = coefficient(run, last, "mom_i", "phi", mode=mode)
        nc.append(n[iz]); pc.append(p[iz])
    ns, ps = real_fourier(np.asarray(nc), phi), real_fourier(np.asarray(pc), phi)
    active = np.broadcast_to(cell.active[iz,:,None], ns.shape)
    if np.any(ns[active] <= 0):
        raise ValueError("Reconstructed active ion density is not positive")
    velocity = np.full(ns.shape, np.nan)
    np.divide(ps, float(run.resolved["species"]["mi"])*ns, out=velocity, where=active)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    fig.suptitle(f"A0 accepted run: {run.resolved['case_name']}\n"
                 f"{steps[-1]} steps, final time {t*1e9:g} ns; actual coarse-grid outputs", fontsize=16)
    ax = axes[0,0]
    rr, zz = np.meshgrid(cell.R, cell.Z)
    ax.scatter(rr, zz, s=14, color="0.8", label="Inactive centers")
    ax.scatter(rr[cell.active], zz[cell.active], s=25, color=COLORS[0], label="Active cells")
    re, ze = edges(cell.R), edges(cell.Z)
    for j,i in zip(*np.nonzero(cell.active)):
        for dj,di,xx,yy in [(-1,0,[re[i],re[i+1]],[ze[j],ze[j]]),
                            (1,0,[re[i],re[i+1]],[ze[j+1],ze[j+1]]),
                            (0,-1,[re[i],re[i]],[ze[j],ze[j+1]]),
                            (0,1,[re[i+1],re[i+1]],[ze[j],ze[j+1]])]:
            nj,ni=j+dj,i+di
            if not (0 <= nj < cell.active.shape[0] and 0 <= ni < cell.active.shape[1] and cell.active[nj,ni]):
                ax.plot(xx, yy, color=COLORS[0], lw=1.8)
    ax.plot(np.r_[pr,pr[0]],np.r_[pz,pz[0]],"--",color=COLORS[1],label="Input contour")
    ax.set(aspect="equal",xlabel="R [m]",ylabel="Z [m]",title="(a) Input contour and staircase boundary")
    ax.legend(fontsize=8, loc="best")

    ax=axes[0,1]; br,bz=np.meshgrid(vertex.R,vertex.Z)
    if np.max(np.abs(magnetic.imag)) > 1e-10*max(np.max(np.abs(magnetic)),np.finfo(float).tiny):
        raise ValueError("Axisymmetric magnetic component is not real")
    im=ax.scatter(br[vertex.active],bz[vertex.active],c=magnetic.real[vertex.active]*1e6,
                  cmap="viridis",marker="s",s=95)
    ax.plot(np.r_[pr,pr[0]],np.r_[pz,pz[0]],"--",color="0.3",lw=1)
    ax.set(aspect="equal",xlabel="R [m]",ylabel="Z [m]",title=r"(b) $B_\phi$, mode 0: native vertex samples")
    fig.colorbar(im,ax=ax,label=r"$B_\phi$ [$\mu$T]",shrink=0.8)

    ax=axes[0,2]
    vmax=float(np.nanmax(np.abs(velocity)))
    if not np.isfinite(vmax) or vmax <= 0:
        ax.text(0.5,0.5,"Zero toroidal velocity",transform=ax.transAxes,ha="center")
    else:
        im=ax.pcolormesh(np.degrees(phi),cell.R,np.ma.masked_invalid(velocity),
                         shading="nearest",cmap="RdBu_r",vmin=-vmax,vmax=vmax,rasterized=True)
        fig.colorbar(im,ax=ax,label=r"$u_{i,\phi}$ [m/s]",shrink=0.8)
    ax.set(xlim=(0,360),xlabel=r"Toroidal angle $\phi$ [deg]",ylabel="R [m]",
           title=f"(c) Stored Fourier series at Z = {cell.Z[iz]:g} m")

    ax=axes[1,0]; drift={}
    for key,label,color in [("number_i",r"$\Delta N_i/N_i(0)$",COLORS[0]),
                             ("number_e",r"$\Delta N_e/N_e(0)$",COLORS[1]),
                             ("total_energy_J",r"$\Delta E/E(0)$",COLORS[2])]:
        values=np.array([float(r[key]) for r in history])
        if not np.all(np.isfinite(values)) or values[0] == 0:
            raise ValueError("Nonfinite or zero baseline in conserved history")
        y=(values-values[0])/abs(values[0]); drift[key]=y.tolist()
        ax.plot(times*1e9,y,color=color,marker=".",ms=4,label=label)
    ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    ax.set(xlabel="Time [ns]",ylabel="Relative change",title="(d) Actual total drift over this short window")
    ax.legend(fontsize=8); ax.grid(alpha=0.18)

    ax=axes[1,1]; obs_path=run.paths["observations.jsonl"]
    observations=[json.loads(line) for line in obs_path.read_text().splitlines() if line.strip()]
    definitions=[o["definition"] for o in observations if o["definition"].get("field")=="mom_i" and o["definition"].get("component")=="phi" and o["definition"].get("mode")==-1]
    observation_record={"status":"UNAVAILABLE","reason":"No nonzero ion toroidal momentum mode -1 observation"}
    if definitions:
        definition=definitions[0]
        selected=[o for o in observations if o["definition"]==definition]
        accepted={int(row["step"]):float(row["time_s"]) for row in history}
        if any(o["step"] not in accepted or o["time"]!=accepted[o["step"]] for o in selected):
            raise ValueError("Observation is not an accepted history row")
        ot=np.array([o["time"] for o in selected]); signal=np.array([complex(o["real"],o["imag"]) for o in selected])
        if np.any(np.diff(ot)<=0) or not np.all(np.isfinite(signal)):
            raise ValueError("Invalid observation samples")
        if abs(signal[0])>0:
            change=(signal-signal[0])/abs(signal[0])
            ax.plot(ot*1e9,change.real,color=COLORS[0],marker=".",label="Real change")
            ax.plot(ot*1e9,change.imag,color=COLORS[1],ls="--",label="Imaginary change")
            ax.legend(fontsize=8)
            observation_record={"status":"AVAILABLE","definition":definition,"time_s":ot.tolist(),"normalization_kg_m-2_s-1":float(abs(signal[0])),"relative_real":change.real.tolist(),"relative_imag":change.imag.tolist()}
    if observation_record["status"]=="UNAVAILABLE":
        ax.text(0.5,0.5,"No nonzero selected observation",ha="center",transform=ax.transAxes)
    ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    ax.set(xlabel="Time [ns]",ylabel="(coefficient - initial) / |initial|",
           title=r"(e) Actual $n=-1$ momentum response; no mode fit")
    ax.grid(alpha=0.18)

    ax=axes[1,2]; checks_path=run.directory/"checks.jsonl"
    checks=[json.loads(line) for line in checks_path.read_text().splitlines() if line.strip()]
    accepted={int(row["step"]):float(row["time_s"]) for row in history}
    ratios={}; zeros=0
    for name,color in zip(["gauss","current_ampere","current_divB","CT_increment"],COLORS):
        grouped={}; gates=set(); unavailable=0
        for c in checks:
            if c["name"]!=name or c["region"]!="all" or c["stage"] not in {"initial","endpoint"}: continue
            if c["gate"] not in {"hard","observe"}: raise ValueError("Unknown diagnostic gate")
            gates.add(c["gate"])
            if c["status"] in {"UNAVAILABLE","NOT_RUN"}:
                unavailable+=1; continue
            if c["status"] not in {"PASS","FAIL"}: raise ValueError("Unknown diagnostic status")
            if c["status"]=="FAIL" and c["gate"]=="hard": raise ValueError("Required diagnostic did not pass")
            if c["step"] not in accepted or c["time"]!=accepted[c["step"]]: raise ValueError("Diagnostic time does not match accepted history")
            raw,threshold=abs(float(c["raw"])),float(c["threshold"])
            if not np.isfinite(raw) or not np.isfinite(threshold) or threshold<=0: raise ValueError("Invalid diagnostic budget")
            grouped[c["step"]]=max(grouped.get(c["step"],0),raw/threshold)
        if not grouped: raise ValueError("Missing diagnostic: "+name)
        indices=sorted(grouped); values=np.array([grouped[i] for i in indices]); xt=np.array([accepted[i] for i in indices])
        keep=values>0; zeros+=int(np.sum(~keep))
        gate_label="/".join(sorted(gates))
        ax.plot(xt[keep]*1e9,values[keep],marker=".",ms=4,color=color,label=name+" ["+gate_label+"]")
        ratios[name]={"time_s":xt.tolist(),"maximum_ratio":values.tolist(),"gates":sorted(gates),"unavailable_records":unavailable}
    ax.axhline(1,color="0.4",ls="--",label="Recorded budget")
    ax.set(yscale="log",xlabel="Time [ns]",ylabel="max |residual| / threshold",
           title="(f) Diagnostic budgets: hard / observe as recorded")
    ax.legend(fontsize=7.5,loc="best"); ax.grid(alpha=0.18)
    fig.supxlabel(f"Native masks; no spatial smoothing. Fourier synthesis uses stored coefficients. {zeros} exact-zero budget points omitted on log axis. Short-window verification, not long-time stability.",fontsize=9)
    output.mkdir(parents=True,exist_ok=False)
    fig.savefig(output/"run_overview.png",dpi=180,facecolor="white")
    fig.savefig(output/"run_overview.pdf",facecolor="white")
    plt.close(fig)
    plotted={"time_s":times.tolist(),"step":steps.tolist(),"relative_drift":drift,"observation":observation_record,
             "budget_ratios":ratios,"exact_zero_budget_points":zeros,"R_cells_m":cell.R.tolist(),"Z_cells_m":cell.Z.tolist(),
             "active_cells":cell.active.tolist(),"R_vertices_m":vertex.R.tolist(),"Z_vertices_m":vertex.Z.tolist(),
             "active_vertices":vertex.active.tolist(),"B_phi_mode0_T":magnetic.real.tolist(),"phi_rad":phi.tolist(),
             "velocity_slice_Z_m":float(cell.Z[iz]),"u_i_phi_m_s":np.where(active,velocity,0).tolist(),
             "velocity_active_mask":active.tolist(),"first_snapshot":str(first),"last_snapshot":str(last)}
    (output/"plot-data.json").write_text(json.dumps(plotted,indent=2,allow_nan=False)+"\n")
    paths=[*run.paths.values(),first,last,checks_path]
    provenance={"run_directory":str(run.directory),"input_hashes":{str(p):sha(p) for p in paths},
                "verified_run_hashes":run.hashes,"script_sha256":sha(__file__),"analyzer_sha256":sha(analyze.__file__),
                "command":sys.argv,"python":platform.python_version(),"numpy":np.__version__,"matplotlib":matplotlib.__version__}
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    (output/"README.md").write_text("# Accepted run overview\n\nThis figure reads actual accepted snapshots, histories and diagnostic records. The dashed input contour differs from the solid staircase wall. Magnetic samples retain the vertex layout. The velocity panel evaluates the stored Fourier series and divides physical momentum by physical mass density; it does not divide Fourier coefficients. Inactive sites are masked. Conservation drifts are changes from the first accepted history row, not corrected error estimates. The momentum trace is not interpreted as an eigenmode or growth-rate fit. Diagnostic curves show maximum usage across available global norms at initial/endpoint stages; hard versus observe semantics are retained in the legend. An observe budget is not an acceptance gate. Exact zeros are omitted only on the logarithmic axis and remain in plot-data.json.\n")
    return {"output":str(output.resolve()),"steps":int(steps[-1]),"snapshots":[first.name,last.name],"active_cells":int(cell.active.sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    # Recover a known cosine before using the same convention on solver fields.
    phase=np.linspace(0,2*np.pi,17)
    if not np.allclose(real_fourier(np.array([[0.5],[2],[0.5]],dtype=complex),phase)[0],2+np.cos(phase),rtol=0,atol=1e-15):
        raise ValueError("Fourier reconstruction self-check failed")
    plt.rcParams.update({"font.size":10,"axes.titlesize":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    print(json.dumps(render(analyze.load_run(args.run),args.output)))


if __name__=="__main__":
    main()
