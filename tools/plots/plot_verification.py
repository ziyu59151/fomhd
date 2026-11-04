#!/usr/bin/env python3
"""Plot existing temporal/spatial MMS and a reviewed three-run time scan."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIELDS = ["n_i", "n_e", "mom_i", "mom_e", "p_i", "p_perp_e", "p_parallel_e", "B"]
LABELS = [r"$n_i$", r"$n_e$", r"$\pi_i$", r"$\pi_e$", r"$p_i$",
          r"$p_{\perp,e}$", r"$p_{\parallel,e}$", r"$B$"]
OPS = ["gradient", "curl", "divergence", "divergence_adjoint"]
OP_LABELS = ["G", "C", "DB", "DJ"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#332288", "#555555"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def orders(h, error):
    h, error = np.asarray(h, dtype=float), np.asarray(error, dtype=float)
    if h.shape != (3,) or error.shape != (3,) or not np.all(np.isfinite([h,error])) or np.any(h<=0) or np.any(error<=0) or np.any(np.diff(h)>=0):
        raise ValueError("Expected three positive finite errors and decreasing spacings")
    return np.log(error[:-1]/error[1:]) / np.log(h[:-1]/h[1:])


def parse_spatial(path):
    text = path.read_text()
    if "SPATIAL_VERIFICATION_PASS" not in text: raise ValueError("Spatial verification did not pass")
    levels=[]
    for line in text.splitlines():
        m=re.fullmatch(r"level n=(\d+) h=(\S+)",line)
        if m: levels.append({"n":int(m[1]),"h":float(m[2]),"errors":{}})
        m=re.fullmatch(r"\s+(gradient|curl|divergence|divergence_adjoint) interior=(\S+) wall=(\S+) global=(\S+)",line)
        if m:
            if not levels or m[1] in levels[-1]["errors"]: raise ValueError("Ambiguous spatial row")
            values=[float(m[i]) for i in [2,3,4]]
            if not np.all(np.isfinite(values)) or min(values)<0: raise ValueError("Invalid spatial error")
            levels[-1]["errors"][m[1]]=dict(zip(["interior","wall","global"],values))
    if [l["n"] for l in levels] != [8,16,32] or any(set(l["errors"])!=set(OPS) for l in levels): raise ValueError("Incomplete spatial levels")
    h=[l["h"] for l in levels]
    measured={op:orders(h,[l["errors"][op]["interior"] for l in levels]).tolist() for op in OPS}
    recorded=re.findall(r"^orders (\d+)->(\d+) gradient=(\S+) curl=(\S+) divergence=(\S+) divergence_adjoint=(\S+)$",text,re.M)
    if len(recorded)!=2: raise ValueError("Missing reported spatial orders")
    for j,row in enumerate(recorded):
        if [int(row[0]),int(row[1])] != [[8,16],[16,32]][j]: raise ValueError("Spatial order level mismatch")
        if not np.allclose([measured[op][j] for op in OPS],[float(v) for v in row[2:]],rtol=1e-12,atol=1e-12): raise ValueError("Spatial order mismatch")
    return levels, measured


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--evidence",type=Path,required=True);p.add_argument("--scan",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    source=a.evidence/"coupled-final-current.json"; spatial_path=a.evidence/"spatial-final-output.log"
    temporal=json.loads(source.read_text()); scan=json.loads(a.scan.read_text())
    if temporal.get("passed") is not True or temporal.get("accuracy_passed") is not True or scan.get("status")!="PASS": raise ValueError("Accepted evidence is required")
    levels=temporal["levels"]; dt=np.array([l["dt_s"] for l in levels])
    if [l["steps"] for l in levels]!=[8,16,32]: raise ValueError("Unexpected temporal levels")
    errors={f:[l["errors"][f] for l in levels] for f in FIELDS}
    temporal_orders={f:orders(dt,errors[f]).tolist() for f in FIELDS}
    if any(not np.allclose(temporal_orders[f],temporal["orders"][f],rtol=1e-12,atol=1e-12) for f in FIELDS): raise ValueError("Temporal order mismatch")
    spatial,spatial_orders=parse_spatial(spatial_path); h=np.array([l["h"] for l in spatial])
    if scan.get("scan_variable")!="time.dt" or scan.get("method")!="richardson" or scan["reference"]["kind"]!="none": raise ValueError("Expected a time Richardson scan")
    sdt=np.array([l["h"] for l in scan["levels"]]); diff=np.array([scan["differences"][k]["value"] for k in ["d01","d12"]])
    guard=float(scan["roundoff_guard"]["effective_norm"])
    if sdt.shape!=(3,) or diff.shape!=(2,) or np.any(sdt<=0) or np.any(np.diff(sdt)>=0) or not np.all(np.isfinite(sdt)) or not np.all(np.isfinite(diff)) or not np.isfinite(guard) or guard<=0 or np.any(diff<=guard): raise ValueError("Invalid or unresolved scan differences")
    if not np.isclose(sdt[0]/sdt[1],sdt[1]/sdt[2],rtol=1e-12,atol=0): raise ValueError("Only equal-ratio scans are supported")
    order=float(np.log(diff[0]/diff[1])/np.log(sdt[0]/sdt[1]))
    if not np.isclose(order,scan["orders"]["richardson"]["value"],rtol=1e-12,atol=1e-12): raise ValueError("Richardson order mismatch")
    if scan["observable"]["unit"]!="kg/(m^2*s)": raise ValueError("Unexpected momentum unit")
    plt.rcParams.update({"font.size":10,"axes.titlesize":11,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(2,2,figsize=(12.5,9),layout="constrained")
    fig.suptitle("A0 verification: measured convergence",fontsize=17)
    # Set log scales before limits; linear autoscaling can retain a negative lower bound.
    for ax in axes.flat: ax.set(xscale="log",yscale="log");ax.grid(alpha=0.18)
    ax=axes[0,0]
    for i,(f,label) in enumerate(zip(FIELDS,LABELS)):
        ax.plot(dt*1e9,errors[f],color=COLORS[i],marker="os^DvPXh"[i],ms=5,
                label=label+f" ({temporal_orders[f][0]:.3f}, {temporal_orders[f][1]:.3f})")
    ax.plot(dt*1e9,errors["B"][0]*(dt/dt[0])**2,"--",color="0.3",label=r"$\Delta t^2$ guide, anchored at B")
    ax.set(xlabel=r"Time step $\Delta t$ [ns]",ylabel="Scaled coefficient RMS error",
           title="(a) Coupled manufactured solution: eight evolved fields")
    ax.legend(fontsize=8,ncol=2,loc="upper left"); ax.set_ylim(top=max(max(v) for v in errors.values())*8)
    ax=axes[0,1]
    for i,op in enumerate(OPS):
        y=[l["errors"][op]["interior"] for l in spatial]
        ax.plot(h,y,color=COLORS[i],marker="os^D"[i],ms=5,ls="--" if i==1 else "-",
                markerfacecolor="white" if i==1 else COLORS[i],label=OP_LABELS[i]+f" ({spatial_orders[op][0]:.3f}, {spatial_orders[op][1]:.3f})")
    guide=spatial[0]["errors"]["gradient"]["interior"]*(h/h[0])**4
    ax.plot(h,guide,"--",color="0.3",label=r"$h^4$ guide, anchored at G, N=8")
    ax.set(xlabel="Grid spacing h [m]",ylabel="Interior relative maximum error",title="(b) Staggered spatial MMS: interior accuracy")
    ax.legend(fontsize=8,ncol=2,loc="upper left");ax.set_ylim(top=max(l["errors"][op]["interior"] for l in spatial for op in OPS)*8)
    ax=axes[1,0]
    for i,op in enumerate(["gradient","divergence_adjoint"]):
        for region,style in [("interior","-"),("global","--")]:
            y=[l["errors"][op][region] for l in spatial]
            orders(h,y)
            ax.plot(h,y,style,marker="os"[i],color=COLORS[i],markerfacecolor="white" if region=="global" else COLORS[i],label=("G" if i==0 else "DJ")+" "+region)
    ax.set(xlabel="Grid spacing h [m]",ylabel="Region-normalized maximum error",title="(c) Interior and global errors remain distinct")
    ax.legend(fontsize=9,ncol=2,loc="upper left");ax.set_ylim(top=0.5)
    ax.text(0.97,0.05,"No global fourth-order wall claim.\nExact wall zeros remain in the saved data.",ha="right",transform=ax.transAxes,fontsize=9,bbox={"facecolor":"white","edgecolor":"none","alpha":0.85})
    ax=axes[1,1]
    ax.plot(sdt[:2]*1e9,diff,"o-",color=COLORS[0],label="Differences between numerical runs")
    ax.plot(sdt[:2]*1e9,diff[0]*(sdt[:2]/sdt[0])**2,"--",color="0.3",label="Order-2 guide at coarse difference")
    ax.axhline(guard,color=COLORS[1],ls=":",label=f"Recorded roundoff guard: {guard:.0e}")
    ax.set(xlabel=r"Coarser time step $\Delta t$ [ns]",ylabel=r"Momentum difference [kg/(m$^2$ s)]",
           title=f"(d) Actual CLI refinement: measured order {order:.4f}")
    ax.set_ylim(guard*0.6,max(diff)*8);ax.legend(fontsize=8,loc="upper left")
    ax.text(0.04,0.12,"Three runs at one final time.\nDifferences are not exact-solution errors.",transform=ax.transAxes,fontsize=9)
    for ax in axes.flat:
        lower,upper=ax.get_ylim()
        if not 0<lower<upper or any(np.any((np.asarray(line.get_ydata())<lower)|(np.asarray(line.get_ydata())>upper)) for line in ax.lines):
            raise ValueError("Logarithmic plot limits hide recorded data")
    fig.supxlabel("Legend pairs give adjacent measured orders. Manufactured solutions verify implementation; they are not experimental validation.",fontsize=10)
    a.output.mkdir(parents=True,exist_ok=False)
    fig.savefig(a.output/"a0_verification_summary.png",dpi=180,facecolor="white");fig.savefig(a.output/"a0_verification_summary.pdf",facecolor="white");plt.close(fig)
    plotted={"dt_s":dt.tolist(),"temporal_errors":errors,"temporal_orders":temporal_orders,"spatial_levels":spatial,"spatial_orders":spatial_orders,"scan_dt_s":sdt.tolist(),"scan_differences":diff.tolist(),"scan_roundoff_guard":guard,"scan_order":order,
             "temporal_error_definition":"norm(error)/(scale*sqrt(size)): coefficient RMS divided by prescribed test scale",
             "spatial_error_definition":"max absolute error / max analytic magnitude in the stated region; wall uses global reference",
             "scan_norm":scan["observable"]["norm"],"scan_observable":scan["observable"]}
    (a.output/"plot-data.json").write_text(json.dumps(plotted,indent=2,allow_nan=False)+"\n")
    provenance={"inputs":{str(f.resolve()):sha(f) for f in [source,spatial_path,a.scan]},"script_sha256":sha(__file__),"command":sys.argv,"python":platform.python_version(),"numpy":np.__version__,"matplotlib":matplotlib.__version__}
    (a.output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    (a.output/"README.md").write_text("# Verification figure\n\nTemporal errors are coefficient RMS divided by each prescribed test scale. Spatial errors use the analytic maximum within their stated region; the comparison does not establish global fourth-order wall accuracy. Log axes contain only positive recorded values. Exact wall zeros are retained in plot-data.json. The final panel shows the reviewed norm of differences between three numerical runs, not error against an exact physical wave. Order-2 and order-4 guides are explicitly anchored at measured coarse points.\n")
    print(json.dumps({"output":str(a.output.resolve()),"temporal_order_range":[min(min(v) for v in temporal_orders.values()),max(max(v) for v in temporal_orders.values())],"scan_order":order}))


if __name__=="__main__":
    main()
