#!/usr/bin/env python3
"""Replay the archived disk pressure check with read-only energy monitoring."""
from __future__ import annotations
import argparse
import hashlib
import shutil
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'tests/warp')]
from fomhd import io,stage,stability
from fomhd.core import forward
from fomhd.time import advance
from fomhd.types import Budget,Norm,cumulative_record
from tools import warp_spatial_study as study
from tools.warp_tokamak_cae import observation,snapshot_sources,host


def observe_only(p):
    """Keep existing thresholds; change only authorized physical-budget gates."""
    for key,(unit,norm,hard) in io._massless_budget_registry(p).items():
        if unit not in ('W','J','kg*m/s','kg*m^2/s'):continue
        if key not in p.budgets:
            p.budgets[key]=Budget(0.,0.,1.,unit,Norm.weighted_l2 if norm=='weighted_L2' else Norm(norm),False)
        else:p.budgets[key].hard=False


def run(output,device,nodes,steps):
    if device=='cuda' and not os.environ.get('SLURM_JOB_ID'):raise ValueError('CUDA requires owned Slurm allocation')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(status='initializing',device=device,case=dict(kind='archived_pressure_control',nodes=nodes,K=1,Nphi=5,nphi=1,
                wall='fixed closed staircase',profile_assumption='retained disk/wall pressure study; gates only changed'),
                source_hashes=snapshot_sources(output))
    for name in ('tools/warp_pressure_observe.py','tools/warp_spatial_study.py','tests/warp/massless_verification_fixture.py','tests/warp/foundation_fixture.py','tests/warp/mms_fixture.py'):
        source=ROOT/name;target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        report['source_hashes'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
    checks=Counter();history=[]
    try:
        context=study.build_context(nodes,'disk',device,24<<30)
        p=context.p;observe_only(p)
        state=stage.closed_reference(context,study.ELECTRON_DENSITY,study.P0,study.P0,study.TOROIDAL_MAGNETIC)
        nb=context.op.Nb;harmonic=state.B[nb:2*nb]
        harmonic*=.1*math.log(2)/context.compute.scalar((context.op.topology_flux_row@harmonic)[0])
        phi=2*np.pi*np.arange(5)/5;g=context.geometry
        variation=study.analytic_pressure(np.asarray(g.cells.R)[None,:],np.asarray(g.cells.Z)[None,:],phi[:,None],'wall')[0]
        pressure=forward(context.compute.asarray((study.P0+variation).reshape(-1)),context.op.Nc,context)
        state.fluid[1].perp=pressure.copy();state.fluid[1].parallel=pressure.copy()
        captured={};ledger=cumulative_record();committed=stage.initialize(state,context,ledger,False,captured)
        frame=SimpleNamespace(**captured);baseline=frame.totals.energy
        estimate=stability.stage_frequency_estimate(frame,context)
        p.dt=.05/estimate['omega_estimate'];p.nsteps=steps;p.dt_guard='error'
        history.append(observation(committed.state,frame,ledger,context,baseline))
        with (output/'perturbed-checks.jsonl').open('w') as log:
            def record(records):
                for c in records:
                    checks['unavailable' if not c.available else 'pass' if c.passed else 'hard_fail' if c.hard else 'observe_fail']+=1
                    log.write(json.dumps(io._check_json(c))+'\n')
            record(frame.checks)
            for _ in range(steps):
                result=advance(committed,ledger,context);record(result.checks)
                history.append(observation(committed.state,result.frame,ledger,context,baseline))
        report.update(status='completed',accepted_steps=committed.state.step,dt_s=p.dt,
                      initial_frequency_estimate=estimate,checks=dict(checks))
        np.savez_compressed(output/'input_mapping.npz',R=g.cells.R,Z=g.cells.Z,polygon=np.array([[q.R,q.Z] for q in g.polygon]))
    except Exception as error:
        report.update(status='failed',failure=dict(code=getattr(error,'code',type(error).__name__),message=str(error)))
        if getattr(error,'checks',None):
            (output/'failure-checks.json').write_text(json.dumps([io._check_json(c) for c in error.checks],indent=2)+'\n')
    if history:
        (output/'perturbed-history.json').write_text(json.dumps([{k:v for k,v in h.items() if k in ('time_s','step','energy','energy_raw','momentum')} for h in history],indent=2)+'\n')
        np.savez_compressed(output/'perturbed-signals.npz',time_s=np.array([h['time_s'] for h in history]),**{k:np.stack([h[k] for h in history]) for k in ('B','BN','density','velocity')})
    report['elapsed_s']=time.monotonic()-started
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['status']=='completed' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--nodes',type=int,default=20);p.add_argument('--steps',type=int,default=8)
    a=p.parse_args();raise SystemExit(run(a.output,a.device,a.nodes,a.steps))
