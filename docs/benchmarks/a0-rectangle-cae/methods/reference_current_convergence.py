"""Measure independent-reference current convergence separately from B shape."""
import argparse
from pathlib import Path
import json

import numpy as np

from box_reference import evaluate, load
from prepare import choose_branch


def integrate(references: Path, output: Path):
    q,w=np.polynomial.legendre.leggauss(33)
    r,z=np.meshgrid(15+.6*q,.6*q)
    weight=(r*np.outer(w,w)*.6**2).ravel()
    dist=np.minimum(.6-abs(r-15),.6-abs(z)).ravel()
    fields={}
    branches={}
    for p in (12,14,16,18):
        path=references / f'ritz-R15-n2e19-a0p6-p{p}.npz'
        if not path.exists():
            continue
        ref=load(path)
        i=choose_branch(ref)
        fields[p]=evaluate(ref,i,r,z)
        branches[p]=float(ref['omega'][i]/(2*np.pi))
    target=fields[16]
    out={'scope':'Independent reference resolution check only. Same physical points and fixed 0.20 m collar.',
         'orders':{}, 'concentration':{}}
    sort=np.argsort(dist)
    for p,data in fields.items():
        jenergy=weight*np.sum(abs(data['J'])**2,axis=1)
        mass=np.cumsum(jenergy[sort])/jenergy.sum()
        out['concentration'][p]=dict(
            frequency_hz=branches[p],
            d50_J_energy_from_wall_m=float(np.interp(.5,mass,dist[sort])),
            J_energy_within_0p05m_fraction=float(jenergy[dist<.05].sum()/jenergy.sum()),
            J_energy_within_0p20m_fraction=float(jenergy[dist<.20].sum()/jenergy.sum()),
        )
    for p in fields:
        if p==16:
            continue
        candidate=fields[p]
        b=target['B'][:,1]
        c=candidate['B'][:,1]
        phase=np.vdot(c,weight*b)/np.vdot(c,weight*c)
        values={}
        for field in ('B','J'):
            exact=target[field]
            current=candidate[field]*phase
            for tag,idx in (('domain',dist>=0),('inner',dist>=.2),('collar',dist<.2)):
                rms=np.sum(weight[idx,None]*abs(current[idx]-exact[idx])**2)
                scale=np.sum(weight[idx,None]*abs(exact[idx])**2)
                values[field+'_'+tag+'_relative_RMS']=float(np.sqrt(rms/scale))
        out['orders'][p]=values
    output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    integrate(args.references, args.output)
