"""Deterministic, exact-nominal variable-area X13c/X15c searches.

Equal segment durations, amplitudes in [0.05,2], total area <= N*pi.
The mean objective equally weights native/common-peak half-width-0.15 boxes.
No existing selected controls are replaced by this program.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'studies/control_refinement')]
from search import compiled,evaluate
from rectangular import infidelity,metrics


def split_controls(s,N):
    p=list(s['phases']);a=list(s['amplitudes'])
    while len(p)<N:
        i=int(np.argmax(a));value=a[i]/2
        a[i:i+1]=[value,value];p[i:i+1]=[p[i],p[i]]
    return np.r_[p,a]


def starts_for(N,registry,seed):
    rng=np.random.default_rng(seed)
    starts=[];labels=[]
    for family in ['X13a','X13b']:
        s=registry[family]
        phases=list(s['phases']);amps=list(s['amplitudes'])
        if N==15:
            phases += [.731*np.pi,1.731*np.pi];amps += [1.,1.]
        starts.append(np.r_[phases,amps]);labels.append(family+(' + nominal identity pair' if N==15 else ''))
    v=split_controls(registry['X11c'],N)
    starts.append(v);labels.append('split X11c; fully reoptimized')
    for scale,i in [(.12,0),(.35,2)]:
        v=starts[i]
        starts.append(np.r_[v[:N]+rng.normal(0,scale,N),
                            np.clip(v[N:]+rng.normal(0,scale/3,N),.05,2.)])
        labels.append(f'perturbed start {i}, scale {scale}')
    starts.append(np.r_[rng.uniform(-np.pi,np.pi,N),rng.uniform(.3,1.2,N)])
    labels.append('global random start')
    return starts,labels


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gate',choices=['X13c','X15c'],required=True)
    parser.add_argument('--starts',type=int,default=6);parser.add_argument('--iterations',type=int,default=900)
    parser.add_argument('--stage',choices=['all','polish'],default='all')
    args=parser.parse_args();name=args.gate;N=int(name[1:-1]);seed=202610190+N
    registry=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
    seeds,labels=starts_for(N,registry,seed)
    seed_path=BASE/(name.lower()+'_seeds.json')
    if seed_path.exists():
        saved=json.loads(seed_path.read_text());seeds=[np.asarray(v) for v in saved['vectors']];labels=saved['labels']
    else:seed_path.write_text(json.dumps(dict(seed=seed,vectors=[v.tolist() for v in seeds],labels=labels),indent=2))
    area=float(N);bounds=[(None,None)]*N+[(.05,2.)]*N
    record_path=BASE/(name.lower()+'_search_records.json')
    records=json.loads(record_path.read_text()) if args.stage=='polish' else []
    template=dict(name=name,N=N,theta=float(np.pi),model='drive',durations_T=[1.]*N,
                  origin=f'studies/x_extension/extend_x.py; seed={seed}')
    def control(v):return template | dict(phases=((v[:N]+np.pi)%(2*np.pi)-np.pi).tolist(),amplitudes=v[N:].tolist())
    def record(fit,stage,index,grid):
        s=control(fit.x);m=evaluate(s,n=201 if stage=='search' else 401)
        row=dict(controls=s,metrics=m,stage=stage,start_index=index,seed=seed,
                 grid=grid,half_width=.15,peak_weight=.5,area_cap=N,amplitude_bounds=[.05,2.],
                 success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit))
        records.append(row);record_path.write_text(json.dumps(records,indent=2))
        print(name,stage,index,'ok',fit.success,'means',m['native_mean'],m['peak_mean'],
              'I0',m['nominal'],'area/pi',sum(s['amplitudes']),'peak',max(s['amplitudes']),flush=True)
    stages=[('search',25,list(enumerate(seeds[:args.starts]))),('polish',51,None)]
    if args.stage=='polish':stages=stages[1:]
    for stage,n,stage_starts in stages:
        fg,nom,jac=compiled(N,np.pi,n=n,peak_weight=.5)
        def value(v,weight=0.):
            loss,gradient=fg(jnp.asarray(v),weight,area)
            return float(loss),np.asarray(gradient)
        constraints=[dict(type='eq',fun=lambda v:np.asarray(nom(jnp.asarray(v))),
                          jac=lambda v:np.asarray(jac(jnp.asarray(v)))),
                     dict(type='ineq',fun=lambda v:area-np.sum(v[N:]),
                          jac=lambda v:np.r_[np.zeros(N),-np.ones(N)])]
        if stage_starts is None:
            eligible=[r for r in records if r['metrics']['nominal']<1e-7]
            eligible.sort(key=lambda r:r['metrics']['native_mean']+r['metrics']['peak_mean'])
            stage_starts=[(r['start_index'],np.r_[r['controls']['phases'],r['controls']['amplitudes']]) for r in eligible[:3]]
            if not stage_starts:raise RuntimeError('No nominally matched coarse candidate')
        for i,start in stage_starts:
            start=np.asarray(start)
            if stage=='search' and i>=3:
                warm=minimize(lambda v:value(v,5.),start,jac=True,method='L-BFGS-B',bounds=bounds,
                              options=dict(maxiter=600,ftol=1e-13,gtol=1e-8,maxls=40))
                start=warm.x
            fit=minimize(value,start,jac=True,method='SLSQP',bounds=bounds,constraints=constraints,
                         options=dict(maxiter=args.iterations,ftol=1e-12))
            record(fit,stage,i,n)
            if stage=='polish':
                # Restore the nominal constraints with the smallest parameter
                # change; preserve the mean-fit termination separately above.
                center=fit.x.copy()
                projection=minimize(lambda v:(float(np.sum((v-center)**2)),2*(v-center)),
                    center,jac=True,method='SLSQP',bounds=bounds,constraints=constraints,
                    options=dict(maxiter=250,ftol=1e-15))
                record(projection,'nominal_projection',i,n)
    print(name,'search complete; independent selection required.',flush=True)


if __name__=='__main__':main()
