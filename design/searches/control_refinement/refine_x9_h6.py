"""Bounded multistart refinement of X9c and H6; no registry mutation.

Full-unitary average gate infidelity, fixed Omega_0*T=pi and fixed detuning.
Nominal matching and original peak/area bounds are imposed during polishing.
All candidates, including failed fits, are saved for independent selection.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
from search import compiled,evaluate
from rectangular import metrics
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gate',choices=['X9c','H6','both'],default='both')
    parser.add_argument('--starts',type=int,default=13);args=parser.parse_args()
    reg=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
    original=HERE/'input_controls.json';oldreg=json.loads(original.read_text())
    for name in ['X9c','H6']:
        if name not in oldreg:oldreg[name]=reg[name]
    original.write_text(json.dumps(oldreg,indent=2)+'\n')
    for ci,name in enumerate(['X9c','H6']):
        if args.gate not in [name,'both']:continue
        old=oldreg[name];N=old['N'];area=sum(old['amplitudes']);cap=max(old['amplitudes'])
        seed=202610146+ci;rng=np.random.default_rng(seed);base=np.r_[old['phases'],old['amplitudes']]
        starts=[base]
        for k in range(args.starts-1):
            scale=[.04,.12,.3,.6][k//3%4]
            starts.append(np.r_[base[:N]+rng.normal(0,scale,N),np.clip(base[N:]+rng.normal(0,scale/3,N),.02,cap)])
        bounds=[(None,None)]*N+[(1e-5,cap)]*N
        records=[];fitted=[];fg,nom,jac=compiled(N,old['theta'],n=25)
        def seq(x):return {**old,'phases':((x[:N]+np.pi)%(2*np.pi)-np.pi).tolist(),'amplitudes':x[N:].tolist(),
            'origin':f'studies/control_refinement/refine_x9_h6.py; seed={seed}'}
        def record(fit,stage,k,r=.15):
            s=seq(fit.x);m=evaluate(s,n=401 if stage=='exact_polish' else 201)
            wide=metrics(s,r=.5,n=401)
            row=dict(name=name,stage=stage,start_index=k,training_halfwidth=r,controls=s,metrics=m,
                wide_domain_metrics=wide,success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit))
            records.append(row);(HERE/(name.lower()+'_refinement_records.json')).write_text(json.dumps(records,indent=2)+'\n')
            print(name,stage,k,'mean',m['native_mean'],m['peak_mean'],'I0',m['nominal'],
                'wide_fraction',wide['fraction_below_1e4'],'area',sum(s['amplitudes']),flush=True)
            return m
        for k,start in enumerate(starts):
            weight=1. if k%2==0 else 10.
            def f(x):v,g=fg(jnp.asarray(x),weight,area);return float(v),np.asarray(g)
            fit=minimize(f,start,jac=True,method='L-BFGS-B',bounds=bounds,
                options=dict(maxiter=1600,ftol=1e-13,gtol=1e-8,maxls=40,maxcor=20))
            m=record(fit,'penalty',k);fitted.append((fit.x,m))
        # Include original and diverse low-mean fits; repeat at a broader interval.
        seeds=[base]+[v[0] for v in sorted(fitted,key=lambda v:v[1]['native_mean']+v[1]['peak_mean'])[:6]]
        for r in [.15,.22]:
            fg,nom,jac=compiled(N,old['theta'],n=41,r=r)
            constraints=[dict(type='eq',fun=lambda x:np.asarray(nom(jnp.asarray(x))),
                jac=lambda x:np.asarray(jac(jnp.asarray(x)))),dict(type='ineq',
                fun=lambda x:area-sum(x[N:]),jac=lambda x:np.r_[np.zeros(N),-np.ones(N)])]
            for k,start in enumerate(seeds):
                def f(x):v,g=fg(jnp.asarray(x),0.,area);return float(v),np.asarray(g)
                fit=minimize(f,start,jac=True,method='SLSQP',bounds=bounds,constraints=constraints,
                    options=dict(maxiter=1600,ftol=1e-12))
                record(fit,'exact_polish',k,r)
        print(name,'completed',len(records),'fits',flush=True)
if __name__=='__main__':main()
