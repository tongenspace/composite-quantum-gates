"""Width-oriented X13 search under the existing depth design's resources.

This writes only new records; the canonical controls are not modified.
"""
from pathlib import Path
import argparse,importlib.util,json,sys,time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64',True)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'studies/control_refinement')]
from search import compiled
from rectangular import metrics,infidelity
spec=importlib.util.spec_from_file_location('width_objectives',ROOT/'studies/objective_selection/search_width.py')
width=importlib.util.module_from_spec(spec);spec.loader.exec_module(width)

def split(s,N):
    p=list(s['phases']);a=list(s['amplitudes'])
    while len(p)<N:
        i=int(np.argmax(a));v=a[i]/2
        p[i:i+1]=[p[i],p[i]];a[i:i+1]=[v,v]
    return np.r_[p,a]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--starts',type=int,default=4)
    ap.add_argument('--polish',action='store_true');args=ap.parse_args()
    reg=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
    old=reg['X13c'];N=13;area=sum(old['amplitudes']);cap=max(old['amplitudes'])
    seed=2026102130;rng=np.random.default_rng(seed)
    base=np.r_[old['phases'],old['amplitudes']]
    starts=[base,split(reg['X11c'],N)]
    for i,scale in [(0,.10),(1,.18),(0,.35),(1,.5)]:
        v=starts[i];starts.append(np.r_[v[:N]+rng.normal(0,scale,N),np.clip(v[N:]+rng.normal(0,scale/4,N),.05,cap)])
    records=[]
    if args.polish:
        records=json.loads((HERE/'search_records.json').read_text())
        valid=[r for r in records if r['success'] and r['nominal']<1e-12 and r['grid']>=61]
        valid.sort(key=lambda r:r['wide_metrics']['fraction_below_1e4'],reverse=True)
        starts=[np.r_[r['controls']['phases'],r['controls']['amplitudes']] for r in valid[:args.starts]]
    (HERE/'search_protocol.json').write_text(json.dumps(dict(seed=seed,baseline_name='X13c (renamed X13p in revised manuscript)',
        area_cap=area,peak_cap=cap,N=N,protected_half_width=.20,threshold=1e-4,
        starts=[v.tolist() for v in starts],selection='Successful feasible fits, largest independently checked connected area'),indent=2))
    _,nom,jac=compiled(N,np.pi,n=3)
    center=width.objective(N,np.pi,41 if args.polish else 21,r=.20,central=True)
    def cb(v):
        f,g=center(jnp.asarray(v),1e-4);return 1-f,-np.asarray(g)
    constraints=[dict(type='eq',fun=lambda v:np.asarray(nom(jnp.asarray(v))),jac=lambda v:np.asarray(jac(jnp.asarray(v)))),
       dict(type='ineq',fun=lambda v:float(cb(v)[0]),jac=lambda v:cb(v)[1]),
       dict(type='ineq',fun=lambda v:area-sum(v[N:]),jac=lambda v:np.r_[np.zeros(N),-np.ones(N)])]
    stages=[(31,1e-3,650),(41,1e-4,900),(61,1e-4,800)] if not args.polish else [(81,1e-4,1000)]
    for i,start in enumerate(starts[:args.starts]):
        v=start
        for n,tau,it in stages:
            fg=width.objective(N,np.pi,n)
            def fun(v):
                f,g=fg(jnp.asarray(v),tau);return float(f),np.asarray(g)
            t=time.perf_counter()
            result=minimize(fun,v,jac=True,method='SLSQP',bounds=[(None,None)]*N+[(.05,cap)]*N,
                            constraints=constraints,options=dict(ftol=2e-11,maxiter=it))
            v=result.x
            s=old|dict(name='X13w',phases=((v[:N]+np.pi)%(2*np.pi)-np.pi).tolist(),amplitudes=v[N:].tolist(),
                       origin='studies/x13_width/search_x13_width.py; seed='+str(seed))
            m=metrics(s,r=.5,n=401,peak=True)
            row=dict(controls=s,start=i,grid=n,tau=tau,success=bool(result.success),message=str(result.message),
               iterations=int(result.nit),seconds=time.perf_counter()-t,nominal=float(infidelity(s,0,0)),
               constraint_margin=float(cb(v)[0]),wide_metrics=m,stage='fine' if args.polish else 'multistart')
            records.append(row);(HERE/'search_records.json').write_text(json.dumps(records,indent=2))
            print('X13w',i,n,'success',result.success,'coverage',m['fraction_below_1e4'],'nominal',row['nominal'],
                  'margin',row['constraint_margin'],'seconds',round(row['seconds'],1),flush=True)
    print('Search complete',flush=True)
if __name__=='__main__':main()
