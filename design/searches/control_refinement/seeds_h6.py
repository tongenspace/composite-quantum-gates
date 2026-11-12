"""Additional H6 starts made by splitting retained H3/H4/H5 pulses.

Splitting preserves the resonant area but changes duration/detuning response;
each six-pulse candidate is fully reoptimized and independently evaluated.
"""
from pathlib import Path
import json
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
from search import compiled,evaluate
from rectangular import metrics
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
reg=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
old=json.loads((HERE/'input_controls.json').read_text())['H6'];N=6;area=sum(old['amplitudes']);cap=max(old['amplitudes'])
rng=np.random.default_rng(202610148);starts=[]
for name,parts in [('H3',[2,2,2]),('H4',[2,1,2,1]),('H4',[1,2,1,2]),('H5',[1,1,2,1,1])]:
    source=reg[name];ph=[];a=[]
    for phi,amp,n in zip(source['phases'],source['amplitudes'],parts):ph.extend([phi]*n);a.extend([amp/n]*n)
    base=np.r_[ph,a]
    starts.append((name,base))
    for scale in [.15,.6]:starts.append((name,np.r_[base[:N]+rng.normal(0,scale,N),np.clip(base[N:]+rng.normal(0,scale/3,N),.02,cap)]))
records=[]
bounds=[(None,None)]*N+[(1e-5,cap)]*N
for index,(source,start) in enumerate(starts):
    peak_weight=.5 if index%2 else .15
    fg,nom,jac=compiled(N,old['theta'],n=31,peak_weight=peak_weight)
    def fun(x,w):v,g=fg(jnp.asarray(x),w,area);return float(v),np.asarray(g)
    fit=minimize(lambda x:fun(x,5.),start,jac=True,method='L-BFGS-B',bounds=bounds,
        options=dict(maxiter=1600,ftol=1e-13,gtol=1e-8,maxls=40))
    cons=[dict(type='eq',fun=lambda x:np.asarray(nom(jnp.asarray(x))),jac=lambda x:np.asarray(jac(jnp.asarray(x)))),
        dict(type='ineq',fun=lambda x:area-sum(x[N:]),jac=lambda x:np.r_[np.zeros(N),-np.ones(N)])]
    fit=minimize(lambda x:fun(x,0.),fit.x,jac=True,method='SLSQP',bounds=bounds,constraints=cons,
        options=dict(maxiter=1600,ftol=1e-12))
    x=fit.x;s={**old,'phases':((x[:N]+np.pi)%(2*np.pi)-np.pi).tolist(),'amplitudes':x[N:].tolist(),
        'origin':'studies/control_refinement/seeds_h6.py; seed=202610148'}
    m=evaluate(s,n=401);wide=metrics(s,r=.5,n=401)
    records.append(dict(name='H6',stage='split_seed_polish',start_index=index,seed_source=source,peak_weight=peak_weight,
        controls=s,metrics=m,wide_domain_metrics=wide,success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit)))
    (HERE/'h6_split_seed_records.json').write_text(json.dumps(records,indent=2)+'\n')
    print('H6 split',index,source,peak_weight,'means',m['native_mean'],m['peak_mean'],'nominal',m['nominal'],'wide',wide['fraction_below_1e4'],flush=True)
