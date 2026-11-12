"""H13 width search with an explicitly recorded B_0.15 central constraint."""
from pathlib import Path
import json,time
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
from design_h import OUT,REG,compiled,objective,seq,dynamic,metrics,infidelity

file=OUT/'search/selected_h_long.json';selected=json.loads(file.read_text());p=selected['H13p']['controls'];N=13
area=sum(p['amplitudes']);cap=max(p['amplitudes']);rng=np.random.default_rng(2026102251)
_,nom,jac=compiled(N,np.pi/2,n=3)
center=objective(N,n=15,ns=15,mode='max',r=.15,endpoint=True)
def cb(v):f,g=center(jnp.asarray(v));return 1-f,-np.asarray(g)
cons=[dict(type='eq',fun=lambda v:np.asarray(nom(jnp.asarray(v))),jac=lambda v:np.asarray(jac(jnp.asarray(v)))),
      dict(type='ineq',fun=lambda v:area-sum(v[N:]),jac=lambda v:np.r_[np.zeros(N),-np.ones(N)]),
      dict(type='ineq',fun=lambda v:float(cb(v)[0]),jac=lambda v:cb(v)[1])]
v=np.r_[p['phases'],p['amplitudes']];starts=[v]
for scale in [.03,.12]:starts.append(np.r_[v[:N]+rng.normal(0,scale,N),np.clip(v[N:]+rng.normal(0,scale/4,N),.05,cap)])
records=[];path=OUT/'search/H13_width015_records.json'
(OUT/'search/H13_width015_protocol.json').write_text(json.dumps(dict(seed=2026102251,protected_half_width=.15,area_cap=area,peak_cap=cap,starts=[s.tolist() for s in starts],constraints='exact nominal; sampled smooth maximum for rectangular and finite-edge cases',selection='weighted independent rectangular and noisy-edge threshold coverage'),indent=2))
fg=objective(N,n=31,ns=15,mode='c',r=.5)
def fun(v):f,g=fg(jnp.asarray(v));return float(f),np.asarray(g)
for i,start in enumerate(starts):
    fit=minimize(fun,start,jac=True,method='SLSQP',bounds=[(None,None)]*N+[(.05,cap)]*N,constraints=cons,options=dict(maxiter=1000,ftol=2e-11))
    s=seq(fit.x,N,'H13c');r=dict(controls=s,success=bool(fit.success),message=str(fit.message),start=i,nominal=float(infidelity(s,0,0)),objective=float(fit.fun),margin=float(cb(fit.x)[0]))
    records.append(r);path.write_text(json.dumps(records,indent=2));print('H13 coverage B.15',i,r['success'],r['nominal'],r['margin'],flush=True)
valid=[r for r in records if r['success'] and r['nominal']<1e-12]
if not valid:raise RuntimeError('No feasible H13 width fit')
fg=objective(N,n=51,ns=25,mode='c',r=.5)
for r in sorted(valid,key=lambda r:r['objective'])[:2]:
    start=np.r_[r['controls']['phases'],r['controls']['amplitudes']]
    fit=minimize(fun,start,jac=True,method='SLSQP',bounds=[(None,None)]*N+[(.05,cap)]*N,constraints=cons,options=dict(maxiter=1000,ftol=2e-11))
    s=seq(fit.x,N,'H13c');row=dict(controls=s,success=bool(fit.success),message=str(fit.message),start=r['start'],nominal=float(infidelity(s,0,0)),objective=float(fit.fun),margin=float(cb(fit.x)[0]),stage='fine')
    records.append(row);path.write_text(json.dumps(records,indent=2));print('H13 fine',row['success'],row['nominal'],flush=True)
valid=[r for r in records if r['success'] and r['nominal']<1e-12]
for r in valid:
    r['selection_dynamic']=dynamic(r['controls'],n=61,r=.5);r['selection_rect']=metrics(r['controls'],n=201,r=.5,peak=True)
    r['selection_score']=.3*r['selection_rect']['fraction_below_1e4']+.7*r['selection_dynamic']['coverage']
chosen=max(valid,key=lambda r:r['selection_score']);selected=json.loads(file.read_text());selected['H13c']=dict(controls=chosen['controls'],selection=chosen,paired_area_cap=area,paired_peak_cap=cap,protected_half_width=.15)
file.write_text(json.dumps(selected,indent=2));path.write_text(json.dumps(records,indent=2));print('SELECTED H13c',chosen['selection_score'],flush=True)
