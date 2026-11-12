"""Repair H15c's independently detected off-mesh threshold overshoot."""
import json
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
from design_h import OUT,compiled,objective,seq,infidelity,dynamic,metrics
file=OUT/'search/selected_h_fifteen.json';selected=json.loads(file.read_text());p=selected['H15p']['controls'];c=selected['H15c']['controls'];N=15
area=sum(p['amplitudes']);cap=max(p['amplitudes']);_,nom,jac=compiled(N,np.pi/2,n=3)
center=objective(N,n=41,ns=41,mode='max',r=.2,endpoint=True);fg=objective(N,n=61,ns=31,mode='c',r=.5)
def cb(v):f,g=center(jnp.asarray(v));return 1-f,-np.asarray(g)
def fun(v):f,g=fg(jnp.asarray(v));return float(f),np.asarray(g)
cons=[dict(type='eq',fun=lambda v:np.asarray(nom(jnp.asarray(v))),jac=lambda v:np.asarray(jac(jnp.asarray(v)))),
      dict(type='ineq',fun=lambda v:area-sum(v[N:]),jac=lambda v:np.r_[np.zeros(N),-np.ones(N)]),
      dict(type='ineq',fun=lambda v:float(cb(v)[0]),jac=lambda v:cb(v)[1])]
start=np.r_[c['phases'],c['amplitudes']]
fit=minimize(fun,start,jac=True,method='SLSQP',bounds=[(None,None)]*N+[(.05,cap)]*N,constraints=cons,options=dict(maxiter=1200,ftol=1e-12))
s=seq(fit.x,N,'H15c');row=dict(controls=s,success=bool(fit.success),message=str(fit.message),nominal=float(infidelity(s,0,0)),objective=float(fit.fun),margin=float(cb(fit.x)[0]),constraint_grid=41,objective_rect_grid=61,objective_edge_grid=31,protected_half_width=.2)
(OUT/'search/H15_fine_constraint.json').write_text(json.dumps(row,indent=2));print('H15 constraint refinement',row['success'],row['nominal'],row['margin'],flush=True)
assert row['success'] and row['nominal']<1e-12
row['selection_dynamic']=dynamic(s,n=61,r=.5);row['selection_rect']=metrics(s,n=201,r=.5,peak=True)
selected['H15c']=dict(controls=s,selection=row,paired_area_cap=area,paired_peak_cap=cap,protected_half_width=.2)
file.write_text(json.dumps(selected,indent=2));print('Updated H15c; independent revalidation required.',flush=True)
