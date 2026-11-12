"""Matched-resource, threshold-oriented X15/H15 searches; existing controls frozen.

Common-peak detuning throughout. Minimize a smooth uncovered fraction on the
fixed half-width .5 square: I/(I+tau). Continuation tau=.001 -> .0001 avoids
assigning excessive importance to already deep low-error regions. No fit is
selected on its training objective alone.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import jax,jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64',True)
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'studies/control_refinement')]
from search import compiled
from rectangular import metrics,infidelity

def objective(N,theta,n,r=.5,central=False):
    u=np.linspace(-r,r,n) if central else (np.arange(n)+.5)*2*r/n-r
    D,E=np.meshgrid(u,u)
    e=jnp.array(E.ravel());d=jnp.array(D.ravel())
    c,s=np.cos(theta/2),np.sin(theta/2)
    def errors(v):
        z=jnp.zeros_like(e);o=jnp.ones_like(e);det=jnp.max(v[N:])*d
        def step(q,pa):
            w,x,y,z=q;p,a=pa;om=a*(1+e);norm=jnp.sqrt(om*om+det*det)
            co=jnp.cos(jnp.pi*norm/2);k=jnp.pi/2*jnp.sinc(norm/2)
            px=k*om*jnp.cos(p);py=-k*om*jnp.sin(p);pz=-k*det
            return (co*w-px*x-py*y-pz*z,co*x+w*px+py*z-pz*y,
                    co*y+w*py+pz*x-px*z,co*z+w*pz+px*y-py*x),None
        (w,x,y,z),_=jax.lax.scan(step,(o,z,z,z),(v[:N],v[N:]))
        return (2/3)*((c*x-s*w)**2+y*y+z*z)
    def loss(v,tau):
        I=errors(v)
        if central:
            # Unnormalized smooth maximum is >= every sampled I/tau.
            return jax.scipy.special.logsumexp(50*I/tau)/50
        return jnp.mean(I/(I+tau))
    return jax.jit(jax.value_and_grad(loss))

def main():
    p=argparse.ArgumentParser();p.add_argument('--gate',choices=['X15c','H15'],required=True)
    p.add_argument('--starts',type=int,default=6);p.add_argument('--polish',action='store_true');args=p.parse_args()
    registry=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
    old=registry[args.gate];N=old['N'];theta=old['theta'];area=sum(old['amplitudes']);cap=max(old['amplitudes'])
    name='X15w' if args.gate=='X15c' else 'H15w'
    seed=202610210+(theta<2);rng=np.random.default_rng(seed)
    base=np.r_[old['phases'],old['amplitudes']]
    starts=[base]
    if args.gate=='X15c':
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('extend_x',ROOT/'studies/x_extension/extend_x.py');mod=module_from_spec(spec);spec.loader.exec_module(mod)
        v=mod.split_controls(registry['X11c'],15);v[N:]*=min(1,area/sum(v[N:]),cap/max(v[N:]))
        starts.append(v)
    for scale in [.025,.08,.18,.35,.6]:
        starts.append(np.r_[base[:N]+rng.normal(0,scale,N),np.clip(base[N:]+rng.normal(0,scale/3,N),.05,cap)])
    _,nom,jac=compiled(N,theta,n=3)
    central_grid=41 if args.polish else 21
    center=objective(N,theta,central_grid,r=.20,central=True)
    def center_bound(v):
        f,g=center(jnp.asarray(v),1e-4);return 1-f,-np.array(g)
    constraints=[dict(type='eq',fun=lambda v:np.asarray(nom(jnp.asarray(v))),jac=lambda v:np.asarray(jac(jnp.asarray(v)))),
                 dict(type='ineq',fun=lambda v:float(center_bound(v)[0]),jac=lambda v:center_bound(v)[1]),
                 dict(type='ineq',fun=lambda v:area-sum(v[N:]),jac=lambda v:np.r_[np.zeros(N),-np.ones(N)])]
    bounds=[(None,None)]*N+[(.05,cap)]*N
    records=[]
    stages=[(31,1e-3,700),(41,1e-4,1100),(61,1e-4,700)]
    if args.polish:
        records=json.loads((BASE/(name.lower()+'_bounded_search.json')).read_text())
        eligible=[r for r in records if r['success'] and r['grid']>=61 and r['small_metrics']['center']<1e-12]
        eligible.sort(key=lambda r:r['wide_metrics']['fraction_below_1e4'],reverse=True)
        starts=[np.r_[r['controls']['phases'],r['controls']['amplitudes']] for r in eligible[:args.starts]]
        stages=[(81,1e-4,1000)]
    for i,start in enumerate(starts[:args.starts]):
        v=start
        for n,tau,iterations in stages:
            fg=objective(N,theta,n)
            def fun(v):
                f,g=fg(jnp.asarray(v),tau);return float(f),np.array(g)
            t=time.perf_counter()
            fit=minimize(fun,v,jac=True,method='SLSQP',constraints=constraints,bounds=bounds,
                         options=dict(ftol=2e-11,maxiter=iterations))
            v=fit.x
            ss=old|dict(name=name,phases=((v[:N]+np.pi)%(2*np.pi)-np.pi).tolist(),amplitudes=v[N:].tolist(),
                        origin='studies/objective_selection/search_width.py; seed='+str(seed))
            m=metrics(ss,r=.5,n=401,peak=True);small=metrics(ss,r=.15,n=401,peak=True)
            records.append(dict(controls=ss,seed=int(seed),start_index=i,grid=n,tau=tau,half_width=.5,
                area_cap=area,peak_cap=cap,central_half_width=.20,central_grid=central_grid,
                stage='fine_constraint_polish' if args.polish else 'multistart',
                success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),
                seconds=time.perf_counter()-t,training_loss=float(fit.fun),wide_metrics=m,small_metrics=small))
            (BASE/(name.lower()+'_bounded_search.json')).write_text(json.dumps(records,indent=2))
            print(name,i,n,'ok',fit.success,'wide f4',m['fraction_below_1e4'],'small mean',small['mean'],'I0',m['center'],flush=True)
    print(name,'search complete',flush=True)
if __name__=='__main__':main()
