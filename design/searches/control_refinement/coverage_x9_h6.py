"""Refine the plotted 1e-4 region while respecting original resource bounds.

This optional follow-up uses a sigmoid coverage surrogate on [-0.5,0.5]^2,
exact nominal matching, and mean-error budgets on the native/common-peak
half-width-0.15 squares. Saved mean-oriented fits provide diverse starts.
"""
from pathlib import Path
import json,argparse
import numpy as np
import jax,jax.numpy as jnp
from scipy.optimize import minimize
from search import compiled,evaluate
from rectangular import metrics
HERE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gate',choices=['X9c','H6'],required=True)
    parser.add_argument('--starts',type=int,default=5);parser.add_argument('--split',action='store_true');args=parser.parse_args();name=args.gate
    old=json.loads((HERE/'input_controls.json').read_text())[name];baseline=evaluate(old,n=401)
    N=old['N'];area=sum(old['amplitudes']);cap=max(old['amplitudes'])
    input_name=name.lower()+('_split_seed_records.json' if args.split else '_refinement_records.json')
    records=json.loads((HERE/input_name).read_text())
    feasible=[r for r in records if r['success'] and r['metrics']['nominal']<1e-12 and sum(r['controls']['amplitudes'])<=area+1e-8]
    ranked=sorted(feasible,key=lambda r:-r['wide_domain_metrics']['fraction_below_1e4'])
    starts=[] if args.split else [np.r_[old['phases'],old['amplitudes']]]
    for r in ranked:
        v=np.r_[r['controls']['phases'],r['controls']['amplitudes']]
        if all(np.linalg.norm(v-x)>.01 for x in starts):starts.append(v)
        if len(starts)>=args.starts:break
    fn,nom,jac=compiled(N,old['theta'],n=61,peak_weight=0.)
    fp,_,_=compiled(N,old['theta'],n=61,peak_weight=1.)
    u=(np.arange(101)+.5)/101-.5;D,E=np.meshgrid(u,u);e=jnp.asarray(E.ravel());d=jnp.asarray(D.ravel())
    c,s=np.cos(old['theta']/2),np.sin(old['theta']/2)
    def values(v):
        z=jnp.zeros_like(e);o=jnp.ones_like(e)
        def step(carry,pa):
            w,x,y,z=carry;phi,a=pa;om=a*(1+e);q=jnp.sqrt(jnp.maximum(om*om+d*d,1e-24))
            co=jnp.cos(np.pi*q/2);k=np.pi/2*jnp.sinc(q/2)
            px=k*om*jnp.cos(phi);py=-k*om*jnp.sin(phi);pz=-k*d
            return (co*w-px*x-py*y-pz*z,co*x+w*px+py*z-pz*y,
                    co*y+w*py+pz*x-px*z,co*z+w*pz+px*y-py*x),None
        (w,x,y,z),_=jax.lax.scan(step,(o,z,z,z),(v[:N],v[N:]))
        return (2/3)*((c*x-s*w)**2+y*y+z*z)
    fg=jax.jit(jax.value_and_grad(lambda x,t:1-jnp.mean(jax.nn.sigmoid((1e-4-values(x))/t))))
    def mean(fun,x):v,g=fun(jnp.asarray(x),0.,area);return float(v),np.asarray(g)
    bounds=[(None,None)]*N+[(1e-5,cap)]*N
    cons=[dict(type='eq',fun=lambda x:np.asarray(nom(jnp.asarray(x))),jac=lambda x:np.asarray(jac(jnp.asarray(x)))),
        dict(type='ineq',fun=lambda x:area-sum(x[N:]),jac=lambda x:np.r_[np.zeros(N),-np.ones(N)]),
        dict(type='ineq',fun=lambda x:baseline['native_mean']*1.1e4-mean(fn,x)[0],jac=lambda x:-mean(fn,x)[1]),
        dict(type='ineq',fun=lambda x:baseline['peak_mean']*1.05e4-mean(fp,x)[0],jac=lambda x:-mean(fp,x)[1])]
    out=[]
    for i,start in enumerate(starts):
        x=start
        for temp in [4e-5,1.5e-5,4e-6]:
            def f(v):a,b=fg(jnp.asarray(v),temp);return float(a),np.asarray(b)
            fit=minimize(f,x,jac=True,method='SLSQP',bounds=bounds,constraints=cons,
                options=dict(maxiter=1400,ftol=1e-10));x=fit.x
            control={**old,'phases':((x[:N]+np.pi)%(2*np.pi)-np.pi).tolist(),'amplitudes':x[N:].tolist(),
                'origin':f'studies/control_refinement/coverage_x9_h6.py; deterministic starts from '+input_name}
            m=evaluate(control,n=401);wide=metrics(control,r=.5,n=401)
            row=dict(name=name,stage='coverage_polish',start_index=i,temperature=temp,controls=control,metrics=m,
                wide_domain_metrics=wide,success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),
                protocol=dict(grid=101,halfwidth=.5,nominal='exact unitary',native_mean_budget=1.1,peak_mean_budget=1.05))
            out.append(row);(HERE/(name.lower()+('_split_coverage_records.json' if args.split else '_coverage_records.json'))).write_text(json.dumps(out,indent=2)+'\n')
            print(name,'coverage',i,temp,'wide',wide['fraction_below_1e4'],'means',m['native_mean'],m['peak_mean'],'I0',m['nominal'],flush=True)
if __name__=='__main__':main()
