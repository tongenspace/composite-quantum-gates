"""Search for increased 1e-4 coverage without silently changing error domains."""
from pathlib import Path
import json,time
import argparse
import numpy as np
import jax,jax.numpy as jnp
from scipy.optimize import minimize
from search import compiled,evaluate
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--balanced',action='store_true');args=parser.parse_args()
    nominal_ratio=.5 if args.balanced else .9
    native_ratio=1.1 if args.balanced else 1.
    peak_ratio=1.2 if args.balanced else 1.
    tag='balanced' if args.balanced else 'strict'
    old=json.loads((HERE/'input_controls.json').read_text())['X5c'];baseline=evaluate(old,n=401)
    area=sum(old['amplitudes']);cap=max(old['amplitudes']);base=np.r_[old['phases'],old['amplitudes']]
    fn,nom,jac=compiled(5,np.pi,n=61,peak_weight=0.)
    fp,_,_=compiled(5,np.pi,n=61,peak_weight=1.)
    u=(np.arange(61)+.5)*.3/61-.15;D,E=np.meshgrid(u,u);e=jnp.asarray(E.ravel());d=jnp.asarray(D.ravel())
    def values(x):
        w=jnp.ones_like(e);xx=jnp.zeros_like(e);y=jnp.zeros_like(e);z=jnp.zeros_like(e)
        for phi,a in zip(x[:5],x[5:]):
            om=a*(1+e);q=jnp.sqrt(jnp.maximum(om*om+d*d,1e-24));co=jnp.cos(np.pi*q/2);k=np.pi/2*jnp.sinc(q/2)
            px=k*om*jnp.cos(phi);py=-k*om*jnp.sin(phi);pz=-k*d
            w,xx,y,z=co*w-px*xx-py*y-pz*z,co*xx+w*px+py*z-pz*y,co*y+w*py+pz*xx-px*z,co*z+w*pz+px*y-py*xx
        return (2/3)*(w*w+y*y+z*z)
    cg=jax.jit(jax.value_and_grad(lambda x,temp:1-jnp.mean(jax.nn.sigmoid((1e-4-values(x))/temp))))
    def mean(fun,x):
        v,g=fun(jnp.asarray(x),0.,area);return float(v),np.asarray(g)
    def center(x):
        q=np.asarray(nom(jnp.asarray(x)));J=np.asarray(jac(jnp.asarray(x)));return (2/3)*q@q,(4/3)*q@J
    rng=np.random.default_rng(202610143);starts=[base]
    selected=json.loads((HERE/'selected_controls.json').read_text()).get('X5c')
    if selected:starts.append(np.r_[selected['controls']['phases'],selected['controls']['amplitudes']])
    starts += [np.r_[base[:5]+rng.normal(0,.15,5),np.clip(base[5:]+rng.normal(0,.03,5),.02,cap)] for _ in range(3)]
    records=[]
    for i,start in enumerate(starts):
        x=start
        for temp in [4e-5,1.5e-5,4e-6]:
            def f(x):v,g=cg(jnp.asarray(x),temp);return float(v),np.asarray(g)
            constraints=[dict(type='ineq',fun=lambda x:area-sum(x[5:]),jac=lambda x:np.r_[np.zeros(5),-np.ones(5)]),
                dict(type='ineq',fun=lambda x:1e4*(nominal_ratio*baseline['nominal']-center(x)[0]),jac=lambda x:-1e4*center(x)[1]),
                dict(type='ineq',fun=lambda x:native_ratio*baseline['native_mean']*1e4-mean(fn,x)[0],jac=lambda x:-mean(fn,x)[1]),
                dict(type='ineq',fun=lambda x:peak_ratio*baseline['peak_mean']*1e4-mean(fp,x)[0],jac=lambda x:-mean(fp,x)[1])]
            fit=minimize(f,x,jac=True,method='SLSQP',bounds=[(None,None)]*5+[(1e-5,cap)]*5,
                         constraints=constraints,options=dict(maxiter=1800,ftol=1e-10));x=fit.x
            s={**old,'phases':((x[:5]+np.pi)%(2*np.pi)-np.pi).tolist(),'amplitudes':x[5:].tolist(),
               'origin':'studies/control_refinement/search_coverage.py; seed=202610143'};m=evaluate(s,n=401)
            records.append(dict(name='X5c',start_index=i,temperature=temp,controls=s,metrics=m,
                success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit)))
            (HERE/('x5c_coverage_'+tag+'_records.json')).write_text(json.dumps(records,indent=2))
            print('COVERAGE',i,temp,json.dumps(m),flush=True)
    good=[r for r in records if r['metrics']['native_fraction']>baseline['native_fraction'] and
          r['metrics']['nominal']<baseline['nominal'] and r['metrics']['native_mean']<=1.003*native_ratio*baseline['native_mean'] and
          r['metrics']['peak_mean']<=1.003*peak_ratio*baseline['peak_mean'] and sum(r['controls']['amplitudes'])<=area+1e-8]
    report=dict(baseline=baseline,eligible=good,protocol=dict(nominal_ratio=nominal_ratio,native_ratio=native_ratio,peak_ratio=peak_ratio))
    (HERE/('coverage_'+tag+'_candidates.json')).write_text(json.dumps(report,indent=2));print('COVERAGE CANDIDATES',len(good),flush=True)
if __name__=='__main__':main()
