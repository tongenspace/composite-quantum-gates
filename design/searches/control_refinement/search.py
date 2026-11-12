"""Deterministic constrained refinement of the revised H15 and X5c controls.

The common Omega_0*T=pi convention and average gate infidelity are used.
Training averages equally weight native and common-peak error squares.
Initial bounded fits use a nominal penalty; SLSQP polishing enforces the
nominal gate and does not exceed the original area or peak amplitude.
"""
from pathlib import Path
import json,time,csv,sys,argparse
import numpy as np
import jax,jax.numpy as jnp
from scipy.optimize import minimize
from scipy.stats import qmc
jax.config.update('jax_enable_x64',True)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from rectangular import infidelity,metrics,matrix_infidelity

def compiled(N,theta,n=25,r=.15,peak_weight=.5):
    u=(np.arange(n)+.5)*2*r/n-r;D,E=np.meshgrid(u,u)
    e=jnp.asarray(E.ravel());d=jnp.asarray(D.ravel())
    c,s=np.cos(theta/2),np.sin(theta/2)
    def residual(x,E,D):
        phases=x[:N];amps=x[N:]
        z=jnp.zeros_like(E);o=jnp.ones_like(E)
        def step(carry,pa):
            w,x,y,z=carry;phi,a=pa;om=a*(1+E)
            q=jnp.sqrt(jnp.maximum(om*om+D*D,1e-24))
            co=jnp.cos(jnp.pi*q/2);k=jnp.pi/2*jnp.sinc(q/2)
            px=k*om*jnp.cos(phi);py=-k*om*jnp.sin(phi);pz=-k*D
            return (co*w-px*x-py*y-pz*z,co*x+w*px+py*z-pz*y,
                    co*y+w*py+pz*x-px*z,co*z+w*pz+px*y-py*x),None
        (w,x,y,z),_=jax.lax.scan(step,(o,z,z,z),(phases,amps))
        return jnp.stack([c*x-s*w,y,z],axis=-1)
    def nominal(x):return residual(x,jnp.asarray(0.),jnp.asarray(0.))
    def loss(x,weight,area):
        native=(2/3)*jnp.sum(residual(x,e,d)**2,axis=-1)
        peak=(2/3)*jnp.sum(residual(x,e,jnp.max(x[N:])*d)**2,axis=-1)
        center=(2/3)*jnp.sum(nominal(x)**2)
        over=jnp.maximum(jnp.sum(x[N:])-area,0)
        return 1e4*((1-peak_weight)*native.mean()+peak_weight*peak.mean()+weight*center+.01*over**2)
    return jax.jit(jax.value_and_grad(loss)),jax.jit(nominal),jax.jit(jax.jacfwd(nominal))

def evaluate(s,n=201):
    a=metrics(s,n=n);b=metrics(s,n=n,peak=True)
    rng=qmc.Sobol(2,scramble=True,seed=90173).random_base2(14)
    E,D=((2*rng-1)*.15).T
    z=infidelity(s,E,D);zp=infidelity(s,E,max(s['amplitudes'])*D)
    wide=metrics(s,r=.3,n=201)
    return dict(nominal=a['center'],native_mean=a['mean'],peak_mean=b['mean'],
                area_over_target=a['area_over_target'],duration_over_target=a['duration_over_target'],
                native_fraction=a['fraction_below_1e4'],peak_fraction=b['fraction_below_1e4'],
                native_sobol_mean=float(z.mean()),peak_sobol_mean=float(zp.mean()),
                native_sobol_max=float(z.max()),peak_sobol_max=float(zp.max()),wide_mean=wide['mean'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gate',choices=['X5c','H15','both'],default='H15')
    args=parser.parse_args()
    reg=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
    if not (HERE/'input_controls.json').exists():
        (HERE/'input_controls.json').write_text(json.dumps({k:reg[k] for k in ['X5c','H15']},indent=2))
    originals=json.loads((HERE/'input_controls.json').read_text())
    records=[]
    for ci,name in enumerate(['X5c','H15']):
        if args.gate!='both' and name!=args.gate:continue
        old=originals[name];N=old['N'];area=sum(old['amplitudes']);cap=max(old['amplitudes'])
        base=np.r_[old['phases'],old['amplitudes']];rng=np.random.default_rng(20261014+ci)
        starts=[base]
        if name=='X5c':
            for family in ['X5a','X5b']:
                ss=reg[family];starts.append(np.r_[ss['phases'],ss['amplitudes']])
        for scale in [.03,.1,.3,.6]:
            for _ in range(3):
                starts.append(np.r_[base[:N]+rng.normal(0,scale,N),np.clip(base[N:]+rng.normal(0,scale/3,N),.02,cap)])
        fg,nom,jac=compiled(N,old['theta'])
        bounds=[(None,None)]*N+[(1e-5,cap)]*N
        def seq(x):return {**old,'phases':((x[:N]+np.pi)%(2*np.pi)-np.pi).tolist(),'amplitudes':x[N:].tolist(),
                              'origin':'studies/control_refinement/search.py; seed='+str(20261014+ci)}
        fitted=[]
        for k,start in enumerate(starts):
            weight=1. if k%2==0 else 10.
            def f(x):
                v,g=fg(jnp.asarray(x),weight,area);return float(v),np.asarray(g)
            t=time.perf_counter();fit=minimize(f,start,jac=True,method='L-BFGS-B',bounds=bounds,
                options=dict(maxiter=1800,ftol=1e-13,gtol=1e-8,maxls=40,maxcor=20))
            ss=seq(fit.x);m=evaluate(ss)
            fitted.append((fit.x,m))
            row=dict(name=name,stage='penalty',start_index=k,center_weight=weight,controls=ss,metrics=m,
                     success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),seconds=time.perf_counter()-t)
            records.append(row);(HERE/'search_records.json').write_text(json.dumps(records,indent=2))
            print(name,k,'mean',m['native_mean'],m['peak_mean'],'center',m['nominal'],'area',sum(ss['amplitudes']),flush=True)
        # Exact-gate and area-constrained polishing, on a finer training mesh.
        fg,nom,jac=compiled(N,old['theta'],n=41)
        selected=sorted(fitted,key=lambda v:v[1]['native_mean']+v[1]['peak_mean'])[:5]
        for k,(start,_) in enumerate(selected):
            def f(x):
                v,g=fg(jnp.asarray(x),0.,area);return float(v),np.asarray(g)
            constraints=[dict(type='eq',fun=lambda x:np.asarray(nom(jnp.asarray(x))),
                              jac=lambda x:np.asarray(jac(jnp.asarray(x)))),
                         dict(type='ineq',fun=lambda x:area-sum(x[N:]),jac=lambda x:np.r_[np.zeros(N),-np.ones(N)])]
            t=time.perf_counter();fit=minimize(f,start,jac=True,method='SLSQP',bounds=bounds,constraints=constraints,
                options=dict(maxiter=1600,ftol=1e-12))
            ss=seq(fit.x);m=evaluate(ss,n=401)
            row=dict(name=name,stage='exact_polish',start_index=k,controls=ss,metrics=m,success=bool(fit.success),
                     message=str(fit.message),iterations=int(fit.nit),seconds=time.perf_counter()-t)
            records.append(row);(HERE/'search_records.json').write_text(json.dumps(records,indent=2))
            print(name,'polish',k,'mean',m['native_mean'],m['peak_mean'],'center',m['nominal'],'area',sum(ss['amplitudes']),flush=True)
        base_metrics=evaluate(old,n=401)
        eligible=[r for r in records if r['name']==name and r['metrics']['nominal']<1e-10 and
                  r['metrics']['native_mean']<base_metrics['native_mean'] and r['metrics']['peak_mean']<base_metrics['peak_mean'] and
                  sum(r['controls']['amplitudes'])<=area+1e-8 and max(r['controls']['amplitudes'])<=cap+1e-8]
        if not eligible:raise RuntimeError('No dominating candidate found for '+name)
        winner=min(eligible,key=lambda r:r['metrics']['native_mean']+r['metrics']['peak_mean'])
        selected_file=HERE/'selected_controls.json';out=json.loads(selected_file.read_text()) if selected_file.exists() else {}
        out[name]=dict(controls=winner['controls'],metrics=evaluate(winner['controls'],n=801),baseline_metrics=evaluate(old,n=801))
        # Random matrix-exponential checks are independent of the fitting kernel.
        err=max(abs(float(infidelity(winner['controls'],e,d))-matrix_infidelity(winner['controls'],e,d))
                for e,d in rng.uniform(-.5,.5,(24,2)))
        assert err<1e-13;out[name]['matrix_max_difference']=err
        selected_file.write_text(json.dumps(out,indent=2));print('SELECTED',name,json.dumps(out[name]['metrics']),flush=True)
if __name__=='__main__':main()
