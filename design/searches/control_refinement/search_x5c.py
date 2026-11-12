"""X5c search across explicitly disclosed area/peak budgets."""
from pathlib import Path
import json,time
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
from search import compiled,evaluate,infidelity,matrix_infidelity
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def main():
    old=json.loads((HERE/'input_controls.json').read_text())['X5c']
    reg=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
    base=np.r_[old['phases'],old['amplitudes']];rng=np.random.default_rng(202610142)
    baseline=evaluate(old,n=401);records=[];carry=[base]
    for cap,area in [(2.,7.),(2.,9.),(3.,12.)]:
        starts=carry[:3]+[np.r_[reg[k]['phases'],reg[k]['amplitudes']] for k in ['X5a','X5b','G5']]
        for k in range(12):
            if k<6:
                starts.append(np.r_[base[:5]+rng.normal(0,.6 if k<3 else 1.5,5),
                                    np.clip(base[5:]+rng.normal(0,.3,5),.05,cap)])
            else:starts.append(np.r_[rng.uniform(-np.pi,np.pi,5),rng.uniform(.5,cap,5)])
        for k,start in enumerate(starts):
            peak_weight=[0.,.1,.3][k%3];center_weight=[.05,.5,5.][(k//3)%3]
            fg,nom,jac=compiled(5,np.pi,n=25,peak_weight=peak_weight)
            def f(x):
                v,g=fg(jnp.asarray(x),center_weight,area);return float(v),np.asarray(g)
            t=time.perf_counter();fit=minimize(f,start,jac=True,method='L-BFGS-B',bounds=[(None,None)]*5+[(1e-5,cap)]*5,
                    options=dict(maxiter=1300,ftol=1e-13,gtol=1e-8,maxls=40,maxcor=20))
            s={**old,'phases':((fit.x[:5]+np.pi)%(2*np.pi)-np.pi).tolist(),'amplitudes':fit.x[5:].tolist(),
               'origin':'studies/control_refinement/search_x5c.py; seed=202610142'}
            m=evaluate(s);row=dict(name='X5c',controls=s,metrics=m,cap=cap,area_cap=area,peak_weight=peak_weight,
                center_weight=center_weight,start_index=k,success=bool(fit.success),iterations=int(fit.nit),seconds=time.perf_counter()-t)
            records.append(row);(HERE/'x5c_search_records.json').write_text(json.dumps(records,indent=2))
            print('X5c',cap,area,k,'mean',m['native_mean'],m['peak_mean'],'center',m['nominal'],'area',sum(s['amplitudes']),flush=True)
        group=sorted([r for r in records if r['cap']==cap and r['area_cap']==area],key=lambda r:r['metrics']['native_mean']+r['metrics']['peak_mean'])
        carry=[np.r_[r['controls']['phases'],r['controls']['amplitudes']] for r in group[:3]]
        # Exact nominal polishing of the strongest native-domain candidates.
        group=sorted(group,key=lambda r:r['metrics']['native_mean'])[:4]
        for k,row in enumerate(group):
            fg,nom,jac=compiled(5,np.pi,n=41,peak_weight=.05)
            def f(x):
                v,g=fg(jnp.asarray(x),0.,area);return float(v),np.asarray(g)
            constraints=[dict(type='eq',fun=lambda x:np.asarray(nom(jnp.asarray(x))),jac=lambda x:np.asarray(jac(jnp.asarray(x)))),
                         dict(type='ineq',fun=lambda x:area-sum(x[5:]),jac=lambda x:np.r_[np.zeros(5),-np.ones(5)])]
            fit=minimize(f,np.r_[row['controls']['phases'],row['controls']['amplitudes']],jac=True,method='SLSQP',
                bounds=[(None,None)]*5+[(1e-5,cap)]*5,constraints=constraints,options=dict(maxiter=1300,ftol=1e-12))
            s={**row['controls'],'phases':((fit.x[:5]+np.pi)%(2*np.pi)-np.pi).tolist(),'amplitudes':fit.x[5:].tolist()};m=evaluate(s,n=401)
            records.append(dict(name='X5c',stage='exact_polish',controls=s,metrics=m,cap=cap,area_cap=area,
                                success=bool(fit.success),iterations=int(fit.nit)))
            (HERE/'x5c_search_records.json').write_text(json.dumps(records,indent=2))
            print('X5c POLISH',cap,area,k,'mean',m['native_mean'],m['peak_mean'],'center',m['nominal'],flush=True)
        eligible=[r for r in records if r['metrics']['nominal']<=baseline['nominal'] and
                    r['metrics']['native_mean']<baseline['native_mean']*.95 and r['metrics']['peak_mean']<baseline['peak_mean']]
        if eligible:
            # Prefer a gate-exact solution if it also improves both domain means.
            exact=[r for r in eligible if r['metrics']['nominal']<1e-12]
            winner=min(exact or eligible,key=lambda r:r['metrics']['native_mean']+r['metrics']['peak_mean'])
            s=winner['controls'];result=dict(controls=s,metrics=evaluate(s,n=801),baseline_metrics=evaluate(old,n=801),
                                           constraints=dict(peak_cap=winner['cap'],area_over_pi_cap=winner['area_cap']))
            error=max(abs(float(infidelity(s,e,d))-matrix_infidelity(s,e,d)) for e,d in rng.uniform(-.5,.5,(24,2)))
            assert error<1e-13;result['matrix_max_difference']=error
            p=HERE/'selected_controls.json';out=json.loads(p.read_text()) if p.exists() else {};out['X5c']=result
            p.write_text(json.dumps(out,indent=2));print('SELECTED X5c',json.dumps(result['metrics']),flush=True)
            break
    else:raise RuntimeError('No X5c candidate meeting the disclosed improvement criteria.')
if __name__=='__main__':main()
