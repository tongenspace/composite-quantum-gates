"""Refine the X5c Pareto trade-off while holding its resource budgets fixed."""
from pathlib import Path
import json,time
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
from search import compiled,evaluate,infidelity,matrix_infidelity
HERE=Path(__file__).resolve().parent

def main():
    old=json.loads((HERE/'input_controls.json').read_text())['X5c'];base=evaluate(old,n=401)
    area=sum(old['amplitudes']);cap=max(old['amplitudes']);x0=np.r_[old['phases'],old['amplitudes']]
    fn,nom,jac=compiled(5,np.pi,n=41,peak_weight=0.)
    fp,_,_=compiled(5,np.pi,n=41,peak_weight=1.)
    fm,_,_=compiled(5,np.pi,n=41,peak_weight=.5)
    bounds=[(None,None)]*5+[(1e-5,cap)]*5
    def seq(x):return {**old,'phases':((x[:5]+np.pi)%(2*np.pi)-np.pi).tolist(),'amplitudes':x[5:].tolist(),
                          'origin':'studies/control_refinement/polish_x5c.py'}
    def metric_fun(fun,x):
        v,g=fun(jnp.asarray(x),0.,area);return float(v),np.asarray(g)
    def center(x):
        q=np.asarray(nom(jnp.asarray(x)));J=np.asarray(jac(jnp.asarray(x)))
        return (2/3)*np.dot(q,q),(4/3)*q@J
    def penalized(x):
        v,g=fm(jnp.asarray(x),1.,area);return float(v),np.asarray(g)
    warm=minimize(penalized,x0,jac=True,method='L-BFGS-B',bounds=bounds,
                  options=dict(maxiter=2000,ftol=1e-14,gtol=1e-9,maxls=50))
    starts=[warm.x,x0];records=[]
    for nominal_ratio,native_ratio in [(.95,.95),(.5,1.),(1.,.9),(.5,.95),(.1,1.)]:
        for k,start in enumerate(starts):
            # Constraints are scaled for SLSQP's absolute feasibility tolerance.
            constraints=[dict(type='ineq',fun=lambda x:area-sum(x[5:]),jac=lambda x:np.r_[np.zeros(5),-np.ones(5)]),
                dict(type='ineq',fun=lambda x:1e4*(nominal_ratio*base['nominal']-center(x)[0]),jac=lambda x:-1e4*center(x)[1]),
                dict(type='ineq',fun=lambda x:native_ratio*base['native_mean']*1e4-metric_fun(fn,x)[0],jac=lambda x:-metric_fun(fn,x)[1])]
            t=time.perf_counter();fit=minimize(lambda x:metric_fun(fp,x),start,jac=True,method='SLSQP',
                bounds=bounds,constraints=constraints,options=dict(maxiter=1600,ftol=1e-11))
            s=seq(fit.x);m=evaluate(s,n=401)
            row=dict(name='X5c',controls=s,metrics=m,nominal_ratio=nominal_ratio,native_ratio=native_ratio,start_index=k,
                     success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),seconds=time.perf_counter()-t)
            records.append(row);(HERE/'x5c_pareto_records.json').write_text(json.dumps(records,indent=2))
            print('PARETO',nominal_ratio,native_ratio,k,m,flush=True)
    eligible=[r for r in records if r['metrics']['nominal']<base['nominal'] and r['metrics']['native_mean']<base['native_mean']
              and r['metrics']['peak_mean']<base['peak_mean'] and sum(r['controls']['amplitudes'])<=area+1e-9]
    if not eligible:raise RuntimeError('No resource-matched X5c improvement found.')
    winner=min(eligible,key=lambda r:r['metrics']['native_mean']+r['metrics']['peak_mean']);s=winner['controls']
    result=dict(controls=s,metrics=evaluate(s,n=801),baseline_metrics=evaluate(old,n=801),
                constraints=dict(peak_cap=cap,area_over_pi_cap=area,nominal_ratio=winner['nominal_ratio'],native_ratio=winner['native_ratio']))
    rng=np.random.default_rng(73014);error=max(abs(float(infidelity(s,e,d))-matrix_infidelity(s,e,d))
                for e,d in rng.uniform(-.5,.5,(32,2)))
    assert error<1e-13;result['matrix_max_difference']=error
    p=HERE/'selected_controls.json';out=json.loads(p.read_text());out['X5c']=result;p.write_text(json.dumps(out,indent=2))
    print('SELECTED X5c',json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
