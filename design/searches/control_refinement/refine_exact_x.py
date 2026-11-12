"""Search nominal-target-matched X controls under fixed reference constraints.

No canonical controls are changed. Full-unitary matching, cumulative-area and
peak-amplitude caps are enforced with analytic automatic-differentiation
Jacobians. Means refer to the native/common-peak half-width-0.15 squares;
coverage refers to the plotted native/common-peak half-width-0.5 squares.
"""
from pathlib import Path
import argparse, json, time
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from search import compiled, evaluate
from rectangular import metrics, infidelity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sampled_worst(seq,n=801,peak=False):
    u=np.linspace(-.15,.15,n);D,E=np.meshgrid(u,u)
    scale=max(seq['amplitudes']) if peak else 1.
    return float(infidelity(seq,E,scale*D).max())


def coverage_kernel(N, theta, n=121, peak=False, r=.5, mode='coverage'):
    u = np.linspace(-r,r,n) if mode=='maximum' else (np.arange(n)+.5)*2*r/n-r
    D, E = np.meshgrid(u, u)
    e, d = jnp.asarray(E.ravel()), jnp.asarray(D.ravel())
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    def values(v):
        dd = d * jnp.max(v[N:]) if peak else d
        z = jnp.zeros_like(e)
        def step(carry, pa):
            w, x, y, z = carry
            phi, a = pa
            om = a * (1 + e)
            q = jnp.sqrt(jnp.maximum(om * om + dd * dd, 1e-24))
            co = jnp.cos(np.pi * q / 2)
            k = np.pi / 2 * jnp.sinc(q / 2)
            px, py, pz = k * om * jnp.cos(phi), -k * om * jnp.sin(phi), -k * dd
            return (co*w-px*x-py*y-pz*z, co*x+w*px+py*z-pz*y,
                    co*y+w*py+pz*x-px*z, co*z+w*pz+px*y-py*x), None
        (w, x, y, z), _ = jax.lax.scan(step, (jnp.ones_like(e), z, z, z),
                                     (v[:N], v[N:]))
        return (2/3) * ((c*x-s*w)**2+y*y+z*z)
    if mode=='maximum':return jax.jit(jax.value_and_grad(lambda v:1e4*jnp.max(values(v))))
    return jax.jit(jax.value_and_grad(lambda v, temp:
        1-jnp.mean(jax.nn.sigmoid((1e-4-values(v))/temp))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', choices=['X5c','X7c','X11c'], required=True)
    ap.add_argument('--starts', type=int, default=7)
    ap.add_argument('--stage', choices=['mean','coverage'], default='mean')
    ap.add_argument('--iterations', type=int, default=1100)
    ap.add_argument('--global-starts', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--mean-margin', type=float, default=.998)
    ap.add_argument('--worst-caps', action='store_true')
    ap.add_argument('--seed-file', default='')
    ap.add_argument('--input-records', default='')
    args = ap.parse_args()
    baseline_file = HERE/'exact_x_baselines.json'
    if not baseline_file.exists():
        reg = json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
        baseline_file.write_text(json.dumps({k:reg[k] for k in ['X5c','X7c','X11c']}, indent=2)+'\n')
    name = args.gate
    old = json.loads(baseline_file.read_text())[name]
    N = old['N']; area = sum(old['amplitudes']); cap = max(old['amplitudes'])
    base = evaluate(old, 801)
    wb = {k:metrics(old,r=.5,n=801,peak=p) for k,p in [('native',False),('peak',True)]}
    seed = {'X5c':202610170,'X7c':202610171,'X11c':202610172}[name]
    rng = np.random.default_rng(seed)
    x0 = np.r_[old['phases'],old['amplitudes']]
    fn, nom, jac = compiled(N,old['theta'],n=51,peak_weight=0.)
    fp, _, _ = compiled(N,old['theta'],n=51,peak_weight=1.)
    bounds = [(None,None)]*N+[(1e-5,cap)]*N
    constraints = [dict(type='eq',fun=lambda x:np.asarray(nom(jnp.asarray(x))),
                        jac=lambda x:np.asarray(jac(jnp.asarray(x)))),
                   dict(type='ineq',fun=lambda x:area-np.sum(x[N:]),
                        jac=lambda x:np.r_[np.zeros(N),-np.ones(N)])]
    suffix = ('_'+args.tag) if args.tag else ''
    out_file = HERE/(name.lower()+'_exact_'+args.stage+suffix+'_records.json')
    rows = []
    def value(fun,x,weight=0.):
        v,g = fun(jnp.asarray(x),weight,area)
        return float(v),np.asarray(g)
    def add_worst(cons):
        budgets={}
        if args.worst_caps:
            for key,peak in [('native',False),('peak',True)]:
                mf=coverage_kernel(N,old['theta'],n=51,peak=peak,r=.15,mode='maximum')
                budget=.995*sampled_worst(old,n=801,peak=peak);budgets[key]=budget
                cons.append(dict(type='ineq',fun=lambda x,f=mf,b=budget:b*1e4-f(jnp.asarray(x))[0],
                            jac=lambda x,f=mf:-np.asarray(f(jnp.asarray(x))[1])))
        return budgets
    def control(x):
        return {**old,'phases':((x[:N]+np.pi)%(2*np.pi)-np.pi).tolist(),
                'amplitudes':x[N:].tolist(),
                'origin':f'studies/control_refinement/refine_exact_x.py; seed={seed}'}
    def record(fit,stage,i,details):
        s = control(fit.x); m = evaluate(s,401)
        w = {k:metrics(s,r=.5,n=401,peak=p) for k,p in [('native',False),('peak',True)]}
        row = dict(name=name,stage=stage,start_index=i,controls=s,metrics=m,
                   wide_domain_metrics=w,success=bool(fit.success),message=str(fit.message),
                   iterations=int(fit.nit),protocol=details)
        rows.append(row);out_file.write_text(json.dumps(rows,indent=2)+'\n')
        print(name,stage,i,'means',m['native_mean'],m['peak_mean'],'I0',m['nominal'],
              'wide',w['native']['fraction_below_1e4'],w['peak']['fraction_below_1e4'],
              'area',sum(s['amplitudes']),'ok',bool(fit.success),flush=True)
    if args.stage == 'mean':
        starts = [x0]
        # Smaller X5c exact-nominal families from the bounded seed search.
        if name == 'X5c':
            reg = json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
            for key in ['X5a','X5b']:
                s=reg[key];starts.append(np.r_[s['phases'],s['amplitudes']])
        for scale in [.03,.1,.3,.6,.9]:
            starts.append(np.r_[x0[:N]+rng.normal(0,scale,N),
                                np.clip(x0[N:]+rng.normal(0,scale/4,N),.02,cap)])
        if args.global_starts:
            starts=[]
            for i in range(args.global_starts):
                a=rng.uniform(.2,cap,N)
                a*=min(1.,area/a.sum())
                starts.append(np.r_[rng.uniform(-np.pi,np.pi,N),a])
        if args.seed_file:
            starts=[np.asarray(v) for v in json.loads((HERE/args.seed_file).read_text())[name]]
        worst_budgets=add_worst(constraints)
        for i,start in enumerate(starts[:args.global_starts or args.starts]):
            weight = [0.,.5,1.][i%3]
            if args.global_starts:
                fw,_,_=compiled(N,old['theta'],n=25,peak_weight=weight)
                warm=minimize(lambda x:value(fw,x,10.),start,jac=True,method='L-BFGS-B',
                              bounds=bounds,options=dict(maxiter=1000,ftol=1e-13,gtol=1e-8,maxls=40))
                record(warm,'warm',i,dict(seed=seed,grid=25,peak_weight=weight,nominal_penalty=10.))
                start=warm.x
            fg,_,_=compiled(N,old['theta'],n=51,peak_weight=weight)
            fit=minimize(lambda x:value(fg,x),start,jac=True,method='SLSQP',
                         bounds=bounds,constraints=constraints,
                         options=dict(maxiter=args.iterations,ftol=1e-12))
            record(fit,'mean',i,dict(seed=seed,grid=51,halfwidth=.15,peak_weight=weight,
                                    nominal='exact full unitary',area_cap=area,peak_cap=cap,
                                    seed_file=args.seed_file,worst_budgets=worst_budgets))
    else:
        candidates=json.loads((HERE/(args.input_records or name.lower()+'_exact_mean_records.json')).read_text())
        feasible=[r for r in candidates if r['success'] and r['metrics']['nominal']<1e-12
                  and sum(r['controls']['amplitudes'])<=area+1e-9]
        feasible.sort(key=lambda r:r['metrics']['native_mean']/base['native_mean']+
                                  r['metrics']['peak_mean']/base['peak_mean'])
        starts=[x0]+[np.r_[r['controls']['phases'],r['controls']['amplitudes']] for r in feasible]
        unique=[]
        for v in starts:
            if all(np.linalg.norm(v-u)>.02 for u in unique):unique.append(v)
        # A safety margin avoids calling a training-mesh tie a fine-grid gain.
        cons=constraints+[
            dict(type='ineq',fun=lambda x:args.mean_margin*base['native_mean']*1e4-value(fn,x)[0],
                 jac=lambda x:-value(fn,x)[1]),
            dict(type='ineq',fun=lambda x:args.mean_margin*base['peak_mean']*1e4-value(fp,x)[0],
                 jac=lambda x:-value(fp,x)[1])]
        worst_budgets=add_worst(cons)
        cn=coverage_kernel(N,old['theta']);cp=coverage_kernel(N,old['theta'],peak=True)
        for i,start in enumerate(unique[:args.starts]):
            x=start
            for temp in [4e-5,1.5e-5,4e-6]:
                def fun(v):
                    a,g=cn(jnp.asarray(v),temp);b,h=cp(jnp.asarray(v),temp)
                    return float(.5*(a+b)),np.asarray(.5*(g+h))
                fit=minimize(fun,x,jac=True,method='SLSQP',bounds=bounds,constraints=cons,
                             options=dict(maxiter=args.iterations,ftol=1e-10));x=fit.x
                record(fit,'coverage',i,dict(seed=seed,grid=121,halfwidth=.5,temperature=temp,
                       mean_grid=51,mean_budget_ratio=args.mean_margin,worst_budgets=worst_budgets,
                       area_cap=area,peak_cap=cap,
                       nominal='exact full unitary',baseline_wide=wb))


if __name__ == '__main__':
    main()
