"""Refine H3/H4/H5/H7/H8/H10 with nominal matching and original resource caps.

Mean-oriented multistart fits discover basins; threshold-area polishing then
uses the plotted square. Complete immutable seed inputs and all fit records
are saved. The registry is changed only by a separate validated selection.
"""
from pathlib import Path
import argparse,json
import numpy as np
import jax,jax.numpy as jnp
from scipy.optimize import minimize
from search import compiled,evaluate
from rectangular import metrics
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NAMES=['H3','H4','H5','H7','H8','H10']

def prepare():
    reg=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
    p=HERE/'input_controls.json';old=json.loads(p.read_text())
    for n in NAMES:old.setdefault(n,reg[n])
    p.write_text(json.dumps(old,indent=2)+'\n')
    path=HERE/'remaining_h_seed_inputs.json'
    if path.exists():return
    inputs={}
    for ci,name in enumerate(NAMES):
        s=old[name];N=s['N'];cap=max(s['amplitudes']);area=sum(s['amplitudes']);base=np.r_[s['phases'],s['amplitudes']]
        seed=202610150+ci;rng=np.random.default_rng(seed);starts=[dict(source='original',vector=base.tolist())]
        for scale in [.05,.2,.5,.9]:
            for _ in range(2):starts.append(dict(source=f'perturbation {scale}',vector=np.r_[base[:N]+rng.normal(0,scale,N),np.clip(base[N:]+rng.normal(0,scale/3,N),.02,cap)].tolist()))
        for source in ['Single_H','H3','H4','H5','H6','H7','H8']:
            c=old.get(source,reg[source])
            if c['N']>=N:continue
            ph=list(c['phases']);a=list(c['amplitudes'])
            while len(a)<N:
                k=int(np.argmax(a));v=a[k]/2;a[k:k+1]=[v,v];ph[k:k+1]=[ph[k],ph[k]]
            scale=min(1.,cap/max(a),area/sum(a));a=np.asarray(a)*scale
            starts.append(dict(source='split '+source,vector=np.r_[ph,a].tolist()))
        for _ in range(3):
            a=rng.uniform(.15,cap,N);a*=min(1.,.97*area/sum(a))
            starts.append(dict(source='random global',vector=np.r_[rng.uniform(-np.pi,np.pi,N),a].tolist()))
        inputs[name]=dict(seed=seed,original=s,starts=starts)
    path.write_text(json.dumps(inputs,indent=2)+'\n')

def run(name,inputs,tag='remaining'):
    original=inputs[name]['original'];N=original['N'];area=sum(original['amplitudes']);cap=max(original['amplitudes'])
    baseline=evaluate(original,n=401);bounds=[(None,None)]*N+[(1e-5,cap)]*N
    records=[];fitted=[];path=HERE/(name.lower()+'_'+tag+'_records.json')
    def seq(x):return {**original,'phases':((x[:N]+np.pi)%(2*np.pi)-np.pi).tolist(),'amplitudes':x[N:].tolist(),
        'origin':f'studies/control_refinement/refine_remaining_h.py; seed={inputs[name]["seed"]}; {tag}'}
    def record(fit,stage,index,protocol):
        s=seq(fit.x);m=evaluate(s,n=401);wide=metrics(s,r=.5,n=401)
        row=dict(name=name,stage=stage,start_index=index,controls=s,metrics=m,wide_domain_metrics=wide,
            success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),protocol=protocol)
        records.append(row);path.write_text(json.dumps(records,indent=2)+'\n')
        print(name,stage,index,'I0',m['nominal'],'means',m['native_mean'],m['peak_mean'],'wide',wide['fraction_below_1e4'],fit.success,flush=True)
        return row
    kernels={weight:compiled(N,original['theta'],n=25,peak_weight=weight) for weight in [0.,.15,.5]}
    for k,start in enumerate(inputs[name]['starts']):
        pw=start.get('peak_weight',.15 if k%2==0 else .5);fg,nom,jac=kernels[pw]
        def f(x):v,g=fg(jnp.asarray(x),5.,area);return float(v),np.asarray(g)
        fit=minimize(f,np.asarray(start['vector']),jac=True,method='L-BFGS-B',bounds=bounds,
            options=dict(maxiter=1200,ftol=1e-13,gtol=1e-8,maxls=35))
        row=record(fit,'mean_penalty',k,dict(grid=25,halfwidth=.15,peak_weight=pw,nominal_weight=5.,seed_source=start['source']))
        fitted.append(row)
    # Fine mean polishing preserves diverse basins and imposes exact nominal matching.
    ranked=sorted(fitted,key=lambda r:r['metrics']['native_mean']/baseline['native_mean']+r['metrics']['peak_mean']/baseline['peak_mean'])
    starts=[np.asarray(inputs[name]['starts'][0]['vector'])]
    for row in ranked:
        s=row['controls'];v=np.r_[s['phases'],s['amplitudes']]
        if all(np.linalg.norm(v-x)>.01 for x in starts):starts.append(v)
        if len(starts)>=6:break
    polish_weight=inputs[name].get('polish_peak_weight',.5)
    fg,nom,jac=compiled(N,original['theta'],n=41,peak_weight=polish_weight)
    cons=[dict(type='eq',fun=lambda x:np.asarray(nom(jnp.asarray(x))),jac=lambda x:np.asarray(jac(jnp.asarray(x)))),
        dict(type='ineq',fun=lambda x:area-sum(x[N:]),jac=lambda x:np.r_[np.zeros(N),-np.ones(N)])]
    for k,start in enumerate(starts):
        def f(x):v,g=fg(jnp.asarray(x),0.,area);return float(v),np.asarray(g)
        fit=minimize(f,start,jac=True,method='SLSQP',bounds=bounds,constraints=cons,options=dict(maxiter=800,ftol=1e-12))
        record(fit,'nominal_polish',k,dict(grid=41,halfwidth=.15,peak_weight=polish_weight,nominal='exact'))
    # Cover the full plotted square; nominal matching and resource bounds stay fixed.
    eligible=[r for r in records if r['success'] and r['metrics']['nominal']<1e-12 and sum(r['controls']['amplitudes'])<=area+1e-8]
    eligible.sort(key=lambda r:-r['wide_domain_metrics']['fraction_below_1e4'])
    starts=[np.asarray(inputs[name]['starts'][0]['vector'])]
    for row in eligible:
        s=row['controls'];v=np.r_[s['phases'],s['amplitudes']]
        if all(np.linalg.norm(v-x)>.03 for x in starts):starts.append(v)
        if len(starts)>=4:break
    fn,nom,jac=compiled(N,original['theta'],n=61,peak_weight=0.)
    fp,_,_=compiled(N,original['theta'],n=61,peak_weight=1.)
    u=(np.arange(101)+.5)/101-.5;D,E=np.meshgrid(u,u);e=jnp.asarray(E.ravel());d=jnp.asarray(D.ravel())
    c,s=np.cos(original['theta']/2),np.sin(original['theta']/2)
    def values(v):
        z=jnp.zeros_like(e);o=jnp.ones_like(e)
        def step(carry,pa):
            w,x,y,z=carry;phi,a=pa;om=a*(1+e);q=jnp.sqrt(jnp.maximum(om*om+d*d,1e-24));co=jnp.cos(np.pi*q/2);k=np.pi/2*jnp.sinc(q/2)
            px=k*om*jnp.cos(phi);py=-k*om*jnp.sin(phi);pz=-k*d
            return (co*w-px*x-py*y-pz*z,co*x+w*px+py*z-pz*y,co*y+w*py+pz*x-px*z,co*z+w*pz+px*y-py*x),None
        (w,x,y,z),_=jax.lax.scan(step,(o,z,z,z),(v[:N],v[N:]))
        return (2/3)*((c*x-s*w)**2+y*y+z*z)
    cov=jax.jit(jax.value_and_grad(lambda x,t:1-jnp.mean(jax.nn.sigmoid((1e-4-values(x))/t))))
    def mean(fun,x):v,g=fun(jnp.asarray(x),0.,area);return float(v),np.asarray(g)
    cons=[dict(type='eq',fun=lambda x:np.asarray(nom(jnp.asarray(x))),jac=lambda x:np.asarray(jac(jnp.asarray(x)))),
        dict(type='ineq',fun=lambda x:area-sum(x[N:]),jac=lambda x:np.r_[np.zeros(N),-np.ones(N)]),
        dict(type='ineq',fun=lambda x:baseline['native_mean']*1.1e4-mean(fn,x)[0],jac=lambda x:-mean(fn,x)[1]),
        dict(type='ineq',fun=lambda x:baseline['peak_mean']*1.05e4-mean(fp,x)[0],jac=lambda x:-mean(fp,x)[1])]
    for k,start in enumerate(starts):
        x=start
        for temp in [4e-5,1.5e-5,4e-6]:
            def f(v):a,b=cov(jnp.asarray(v),temp);return float(a),np.asarray(b)
            fit=minimize(f,x,jac=True,method='SLSQP',bounds=bounds,constraints=cons,options=dict(maxiter=600,ftol=1e-10));x=fit.x
            record(fit,'coverage_polish',k,dict(grid=101,halfwidth=.5,temperature=temp,nominal='exact',native_mean_budget=1.1,peak_mean_budget=1.05))
            if not fit.success:break
    print('COMPLETED',name,len(records),'fits',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--gates',nargs='+',choices=NAMES,default=NAMES)
    parser.add_argument('--seed-file',default='remaining_h_seed_inputs.json');parser.add_argument('--tag',choices=['remaining','followup','native'],default='remaining');args=parser.parse_args()
    if args.prepare:prepare()
    else:
        inputs=json.loads((HERE/args.seed_file).read_text())
        for name in args.gates:run(name,inputs,args.tag)
