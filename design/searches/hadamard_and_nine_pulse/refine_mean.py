"""Replacement shorter controls using the existing finite-domain mean objective."""
import argparse,json
import numpy as np
import jax,jax.numpy as jnp
from scipy.optimize import minimize
from design_h import OUT,compiled,errors,split,dynamic,metrics,infidelity

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gate',choices=['X','H'],required=True);ap.add_argument('--N',type=int,nargs='+',default=[9,10]);ap.add_argument('--starts',type=int,default=6);a=ap.parse_args()
    reg=json.loads((OUT/'search/search_input_controls.json').read_text())['sequences'];g=a.gate;theta=np.pi if g=='X' else np.pi/2;old=reg[g+'9c'];area=sum(old['amplitudes']);cap=max(old['amplitudes'])
    for N in a.N:
        rng=np.random.default_rng(2026102330+N+(100 if g=='H' else 0));starts=[];labels=[]
        for name in [g+'9c',g+'7c',g+'5c']+(['X9b','X9a','JonesS9'] if g=='X' else []):
            s=reg[name]
            if s['N']<=N:
                v=split(s,N);v[N:]=np.clip(v[N:],.05,cap);v[N:]*=min(1,area/sum(v[N:]));starts.append(v);labels.append(name)
        while len(starts)<a.starts:
            v=starts[len(starts)%3];scale=[.2,.5,.9][len(starts)%3]
            amps=np.clip(v[N:]+rng.normal(0,.05,N),.05,cap);amps*=min(1,area/sum(amps));starts.append(np.r_[v[:N]+rng.normal(0,scale,N),amps]);labels.append('deterministic perturbation')
        starts=starts[:a.starts];labels=labels[:a.starts];_,nom,jac=compiled(N,theta,n=3)
        cons=[dict(type='eq',fun=lambda v:np.asarray(nom(jnp.asarray(v))),jac=lambda v:np.asarray(jac(jnp.asarray(v)))),dict(type='ineq',fun=lambda v:area-sum(v[N:]),jac=lambda v:np.r_[np.zeros(N),-np.ones(N)])]
        def objective(n,ns):
            u=(np.arange(n)+.5)*.3/n-.15;D,E=np.meshgrid(u,u);e=jnp.asarray(E.ravel());d=jnp.asarray(D.ravel())
            us=(np.arange(ns)+.5)*.3/ns-.15;Ds,Es=np.meshgrid(us,us);es=jnp.asarray(Es.ravel());ds=jnp.asarray(Ds.ravel())
            def f(v):
                if g=='X':
                    native=errors(v,N,e,d/jnp.max(v[N:]),theta=theta)
                    common=errors(v,N,e,d,theta=theta)
                    return 1e4*(native.mean()+common.mean())/2
                z=errors(v,N,e,d,theta=theta);zs=errors(v,N,es,ds,True,theta=theta)
                return 1e4*(.3*z.mean()+.7*zs.mean())
            return jax.jit(jax.value_and_grad(f))
        funcs=[objective(19,9),objective(31,17)];records=[]
        protocol=dict(N=N,gate=g,seed=2026102330+N+(100 if g=='H' else 0),area_cap=area,peak_cap=cap,start_labels=labels,starts=[v.tolist() for v in starts],objective='existing equal-weight native/common-peak B_0.15 mean' if g=='X' else 'existing 0.3 rectangular / 0.7 finite-edge B_0.15 mean',additional_anisotropy_objective=False)
        (OUT/f'search/{g}{N}_mean_protocol.json').write_text(json.dumps(protocol,indent=2))
        for i,v in enumerate(starts):
            for stage,(fg,it) in enumerate(zip(funcs,[250,500])):
                def fun(v):f,grad=fg(jnp.asarray(v));return float(f),np.asarray(grad)
                fit=minimize(fun,v,jac=True,method='SLSQP',bounds=[(None,None)]*N+[(.05,cap)]*N,constraints=cons,options=dict(maxiter=it,ftol=2e-11));v=fit.x
                print('MEAN',g,N,i,stage,fit.success,fit.fun,flush=True)
            s=dict(name=f'{g}{N}c',N=N,theta=theta,model='drive',durations_T=[1.]*N,phases=(v[:N]%(2*np.pi)).tolist(),amplitudes=v[N:].tolist(),origin='research/numerical/refine_mean.py; existing finite-domain mean objective')
            row=dict(controls=s,start=i,success=bool(fit.success),message=str(fit.message),objective=float(fit.fun),nominal=float(infidelity(s,0,0)),area=sum(s['amplitudes']))
            if row['success'] and row['nominal']<1e-12 and row['area']<=area+1e-7:
                row['small']=metrics(s,n=401,peak=True);row['native']=metrics(s,n=401);row['wide']=metrics(s,r=.5,n=401,peak=True)
                if g=='H':row['selection_dynamic']=dynamic(s);row['score']=.3*row['small']['mean']+.7*row['selection_dynamic']['mean']
                else:row['score']=.5*(row['small']['mean']+row['native']['mean'])
                print('VALID',g,N,i,row['score'],row['wide']['fraction_below_1e4'],flush=True)
            records.append(row);(OUT/f'search/{g}{N}_mean_records.json').write_text(json.dumps(records,indent=2))
        valid=[r for r in records if 'score' in r]
        if not valid:raise RuntimeError('No valid fit')
        chosen=min(valid,key=lambda r:r['score']);(OUT/f'search/{g}{N}_mean_selected.json').write_text(json.dumps(chosen,indent=2));print('SELECTED',g,N,chosen['start'],chosen['score'],flush=True)

if __name__=='__main__':main()
