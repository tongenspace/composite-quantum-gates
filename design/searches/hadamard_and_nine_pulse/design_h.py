"""Deterministic H catalogue search with finite-edge-aware objectives.

Common physical peak Omega_max/(2*pi)=5 MHz; 4 ns area-preserving cosine
edges. Stored amplitudes use Omega_0*T=pi. The physical segment body is
100*max(a) ns at the common peak. No stochastic-noise fitting is claimed:
independent continuous-detuning trajectories are used for candidate selection.
"""
from pathlib import Path
import argparse,importlib.util,json,sys,time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64',True)
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
OLD=BASE/'workspace/inputs';OUT=BASE/'workspace/hadamard'
OUT.mkdir(exist_ok=True);(OUT/'search').mkdir(exist_ok=True)
sys.path[:0]=[str(OLD),str(OLD/'studies/control_refinement'),str(OLD/'studies/pulse_shape')]
from search import compiled
from rectangular import infidelity,metrics
import shaped_model as sm
REG=json.loads((BASE/'workspace/initial/pulse_controls.json').read_text())['sequences']
A1=(3-2*np.sqrt(3))/12;A2=(3+2*np.sqrt(3))/12

def split(s,N):
    p=list(s['phases']);a=list(s['amplitudes'])
    while len(p)<N:
        k=int(np.argmax(a));v=a[k]/2;p[k:k+1]=[p[k],p[k]];a[k:k+1]=[v,v]
    return np.r_[p,a]

def axis(n,r,endpoint=False):
    u=np.linspace(-r,r,n) if endpoint else (np.arange(n)+.5)*2*r/n-r
    D,E=np.meshgrid(u,u);return jnp.array(E.ravel()),jnp.array(D.ravel())

def errors(v,N,e,d,shaped=False,edge_steps=3,flat_steps=3,theta=np.pi/2):
    peak=jnp.max(v[N:]);p=v[:N];a=v[N:];det=peak*d
    if shaped:
        r=.04/peak
        hs=[];fs=[]
        for kind,count in [('rise',edge_steps),('flat',flat_steps),('fall',edge_steps)]:
            for i in range(count):
                u1=(i+.5-np.sqrt(3)/6)/count;u2=(i+.5+np.sqrt(3)/6)/count
                if kind=='rise':f1=.5*(1-np.cos(np.pi*u1));f2=.5*(1-np.cos(np.pi*u2))
                elif kind=='fall':f1=.5*(1+np.cos(np.pi*u1));f2=.5*(1+np.cos(np.pi*u2))
                else:f1=f2=1.
                h=(1-r)/count if kind=='flat' else r/count
                for b1,b2 in [(A2,A1),(A1,A2)]:hs.append(h);fs.append(b1*f1+b2*f2)
        h=jnp.tile(jnp.array(hs),N);f=jnp.tile(jnp.array(fs),N)
        p=jnp.repeat(p,len(fs));a=jnp.repeat(a,len(fs));det=.5*det
    else:h=jnp.ones(N);f=jnp.ones(N)
    def step(q,entry):
        phi,amp,dt,env=entry;om=amp*env*(1+e)
        norm=jnp.sqrt(jnp.maximum(om*om+det*det,1e-30));ang=.5*jnp.pi*dt*norm
        c=jnp.cos(ang);k=.5*jnp.pi*dt*jnp.sinc(dt*norm/2)
        px=k*om*jnp.cos(phi);py=-k*om*jnp.sin(phi);pz=-k*det
        w,x,y,z=q
        return (c*w-px*x-py*y-pz*z,c*x+w*px+py*z-pz*y,c*y+w*py+pz*x-px*z,c*z+w*pz+px*y-py*x),None
    zero=jnp.zeros_like(e);one=jnp.ones_like(e)
    (w,x,y,z),_=jax.lax.scan(step,(one,zero,zero,zero),(p,a,h,f))
    return (2/3)*((np.cos(theta/2)*x-np.sin(theta/2)*w)**2+y*y+z*z)

def objective(N,n=21,ns=11,mode='p',r=.15,endpoint=False,theta=np.pi/2):
    e,d=axis(n,r,endpoint);es,ds=axis(ns,r,endpoint)
    def loss(v):
        a=errors(v,N,e,d,theta=theta);b=errors(v,N,es,ds,True,theta=theta)
        if mode=='p':return 1e4*(.3*a.mean()+.7*b.mean())
        if mode=='c':return .3*jnp.mean(a/(a+1e-4))+.7*jnp.mean(b/(b+1e-4))
        return jax.scipy.special.logsumexp(50*jnp.concatenate([a,b])/1e-4)/50
    return jax.jit(jax.value_and_grad(loss))

def seq(v,N,name):
    return dict(name=name,N=N,theta=float(np.pi/2),model='drive',durations_T=[1.]*N,
                phases=((v[:N]+np.pi)%(2*np.pi)-np.pi).tolist(),amplitudes=v[N:].tolist(),
                origin='research/numerical/design_h.py; deterministic edge-aware search')

def dynamic(s,n=31,r=.15,samples=16,seed=2026102201,edge_ns=4.,noise=True,steps=6):
    peak=max(s['amplitudes']);sm.T_NS=100*peak;sm.F_REF_HZ=5e6/peak
    u=(np.arange(n)+.5)*2*r/n-r;D,E=np.meshgrid(u,u)
    vals,err=sm.evaluate(s,(edge_ns/100)/peak,E,peak*D,samples=samples,seed=seed,
           noise_scale=1. if noise else 0.,edge_steps=steps,flat_steps=max(4,steps),modes=96,chunk=2048)
    z=vals.mean(axis=0)
    means=vals.reshape((len(vals),-1)).mean(axis=1)
    pairs=.5*(means[:len(means)//2]+means[len(means)//2:]) if noise and len(means)>1 else np.array([0.])
    se=float(pairs.std(ddof=1)/np.sqrt(len(pairs))) if len(pairs)>1 else 0.
    return dict(mean=float(z.mean()),maximum=float(z.max()),coverage=float(np.mean(z<=1e-4)),norm_error=err,mean_standard_error=se)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,nargs='+',default=[7,9,11,13,15]);ap.add_argument('--starts',type=int,default=4)
    ap.add_argument('--iterations',type=int,default=600);ap.add_argument('--suffix',default='');args=ap.parse_args()
    selected_file=OUT/('search/selected_h'+args.suffix+'.json');selected=json.loads(selected_file.read_text()) if selected_file.exists() else {}
    for N in args.N:
        seed=202610220+N;rng=np.random.default_rng(seed)
        baseline=REG['H7'] if N==7 else REG['H15p'] if N==15 else None
        area=sum(baseline['amplitudes']) if baseline else float(N)
        cap=max(baseline['amplitudes']) if baseline else 2.
        starts=[];labels=[]
        for name in (['H7','H5'] if N==7 else ['H15p','H15c','H5','H10p'] if N==15 else ['H5','H7','H10p','H8']):
            if REG[name]['N']<=N:starts.append(split(REG[name],N));labels.append('split '+name)
        while len(starts)<args.starts:
            v=starts[0];starts.append(np.r_[v[:N]+rng.normal(0,.1,N),np.clip(v[N:]+rng.normal(0,.025,N),.05,cap)]);labels.append('seeded perturbation')
        starts=starts[:args.starts];labels=labels[:args.starts]
        protocol=dict(N=N,seed=seed,area_cap=area,peak_cap=cap,starts=[v.tolist() for v in starts],start_labels=labels,
          mean_weights=dict(rectangular=.3,finite_edges=.7),peak_Rabi_Hz=5e6,edge_ns=4.,stochastic_selection='16 fixed independent Gaussian 1/f trajectories on 31x31 midpoint grid; 1 Hz--1 MHz; A=3.6e5 Hz^2',
          precision_goal='minimize weighted small-domain means',coverage_goal='weighted saturated threshold loss on B_0.5',nominal_constraints='full target quaternion residual zero')
        (OUT/f'search/H{N}_protocol.json').write_text(json.dumps(protocol,indent=2))
        records=[];path=OUT/f'search/H{N}_records.json'
        _,nom,jac=compiled(N,np.pi/2,n=3)
        cons=[dict(type='eq',fun=lambda v:np.asarray(nom(jnp.asarray(v))),jac=lambda v:np.asarray(jac(jnp.asarray(v)))),
              dict(type='ineq',fun=lambda v:area-sum(v[N:]),jac=lambda v:np.r_[np.zeros(N),-np.ones(N)])]
        def fit(start,fg,constraints,cap0,iterations):
            def fun(v):f,g=fg(jnp.asarray(v));return float(f),np.array(g)
            return minimize(fun,start,jac=True,method='SLSQP',bounds=[(None,None)]*N+[(.05,cap0)]*N,constraints=constraints,
                            options=dict(maxiter=iterations,ftol=2e-11))
        def record(result,kind,index,grid):
            s=seq(result.x,N,f'H{N}'+kind[0]);v=result.x
            r=dict(controls=s,kind=kind,start=index,grid=grid,success=bool(result.success),message=str(result.message),iterations=int(result.nit),
                   nominal=float(infidelity(s,0,0)),area=float(sum(v[N:])),peak=float(max(v[N:])),objective=float(result.fun))
            records.append(r);path.write_text(json.dumps(records,indent=2))
            print('H',N,kind,index,'success',r['success'],'nominal',r['nominal'],'objective',r['objective'],flush=True)
            return r
        fg=objective(N,n=19,ns=9)
        for i,v in enumerate(starts):
            result=fit(v,fg,cons,cap,args.iterations);record(result,'precision_coarse',i,19)
        valid=[r for r in records if r['nominal']<1e-10 and r['area']<=area+1e-7]
        if not valid:raise RuntimeError('No nominally matched depth start')
        valid.sort(key=lambda r:r['objective'])
        fg=objective(N,n=31,ns=17)
        for r in valid[:2]:
            v=np.r_[r['controls']['phases'],r['controls']['amplitudes']]
            result=fit(v,fg,cons,cap,args.iterations);record(result,'precision_fine',r['start'],31)
        eligible=[r for r in records if r['kind']=='precision_fine' and r['nominal']<1e-12 and r['area']<=area+1e-8 and r['success']]
        if not eligible:eligible=[r for r in records if r['nominal']<1e-12 and r['area']<=area+1e-8 and r['success']]
        if not eligible:raise RuntimeError('No converged feasible precision fit')
        for r in eligible:
            r['selection_dynamic']=dynamic(r['controls']);r['selection_rect']=metrics(r['controls'],n=101,peak=True)
            r['selection_score']=.3*r['selection_rect']['mean']+.7*r['selection_dynamic']['mean']
        chosen=min(eligible,key=lambda r:r['selection_score']);p=chosen['controls']|dict(name=f'H{N}p')
        selected[f'H{N}p']=dict(controls=p,selection=chosen);selected_file.write_text(json.dumps(selected,indent=2))
        print('SELECTED P',N,'combined mean',chosen['selection_score'],flush=True)
        # Width controls have the selected precision fit's area and peak limits.
        acap=sum(p['amplitudes']);pcap=max(p['amplitudes'])
        wc=[cons[0],dict(type='ineq',fun=lambda v:acap-sum(v[N:]),jac=lambda v:np.r_[np.zeros(N),-np.ones(N)])]
        if N>=13:
            center=objective(N,n=11,ns=11,mode='max',r=.20,endpoint=True)
            def cb(v):f,g=center(jnp.asarray(v));return 1-f,-np.asarray(g)
            wc.append(dict(type='ineq',fun=lambda v:float(cb(v)[0]),jac=lambda v:cb(v)[1]))
        else:
            # Preserve useful central accuracy while improving threshold area.
            mean=objective(N,n=15,ns=11)
            ceiling=2*float(mean(jnp.r_[np.array(p['phases']),np.array(p['amplitudes'])])[0])
            def mb(v):f,g=mean(jnp.asarray(v));return ceiling-f,-np.asarray(g)
            wc.append(dict(type='ineq',fun=lambda v:float(mb(v)[0]),jac=lambda v:mb(v)[1]))
        seeds=[np.r_[p['phases'],p['amplitudes']]]
        if N==15:seeds.append(np.r_[REG['H15c']['phases'],np.minimum(REG['H15c']['amplitudes'],pcap)])
        for scale in [.02,.08]:
            v=seeds[0];seeds.append(np.r_[v[:N]+rng.normal(0,scale,N),np.clip(v[N:]+rng.normal(0,scale/4,N),.05,pcap)])
        fg=objective(N,n=31,ns=13,mode='c',r=.5)
        for i,v in enumerate(seeds[:3]):
            result=fit(v,fg,wc,pcap,args.iterations+300);record(result,'coverage_coarse',i,31)
        cand=[r for r in records if r['kind']=='coverage_coarse' and r['success'] and r['nominal']<1e-12 and r['area']<=acap+1e-8]
        cand.sort(key=lambda r:r['objective'])
        fg=objective(N,n=51,ns=21,mode='c',r=.5)
        for r in cand[:2]:
            v=np.r_[r['controls']['phases'],r['controls']['amplitudes']];result=fit(v,fg,wc,pcap,args.iterations)
            record(result,'coverage_fine',r['start'],51)
        eligible=[r for r in records if r['kind'].startswith('coverage') and r['success'] and r['nominal']<1e-12 and r['area']<=acap+1e-8]
        if not eligible:raise RuntimeError('No converged width fit; retain depth records and refine constraints')
        for r in eligible:
            r['selection_dynamic']=dynamic(r['controls'],n=61,r=.5);r['selection_rect']=metrics(r['controls'],n=201,r=.5,peak=True)
            r['selection_score']=.3*r['selection_rect']['fraction_below_1e4']+.7*r['selection_dynamic']['coverage']
        chosen=max(eligible,key=lambda r:r['selection_score']);c=chosen['controls']|dict(name=f'H{N}c')
        selected[f'H{N}c']=dict(controls=c,selection=chosen,paired_area_cap=acap,paired_peak_cap=pcap)
        selected_file.write_text(json.dumps(selected,indent=2));path.write_text(json.dumps(records,indent=2))
        print('SELECTED C',N,'combined coverage',chosen['selection_score'],flush=True)
    print('Search finished.',flush=True)
if __name__=='__main__':main()
