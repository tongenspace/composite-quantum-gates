"""Reconstruct published AP3/AP4/PD4/AP5/AP6/PD6 branches, then CORPSE-complete.

Low, Yoder & Chuang PRA 89,022341 (2014): Eqs.7,11,12, Table III,
and Sec.V. Rounded published phases are seeds only; solve the published
constraints with 65 decimal digits. These are literature reconstructions,
not newly optimized simultaneous-error sequences.
"""
from pathlib import Path
import json,sys,argparse
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from rectangular import metrics,matrix,infidelity,SX
from scipy.linalg import expm

SEEDS={
 'X':{
 'AP3':[[.87848,-1.93555,2.13129],[2.03611,1.29441,1.64504]],
 'PD4':[[2.30661,-1.30540,-.47998,2.38635],[1.47070,.11256,-2.96678,-1.93782]],
 'AP4':[[1.05532,-2.36238,3.06746,.26240],[2.10426,1.11746,1.80109,1.52196]],
 'AP5':[[1.86484,-2.24227,1.42219,-.43481,1.97474],[.60281,-1.44347,1.45031,-2.28880,2.29567],
        [1.78432,.205507,-2.21897,1.86132,1.69801],[2.17223,.89078,2.05179,1.39042,1.60052]],
 'AP6':[[2.11291,2.26524,-.55309,.48262,2.87662,1.43607],[1.56304,2.92131,-.50059,-1.71787,1.85652,1.29826],
        [.97792,-1.74876,2.49396,2.83905,-.72251,1.19155],[2.26941,.51330,2.35282,1.27588,1.64389,1.56168]],
 'PD6':[[2.71338,-.89153,.34947,2.88508,-2.77240,.12790],[2.40846,-1.52806,3.02225,.14373,-1.19151,1.48740],
        [1.35661,.50760,2.84949,-.34944,-2.58938,-2.05672],[.34769,-2.51801,2.11029,-1.90548,1.90034,-.66420]]},
 'H':{
 'AP3':[[.98173,-1.85668,1.98076],[1.80090,1.42667,1.61136]],
 'PD4':[[2.45079,-.96051,-.28079,2.66423],[1.51926,.73603,-2.38182,-1.76467]],
 'AP4':[[.68460,-2.64974,3.11442,.19308],[1.83302,1.33340,1.70166,1.54125]],
 'AP5':[[1.56763,-2.19365,1.57024,-.61251,1.94290],[.73268,-1.46554,1.41033,-2.18996,2.15661],
        [1.62724,.36022,-2.23779,1.88279,1.64971],[1.86049,1.22698,1.85066,1.44825,1.59325]],
 'AP6':[[1.82740,2.77886,-.11820,.33508,2.95974,1.46074],[1.43968,2.68323,-.67496,-1.73584,1.74153,1.37503],
        [1.09750,-1.53786,2.38588,2.88953,-.61467,1.30211],[1.88994,1.07769,2.11382,1.23820,1.68167,1.55453]],
 'PD6':[[3.12733,-.27154,.71713,-2.88348,-2.34449,.60325],[2.66911,-.96400,-2.59675,.76929,-.86098,1.89620],
        [1.46125,1.00480,2.91140,-.24281,-2.10817,-1.82942],[.61658,-2.23483,2.05399,-1.70970,2.20472,-.51392]]}}

def solve(family,seed,gamma):
    n=len(seed);L=2*n
    want=[sum((-1)**k*mp.binomial((gamma+L)/2,k)*mp.binomial((L-gamma)/2,j-k) for k in range(j+1)) for j in range(n+1)]
    def full(v): return list(v)+[(-1 if family.startswith('AP') else 1)*x for x in reversed(v)]
    def sums(v):
        a=[mp.mpc(1)]+[mp.mpc(0)]*n
        for phi in full(v):
            for j in range(n,0,-1): a[j]+=a[j-1]*mp.exp(1j*(-1)**(j+1)*phi)
        return a
    def f(*v):
        a=sums(v)
        if family.startswith('PD'):
            return [part for j in range(1,n+1,2) for part in (mp.re(a[j])-want[j],mp.im(a[j]))]
        return [(mp.re(a[j])-want[j]) if j%2 else mp.im(a[j]) for j in range(1,n+1)]
    v=list(mp.findroot(f,tuple(mp.mpf(str(x)) for x in seed),tol=mp.mpf('1e-58'),maxsteps=50))
    a=sums(v);res=max(abs(a[j]-want[j]) for j in range(n+1))
    assert res<mp.mpf('1e-50')
    return full(v),res

def mp_validate(phases,angles,n,theta):
    X=mp.matrix([[0,1],[1,0]]);Y=mp.matrix([[0,-1j],[1j,0]]);Z=mp.matrix([[1,0],[0,-1]])
    def U(e,d):
        out=mp.eye(2)
        for p,a in zip(phases,angles): out=mp.expm(-.5j*a*((1+e)*(mp.cos(p)*X-mp.sin(p)*Y)-d*Z))*out
        return out
    # Independent matrix Taylor products in amplitude, retaining factorials.
    product=[mp.eye(2)]+[mp.zeros(2) for _ in range(n+1)]
    for p,a in zip(phases,angles):
        P=-.5j*a*(mp.cos(p)*X-mp.sin(p)*Y);R=mp.expm(P)
        pulse=[R*(P**k)/mp.factorial(k) for k in range(n+2)]
        product=[sum((pulse[k]*product[j-k] for k in range(j+1)),mp.zeros(2)) for j in range(n+2)]
    target=mp.expm(-.5j*theta*X)
    nominal=mp.norm(product[0]-target)
    derivatives=[mp.norm(product[k]*mp.factorial(k)) for k in range(1,n+2)]
    det=mp.norm(mp.matrix([[mp.diff(lambda d:U(0,d)[i,j],0) for j in range(2)] for i in range(2)]))
    assert max([nominal,det]+derivatives[:n])<mp.mpf('1e-43')
    return dict(nominal_residual=float(nominal),pure_amplitude_derivative_norms=[float(v) for v in derivatives],
                first_detuning_derivative_norm=float(det),precision_digits=mp.mp.dps,
                limitation='No claim of cancellation of mixed derivatives or higher pure detuning orders')

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    mp.mp.dps=65;records={};rng=np.random.default_rng(202610211)
    for target,families in SEEDS.items():
        theta=mp.pi if target=='X' else mp.pi/2;gamma=theta/(2*mp.pi)
        for family,seeds in families.items():
            n=int(family[-1])
            for branch,seed in enumerate(seeds,1):
                full,res=solve(family,seed,gamma)
                # Paper product left-to-right vs chronological pulses; plus-Y
                # source axis vs minus-Y manuscript axis both converted here.
                phases=[mp.mpf(0),mp.pi,mp.mpf(0)]+[-v for v in reversed(full)]
                k=mp.asin(mp.sin(theta/2)/2)
                angles=[2*mp.pi+theta/2-k,2*mp.pi-2*k,theta/2-k]+[2*mp.pi]*(2*n)
                name=f'Cin{family}_{branch}_{target}'
                s=dict(name=name,N=2*n+3,theta=float(theta),model='drive',phases=list(map(float,phases)),
                       amplitudes=[1.]*(2*n+3),durations_T=[float(v/mp.pi) for v in angles],
                       origin='Low et al. PRA 89,022341 (2014), Table III / Eq.11 / Sec.V; '+family+' branch '+str(branch))
                valid=mp_validate(phases,angles,n,theta)
                err=0.
                G=expm(-.5j*float(theta)*SX)
                for e,d in rng.uniform(-.5,.5,(24,2)):
                    U=matrix(s,e,d);I=(4-abs(np.trace(G.conj().T@U))**2)/6
                    err=max(err,abs(float(infidelity(s,e,d))-I))
                assert err<1e-13
                small=metrics(s,n=401,peak=True);wide=metrics(s,r=.5,n=401,peak=True)
                records[name]=dict(controls=s,source_seed=seed,source_phase_half=[str(v) for v in full[:n]],
                    published_constraint_residual=float(res),validation=valid|dict(matrix_infidelity_difference=err),
                    small_metrics=small,wide_metrics=wide)
                print(name,'mean',small['mean'],'f4wide',wide['fraction_below_1e4'],flush=True)
    # Exact higher-order NOT references from Jones PRA87,052317 (2013).
    a=-mp.acos((4-mp.sqrt(10))/4);b=2*a+mp.acos(-(1+2*mp.cos(a))/2)
    vectors={'JonesS9':[a,b,b,b-mp.pi,2*b-2*a,b-mp.pi,b,b,a]}
    psi=mp.acos(-mp.mpf(1)/4)
    for label,a in [('ASBO9p',psi),('ASBO9m',-psi)]:
        v=[4*a+psi,3*a+2*psi,2*a+psi,a+mp.pi,mp.mpf(0)]
        vectors[label]=v+[-x for x in reversed(v[:-1])]
    for name,source in vectors.items():
        phases=[-v for v in reversed(source)];angles=[mp.pi]*9
        # n=0 checks nominal/first detuning; separate second detuning below.
        valid=mp_validate(phases,angles,2,mp.pi)
        X=mp.matrix([[0,1],[1,0]]);Y=mp.matrix([[0,-1j],[1j,0]]);Z=mp.matrix([[1,0],[0,-1]])
        def exact(e,d):
            q=mp.sqrt((1+e)**2+d*d);out=mp.eye(2)
            for phase in phases:
                axis=(1+e)*(mp.cos(phase)*X-mp.sin(phase)*Y)-d*Z
                out=(mp.cos(mp.pi*q/2)*mp.eye(2)-1j*mp.sin(mp.pi*q/2)*axis/q)*out
            return out
        dd=mp.norm(mp.matrix([[mp.diff(lambda d:exact(0,d)[i,j],0,2) for j in range(2)] for i in range(2)]))
        ed=mp.norm(mp.matrix([[mp.diff(lambda e:mp.diff(lambda d:exact(e,d)[i,j],0),0) for j in range(2)] for i in range(2)]))
        assert dd<mp.mpf('1e-43')
        valid.update(second_detuning_derivative_norm=float(dd),mixed_amplitude_detuning_derivative_norm=float(ed),
                     limitation='Pure amplitude and pure detuning derivatives checked through order 2; mixed derivative reported separately')
        s=dict(name=name,N=9,theta=float(mp.pi),model='drive',phases=list(map(float,phases)),
               amplitudes=[1.]*9,durations_T=[1.]*9,origin='Jones PRA87,052317 (2013), Eq.'+('(63)-(65)' if name=='JonesS9' else '(62), alpha='+str(psi if name.endswith('p') else -psi)))
        records[name]=dict(controls=s,source_phases=[str(v) for v in source],validation=valid,
                           small_metrics=metrics(s,n=401,peak=True),wide_metrics=metrics(s,r=.5,n=401,peak=True))
        print(name,records[name]['small_metrics']['mean'],records[name]['wide_metrics']['fraction_below_1e4'],flush=True)
    path=BASE/'reconstructed_controls.json'
    if args.verify:
        old=json.loads(path.read_text())
        for name,r in records.items():
            np.testing.assert_allclose(r['controls']['phases'],old[name]['controls']['phases'],rtol=0,atol=1e-14)
        print('All 39 higher-order literature references independently revalidated.',flush=True)
    else: path.write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
