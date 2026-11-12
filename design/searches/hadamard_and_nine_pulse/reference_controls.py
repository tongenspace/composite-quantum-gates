"""Published reduced SKinsC and Jones seven-pulse controls, independently checked."""
from pathlib import Path
import json,sys
import mpmath as mp
import numpy as np
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'workspace/hadamard'
sys.path.insert(0,str(BASE/'workspace/inputs'))
from rectangular import infidelity,matrix_infidelity
mp.mp.dps=60
X=mp.matrix([[0,1],[1,0]]);Y=mp.matrix([[0,-1j],[1j,0]]);Z=mp.matrix([[1,0],[0,-1]]);records={}

def check(name,theta,phases,angles,origin):
    def unitary(e,d):
        q=mp.sqrt((1+e)**2+d*d);out=mp.eye(2)
        for p,t in zip(phases,angles):
            h=(1+e)*(mp.cos(p)*X-mp.sin(p)*Y)-d*Z
            out=(mp.cos(t*q/2)*mp.eye(2)-1j*mp.sin(t*q/2)*h/q)*out
        return out
    target=mp.cos(theta/2)*mp.eye(2)-1j*mp.sin(theta/2)*X
    res=min(mp.norm(unitary(0,0)-target),mp.norm(unitary(0,0)+target));assert res<mp.mpf('1e-50'),(name,res)
    deriv={}
    for a,b in [(1,0),(0,1),(2,0),(1,1),(0,2)]:
        D=mp.matrix([[mp.diff(lambda e:mp.diff(lambda d:unitary(e,d)[i,j],mp.mpf(0),b),mp.mpf(0),a) for j in range(2)] for i in range(2)])
        deriv[f'{a}{b}']=float(mp.norm(D)/mp.sqrt(2))
    assert deriv['10']<1e-45 and deriv['01']<1e-45,(name,deriv)
    s=dict(name=name,N=len(phases),theta=float(theta),model='drive',amplitudes=[1.]*len(phases),durations_T=[float(t/mp.pi) for t in angles],phases=[float(p) for p in phases],origin=origin)
    err=0.;rng=np.random.default_rng(2026102361)
    for e,d in rng.uniform(-.5,.5,(64,2)):err=max(err,abs(float(infidelity(s,e,d))-matrix_infidelity(s,e,d)))
    assert err<1e-13
    records[name]=dict(controls=s,nominal_matrix_residual=float(res),derivative_norms=deriv,matrix_error=err,source_phases_radians=[str(p) for p in phases],source_angles_radians=[str(t) for t in angles])
    print(name,deriv,flush=True)

for target,theta in [('X',mp.pi),('H',mp.pi/2)]:
    k=mp.asin(mp.sin(theta/2)/2);a=theta/2-k;b=2*mp.pi-theta/2-k;alpha=mp.acos(-(2*mp.pi-theta)/(4*mp.pi))
    check('rSKinsC_'+target,theta,[0,-mp.pi,-mp.pi+alpha,-mp.pi-alpha,-mp.pi,0],[a,b,2*mp.pi,2*mp.pi,a,a],'Kukita et al., JPSJ 91,104001 (2022), corrected reduced SKinsC Eq.(9) in arXiv:2112.12945')

alpha=-mp.acos(-mp.sqrt((4+mp.sqrt(13))/2)/2);beta=alpha-2*mp.pi/3
v=[alpha,2*beta,beta,-2*mp.pi/3,-mp.pi/3-alpha+2*beta,-2*mp.pi/3-2*alpha+4*beta,-mp.pi-2*alpha+3*beta]
check('Jones7a',mp.pi,[-p for p in reversed(v)],[mp.pi]*7,'Jones PRA 87,052317 (2013), Eqs.(56)-(57), best birobust branch')
for name,a in [('Jones7b',-mp.acos((3-mp.sqrt(61))/16)),('Jones7c',mp.acos((mp.sqrt(61)-3)/16))]:
    v=[a,2*mp.pi/3+2*a,7*mp.pi/3+3*a,10*mp.pi/3+4*a,7*mp.pi/3+3*a,2*mp.pi/3+2*a,a]
    check(name,mp.pi,[-p for p in reversed(v)],[mp.pi]*7,'Jones PRA 87,052317 (2013), Eqs.(58)-(60)')

registry=json.loads((OUT/'controls.json').read_text())
for name,row in records.items():registry['sequences'][name]=row['controls']
for group,target in [('X_table','X'),('H_table','H')]:
    idx=registry[group].index('rCinBB_'+target)+1
    registry[group][idx:idx]=['rSKinsC_'+target]+(['Jones7a','Jones7b','Jones7c'] if target=='X' else [])
    registry[group]=list(dict.fromkeys(registry[group]))
(OUT/'controls.json').write_text(json.dumps(registry,indent=2));(OUT/'new_reference_validation.json').write_text(json.dumps(records,indent=2))
