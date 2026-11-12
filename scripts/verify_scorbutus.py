"""Reconstruct and verify published SCORBUTUS in the manuscript convention."""
from pathlib import Path
import json, sys
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import expm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))
from rectangular import infidelity, matrix
from static_results import measure


def construction(theta, name):
    # Kukita, Kiya and Kondo, arXiv:2112.12945, Eqs. (5),(8).
    # sinc is unnormalised, sin(x)/x, with the root in (0,pi].
    t1 = np.pi if theta == np.pi else brentq(
        lambda x: np.sinc(x / np.pi) - 2*np.cos(theta/2)/np.pi,
        1e-8, np.pi, xtol=1e-14)
    p1 = np.arccos(np.clip(-np.pi*np.cos(t1)/(2*t1*np.sin(theta/2)), -1, 1))
    p2 = p1 - np.arccos(-np.pi/(2*t1))
    tr = np.arccos((1 - np.pi*np.sin(t1/2)**2/t1)/2)
    angles = np.array([t1, tr, np.pi+2*tr, tr, t1])
    # Negate source phases to match the manuscript's off-diagonal convention.
    phases = -np.array([p1, p2+np.pi, p2, p2+np.pi, p1])
    return dict(name=name, N=5, theta=theta, phases=phases.tolist(),
                amplitudes=[1.]*5, durations_T=(angles/np.pi).tolist(),
                origin='Kukita et al., JPSJ 91,104001 (2022); Eqs.(5),(8)',
                model='drive')


def independent(s, eps, delta):
    # Direct Pauli exponential, separate from quaternion implementation.
    sx=np.array([[0,1],[1,0]],complex)
    sy=np.array([[0,-1j],[1j,0]],complex)
    sz=np.diag([1,-1])
    u=np.eye(2,dtype=complex)
    for p,t in zip(s['phases'],s['durations_T']):
        u=expm(-.5j*np.pi*t*((1+eps)*(np.cos(p)*sx-np.sin(p)*sy)-delta*sz))@u
    g=expm(-.5j*s['theta']*sx)
    return u, float((4-abs(np.trace(g.conj().T@u))**2)/6)


def source_convention(s, eps, delta):
    """Source Eq. (2), positive sigma_y and f=-delta/(1+eps)."""
    sx=np.array([[0,1],[1,0]],complex)
    sy=np.array([[0,-1j],[1j,0]],complex)
    sz=np.diag([1,-1]);u=np.eye(2,dtype=complex)
    f=-delta/(1+eps)
    for manuscript_phase,t in zip(s['phases'],s['durations_T']):
        phi=-manuscript_phase
        u=expm(-.5j*np.pi*t*(1+eps)*(np.cos(phi)*sx+np.sin(phi)*sy+f*sz))@u
    return u


def main():
    out=REPO/'outputs/scorbutus'
    out.mkdir(parents=True,exist_ok=True)
    results={}
    for theta,name in [(np.pi,'SCORBUTUS_X'),(np.pi/2,'SCORBUTUS_H')]:
        s=construction(theta,name)
        small=measure(s,801,.15); wide=measure(s,801,.5)
        u=np.linspace(-.15,.15,1601)
        e=np.r_[u,u,np.full(len(u),-.15),np.full(len(u),.15)]
        d=np.r_[np.full(len(u),-.15),np.full(len(u),.15),u,u]
        small['maximum']=max(small['maximum'],float(infidelity(s,e,d).max()))
        rng=np.random.default_rng(20261026)
        points=rng.uniform(-.5,.5,(32,2))
        difference=max(abs(float(infidelity(s,e,d))-independent(s,e,d)[1]) for e,d in points)
        convention_difference=max(float(np.max(abs(source_convention(s,e,d)-matrix(s,e,d)))) for e,d in points)
        assert convention_difference<1e-13
        h=1e-5
        derivative_eps=np.linalg.norm((matrix(s,h,0)-matrix(s,-h,0))/(2*h))
        derivative_det=np.linalg.norm((matrix(s,0,h)-matrix(s,0,-h))/(2*h))
        results[name]=dict(control=s,normalized_area=float(np.pi*sum(s['durations_T'])/theta),
                           nominal=float(infidelity(s,0,0)),small=small,wide=wide,
                           validation=dict(max_matrix_infidelity_difference=difference,
                                           source_convention_max_unitary_difference=convention_difference,
                                           first_derivative_eps_norm=derivative_eps,
                                           first_derivative_det_norm=derivative_det))
        frozen=json.loads((REPO/'data/scorbutus_verified.json').read_text())[name]
        np.testing.assert_allclose(s['phases'],frozen['control']['phases'],atol=2e-14,rtol=0)
        np.testing.assert_allclose(s['durations_T'],frozen['control']['durations_T'],atol=2e-14,rtol=0)
        for domain,new in [('small',small),('wide',wide)]:
            for metric in ['mean','maximum','fraction','connected_fraction']:
                assert abs(new[metric]-frozen[domain][metric])<2e-13,(name,domain,metric)
        p=max(s['amplitudes']); scaled={**s,'amplitudes':(np.array(s['amplitudes'])/p).tolist(),'durations_T':(np.array(s['durations_T'])*p).tolist()}
        for e,d in points:
            np.testing.assert_allclose(matrix(s,e,p*d),matrix(scaled,e,d),atol=1e-13,rtol=0)
        g=independent(s,0.,0.)[0]
        from scipy.linalg import expm
        target=expm(-.5j*s['theta']*np.array([[0,1],[1,0]],complex))
        phase=np.trace(target.conj().T@g); phase/=abs(phase)
        assert np.linalg.norm(g-phase*target)<1e-13
        assert difference<1e-13
        assert max(derivative_eps,derivative_det)<1e-7
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
