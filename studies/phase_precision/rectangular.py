"""Physical two-level rectangular model; all detunings use a fixed reference."""
import numpy as np
from scipy.linalg import expm
SX=np.array([[0,1],[1,0]],complex);SY=np.array([[0,-1j],[1j,0]],complex);SZ=np.diag([1,-1])
def evolve(seq,eps,det):
    eps,det=np.broadcast_arrays(np.asarray(eps,dtype=float),np.asarray(det,dtype=float))
    w=np.ones_like(eps);x=np.zeros_like(eps);y=np.zeros_like(eps);z=np.zeros_like(eps)
    for p,a,t in zip(seq['phases'],seq['amplitudes'],seq['durations_T']):
        om=a*(1+eps);q=np.sqrt(om*om+det*det);duration=np.pi*t
        c=np.cos(duration*q/2);k=duration/2*np.sinc(duration*q/(2*np.pi))
        px=k*om*np.cos(p);py=-k*om*np.sin(p);pz=-k*det
        w,x,y,z=c*w-px*x-py*y-pz*z,c*x+w*px+py*z-pz*y,c*y+w*py+pz*x-px*z,c*z+w*pz+px*y-py*x
    return np.stack([w,x,y,z],axis=-1)
def infidelity(seq,eps,det):
    q=evolve(seq,eps,det);c,s=np.cos(seq['theta']/2),np.sin(seq['theta']/2)
    return (2/3)*((c*q[...,1]-s*q[...,0])**2+q[...,2]**2+q[...,3]**2)
def matrix(seq,eps,det):
    U=np.eye(2,dtype=complex)
    for p,a,t in zip(seq['phases'],seq['amplitudes'],seq['durations_T']):
        H=.5*(a*(1+eps)*(np.cos(p)*SX-np.sin(p)*SY)-det*SZ)
        U=expm(-1j*np.pi*t*H)@U
    return U
def matrix_infidelity(seq,eps,det):
    U=matrix(seq,eps,det);G=expm(-.5j*seq['theta']*SX)
    return float((4-abs(np.trace(G.conj().T@U))**2)/6)
def metrics(seq,r=.15,n=801,peak=False):
    u=(np.arange(n)+.5)*2*r/n-r;D,E=np.meshgrid(u,u)
    scale=max(seq['amplitudes']) if peak else 1.
    z=infidelity(seq,E,scale*D)
    p=max(seq['amplitudes']);theta=seq['theta']
    return dict(name=seq['name'],N=seq['N'],center=float(infidelity(seq,0,0)),mean=float(z.mean()),
        fraction_below_1e4=float(np.mean(z<=1e-4)),area_over_target=float(np.pi*np.dot(seq['amplitudes'],seq['durations_T'])/theta),
        duration_over_target=float(np.pi*p*sum(seq['durations_T'])/theta))
