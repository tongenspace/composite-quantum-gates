"""Two-level Hermitian evolution for flat-top controls and Gaussian 1/f noise.

Physical reference: Omega_0/(2*pi)=5 MHz; T=100 ns; Omega_0*T=pi.
Each pulse preserves plateau amplitude a_k*Omega_0, phase, and area pi*a_k.
Cosine edges each last r*T; plateau length (1-r)*T; total width (1+r)*T.
Time-dependent detuning is delta_0*Omega_0 plus a stationary Gaussian
frequency fluctuation. The latter is generated continuously over the gate,
with one-sided cyclic-frequency PSD S_f(f)=A/f on explicit finite cutoffs.
All realizations use a 2x2 Hermitian Hamiltonian; no dissipative terms.
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
jax.config.update('jax_enable_x64',True)
T_NS=100.;F_REF_HZ=5e6;NOISE_A=3.6e5
C1=.5-np.sqrt(3)/6;C2=.5+np.sqrt(3)/6
A1=(3-2*np.sqrt(3))/12;A2=(3+2*np.sqrt(3))/12

def envelope(t,r):
    t=np.asarray(t)
    if r==0:return np.where((t>=0)&(t<=1),1.,0.)
    return np.where(t<0,0.,np.where(t<r,.5*(1-np.cos(np.pi*t/r)),
                    np.where(t<=1,1.,np.where(t<=1+r,.5*(1+np.cos(np.pi*(t-1)/r)),0.))))

def schedule(seq,r,edge_steps=8,flat_steps=4):
    if not 0<=r<1:raise ValueError('Edge duration must be smaller than reference T.')
    out=[]
    for pulse,(phi,a) in enumerate(zip(seq['phases'],seq['amplitudes'])):
        parts=[(0.,r,edge_steps),(r,1.,flat_steps),(1.,1+r,edge_steps)] if r else [(0.,1.,flat_steps)]
        for left,right,count in parts:
            h=(right-left)/count
            for i in range(count):
                l=left+i*h;t1=l+C1*h;t2=l+C2*h
                f1=float(envelope(t1,r));f2=float(envelope(t2,r))
                # The earlier exponential is weighted toward the earlier node.
                for b1,b2 in [(A2,A1),(A1,A2)]:
                    out.append((h,a*(b1*f1+b2*f2)*np.cos(phi),
                                -a*(b1*f1+b2*f2)*np.sin(phi),
                                pulse*(1+r)+t1,pulse*(1+r)+t2,b1,b2))
    return np.asarray(out)

def frequency_basis(fmin=1.,fmax=1e6,modes=96,A=NOISE_A):
    edges=np.geomspace(fmin,fmax,modes+1)
    return np.sqrt(edges[:-1]*edges[1:]),np.sqrt(A*np.diff(np.log(edges)))

def gaussian_coefficients(samples,modes=96,seed=20261013):
    if samples%2:raise ValueError('Use even samples for antithetic pairs.')
    rng=np.random.default_rng(seed)
    v=rng.standard_normal((samples//2,2*modes))
    return np.concatenate([v,-v],axis=0)

def noise_at(times,coefficients,fmin=1.,fmax=1e6,A=NOISE_A):
    modes=coefficients.shape[1]//2;f,w=frequency_basis(fmin,fmax,modes,A)
    angles=2*np.pi*np.asarray(times)[:,None]*T_NS*1e-9*f[None,:]
    basis=np.concatenate([np.cos(angles)*w,np.sin(angles)*w],axis=1)/F_REF_HZ
    return basis@coefficients.T

def prepared(seq,r,samples=32,seed=20261013,fmin=1.,fmax=1e6,noise_scale=1.,edge_steps=8,flat_steps=4,modes=96,coefficients=None):
    tab=schedule(seq,r,edge_steps,flat_steps)
    if noise_scale:
        coefficients=gaussian_coefficients(samples,modes,seed) if coefficients is None else coefficients
        v1=noise_at(tab[:,3],coefficients,fmin,fmax)*noise_scale
        v2=noise_at(tab[:,4],coefficients,fmin,fmax)*noise_scale
        noise=tab[:,5,None]*v1+tab[:,6,None]*v2
    else:noise=np.zeros((len(tab),1))
    return tab[:,:3],noise,coefficients

@partial(jax.jit,static_argnames=('theta',))
def evolve_jax(e,d,steps,noise,theta):
    def step(q,entry):
        h,px0,py0=entry[:3];eta=entry[3:]
        px=px0*(1+e[None,:]);py=py0*(1+e[None,:]);pz=-(.5*d[None,:]+eta[:,None])
        norm=jnp.sqrt(jnp.maximum(px*px+py*py+pz*pz,1e-30))
        arg=.5*jnp.pi*h*norm;c=jnp.cos(arg);k=.5*jnp.pi*h*jnp.sinc(.5*h*norm)
        px,py,pz=k*px,k*py,k*pz
        w,x,y,z=q
        return (c*w-px*x-py*y-pz*z,c*x+w*px+py*z-pz*y,
                c*y+w*py+pz*x-px*z,c*z+w*pz+px*y-py*x),None
    shape=(noise.shape[1],e.shape[0]);one=jnp.ones(shape);zero=jnp.zeros(shape)
    (w,x,y,z),_=jax.lax.scan(step,(one,zero,zero,zero),jnp.concatenate([steps,noise],axis=1))
    c,s=np.cos(theta/2),np.sin(theta/2)
    values=(2/3)*((c*x-s*w)**2+y*y+z*z)
    return values,jnp.max(jnp.abs(w*w+x*x+y*y+z*z-1))

def evaluate(seq,r,e,d,*,samples=32,seed=20261013,noise_scale=1.,fmin=1.,fmax=1e6,edge_steps=8,flat_steps=4,modes=96,coefficients=None,chunk=4096):
    e,d=np.broadcast_arrays(np.asarray(e,dtype=float),np.asarray(d,dtype=float));shape=e.shape
    steps,noise,coefficients=prepared(seq,r,samples,seed,fmin,fmax,noise_scale,edge_steps,flat_steps,modes,coefficients)
    ef,df=e.ravel(),d.ravel();out=[];norm_error=0.
    for i in range(0,len(ef),chunk):
        vals,err=evolve_jax(jnp.asarray(ef[i:i+chunk]),jnp.asarray(df[i:i+chunk]),jnp.asarray(steps),jnp.asarray(noise),theta=seq['theta'])
        out.append(np.asarray(vals));norm_error=max(norm_error,float(err))
    vals=np.concatenate(out,axis=1)
    return vals.reshape((len(noise[0]),)+shape),norm_error

def reference_matrix(seq,r,e,d,coefficients=None,fmax=1e6,noise_scale=1.):
    sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]],complex);sz=np.diag([1,-1])
    U=np.eye(2,dtype=complex)
    def eta(t):
        if coefficients is None:return 0.
        return float(noise_at([t],coefficients[None,:],fmax=fmax)[0,0]*noise_scale)
    for k,(p,a) in enumerate(zip(seq['phases'],seq['amplitudes'])):
        def rhs(t,y):
            H=(np.pi/2)*((1+e)*a*float(envelope(t,r))*(np.cos(p)*sx-np.sin(p)*sy)-(d+eta(k*(1+r)+t))*sz)
            return (-1j*H@y.reshape(2,2)).ravel()
        # Integrate each smooth interval separately to resolve the edge boundaries.
        edges=[0,r,1,1+r] if r else [0,1]
        for left,right in zip(edges[:-1],edges[1:]):
            result=solve_ivp(rhs,(left,right),U.ravel(),method='DOP853',rtol=2e-12,atol=2e-14)
            assert result.success;U=result.y[:,-1].reshape(2,2)
    c,s=np.cos(seq['theta']/2),np.sin(seq['theta']/2);target=c*np.eye(2)-1j*s*sx
    # Direct trace formula is sufficient for checks well above machine precision.
    I=(4-abs(np.trace(target.conj().T@U))**2)/6
    return float(I.real),float(np.max(abs(U.conj().T@U-np.eye(2))))
