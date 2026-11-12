from pathlib import Path
import json,numpy as np
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
registry=json.loads((ROOT/'data/controls.json').read_text())['sequences']
def qfun(seq,v):
    e,d=v;w,x,y,z=1.,0.,0.,0.
    for phi,a,t in zip(seq['phases'],seq['amplitudes'],seq['durations_T']):
        om=a*(1+e);q=jnp.sqrt(om*om+d*d);c=jnp.cos(jnp.pi*t*q/2);k=jnp.pi*t/2*jnp.sinc(t*q/2)
        px=k*om*jnp.cos(phi);py=-k*om*jnp.sin(phi);pz=-k*d
        w,x,y,z=c*w-px*x-py*y-pz*z,c*x+w*px+py*z-pz*y,c*y+w*py+pz*x-px*z,c*z+w*pz+px*y-py*x
    return jnp.array([w,x,y,z])
results=[]
for name in ['X5a','X5b','X7a','X7b','X9a','X9b','X11a','X11b','X13a','X13b']:
    s=registry[name];f=lambda v:qfun(s,v)
    d1=np.asarray(jax.jacfwd(f)(jnp.zeros(2)));d2=np.asarray(jax.jacfwd(jax.jacfwd(f))(jnp.zeros(2)))
    d3=np.asarray(jax.jacfwd(jax.jacfwd(jax.jacfwd(f)))(jnp.zeros(2)))
    row=dict(name=name,derivatives={str(key):float(np.linalg.norm((d1 if len(key)==1 else d2 if len(key)==2 else d3)[(slice(None),)+key])) for key in [(0,),(1,),(0,1),(0,0),(1,1),(0,0,1),(0,1,1)]})
    results.append(row);print(row,flush=True)
(ROOT/'outputs/checks').mkdir(parents=True,exist_ok=True)
(ROOT/'outputs/checks/derivative_audit.json').write_text(json.dumps(results,indent=2))

expected={'X5a':[(0,),(1,),(0,1)],'X5b':[(0,),(1,),(0,1)],
          'X7a':[(0,),(1,),(0,1),(0,0)],'X7b':[(0,),(1,),(0,1),(1,1)],
          'X9a':[(0,),(1,),(0,1),(0,0),(1,1)],'X9b':[(0,),(1,),(0,1),(0,0),(1,1)],
          'X11a':[(0,),(1,),(0,1),(0,0),(1,1),(0,0,1)],
          'X11b':[(0,),(1,),(0,1),(0,0),(1,1),(0,1,1)],
          'X13a':[(0,),(1,),(0,1),(0,0),(1,1),(0,0,1),(0,1,1)],
          'X13b':[(0,),(1,),(0,1),(0,0),(1,1),(0,0,1),(0,1,1)]}
for rec in results:
    tol=1e-10 if int(rec['name'][1:-1])<=9 else 1e-2
    for key in expected[rec['name']]:assert rec['derivatives'][str(key)]<tol
# Generic five-pulse expressions independently agree with direct differentiation.
for p1,p2 in [(.31,.87),(-.72,1.21),(2.09,-.52)]:
    s=dict(phases=[p1,p2,2*p2-2*p1,p2,p1],amplitudes=[1.]*5,durations_T=[1.]*5)
    D=np.asarray(jax.jacfwd(lambda v:qfun(s,v))(jnp.zeros(2)))
    got=D[0]-1j*D[3]
    want=np.array([-np.pi/2*(1+2*np.cos(p1)+2*np.cos(2*p1-p2)),
                   1j*(1+2*np.cos(p1)-2*np.cos(2*p1-p2))])
    np.testing.assert_allclose(got,want,atol=2e-14)
print('Derivative conditions and five-pulse expressions checked; finite-precision residuals are recorded.')
