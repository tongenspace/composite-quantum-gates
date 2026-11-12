"""Independent selection, held-out validation and depth/width comparisons."""
from pathlib import Path
import argparse,json,sys,csv
import numpy as np
from scipy.ndimage import label
from scipy.stats import qmc
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from rectangular import infidelity,matrix,SX,metrics

def measure(s,n=801,r=.5):
    u=(np.arange(n)+.5)*2*r/n-r;D,E=np.meshgrid(u,u)
    z=infidelity(s,E,max(s['amplitudes'])*D);mask=z<=1e-4
    labels,_=label(mask);center=labels[n//2,n//2]
    connected=float(np.mean(labels==center)) if center else 0.
    bad=np.maximum(abs(E),abs(D))[~mask]
    rho=float(max(0,bad.min()-r/n)) if len(bad) else r
    out=metrics(s,n=n,r=r,peak=True)
    out.update(central_connected_fraction=connected,sampled_inscribed_square_half_width=rho,sampled_maximum=float(z.max()))
    return out

def validate(s,old):
    assert s['N']==old['N']==15
    assert sum(s['amplitudes'])<=sum(old['amplitudes'])+1e-8
    assert max(s['amplitudes'])<=max(old['amplitudes'])+1e-9
    assert min(s['amplitudes'])>=.05-1e-9
    assert infidelity(s,0,0)<1e-12
    printed=dict(s,amplitudes=np.round(s['amplitudes'],8).tolist(),phases=(np.round(np.array(s['phases'])/np.pi,8)*np.pi).tolist())
    assert infidelity(printed,0,0)<1e-12
    axis=np.linspace(-.20,.20,501);D,E=np.meshgrid(axis,axis)
    maxcentral=float(infidelity(s,E,max(s['amplitudes'])*D).max())
    assert maxcentral<1e-4
    G=expm(-.5j*s['theta']*SX);rng=np.random.default_rng(202610212)
    matrixerr=unitarity=peakerr=0.
    peak=max(s['amplitudes']);scaled=dict(s,amplitudes=(np.array(s['amplitudes'])/peak).tolist(),durations_T=(np.array(s['durations_T'])*peak).tolist())
    for e,d in rng.uniform(-.5,.5,(96,2)):
        U=matrix(s,e,d*peak);I=(4-abs(np.trace(G.conj().T@U))**2)/6
        matrixerr=max(matrixerr,abs(I-float(infidelity(s,e,d*peak))))
        unitarity=max(unitarity,np.max(abs(U.conj().T@U-np.eye(2))))
        peakerr=max(peakerr,np.max(abs(U-matrix(scaled,e,d))))
    assert max(matrixerr,unitarity,peakerr)<1e-13
    heldout=[]
    for seed in [202610213,202610214]:
        E,D=(qmc.Sobol(2,scramble=True,seed=seed).random_base2(17)-.5).T
        z=infidelity(s,E,peak*D)
        heldout.append(dict(seed=seed,samples=len(z),wide_fraction_below_1e4=float(np.mean(z<=1e-4)),wide_mean=float(z.mean())))
    return dict(nominal=float(infidelity(s,0,0)),printed_nominal=float(infidelity(printed,0,0)),
                protected_half_width=.20,protected_window_grid_maximum=maxcentral,matrix_max_difference=float(matrixerr),unitarity_error=float(unitarity),
                peak_rescaling_error=float(peakerr),held_out=heldout,area_cap=sum(old['amplitudes']),peak_cap=max(old['amplitudes']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    registry=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences'];selected={}
    for name,oldname in [('X15w','X15c'),('H15w','H15')]:
        rows=json.loads((BASE/(name.lower()+'_bounded_search.json')).read_text())
        candidates=[]
        for i,row in enumerate(rows):
            if not row['success'] or row['grid']<61 or row['small_metrics']['center']>1e-12:continue
            s=row['controls'];m=measure(s,n=401)
            candidates.append((m['central_connected_fraction'],i,row))
        candidates.sort(reverse=True,key=lambda v:v[0])
        chosen=None
        for score,i,row in candidates:
            try:checks=validate(row['controls'],registry[oldname])
            except AssertionError:continue
            chosen=(i,row,checks);break
        if chosen is None:raise RuntimeError('No independently feasible candidate for '+name)
        i,row,checks=chosen;s=row['controls']
        m=measure(s);coarse=measure(s,n=401);finer=measure(s,n=1601)
        assert abs(m['central_connected_fraction']-finer['central_connected_fraction'])<.001
        original=measure(registry[oldname]);assert m['central_connected_fraction']>original['central_connected_fraction']
        selected[name]=dict(controls=s,original_name=oldname,search_row=i,selection='Largest central connected 1e-4 area among successful final-grid fits satisfying original-window and resource checks',
             validation=checks,wide_metrics=m,wide_metrics_401=coarse,wide_metrics_1601=finer,
             small_metrics=metrics(s,n=801,peak=True),original_wide_metrics=original,original_small_metrics=metrics(registry[oldname],n=801,peak=True))
        print(name,'connected',original['central_connected_fraction'],'->',m['central_connected_fraction'],
              'rho',original['sampled_inscribed_square_half_width'],'->',m['sampled_inscribed_square_half_width'],flush=True)
    if args.verify:
        old=json.loads((BASE/'selected_controls.json').read_text())
        for name,r in selected.items():
            assert r['controls']==old[name]['controls']
        print('Depth/width selection, grids, held-out points, matrices, printed precision and matched resource limits verified.',flush=True)
    else:
        (BASE/'selected_controls.json').write_text(json.dumps(selected,indent=2))
        (ROOT/'data/width_controls.json').write_text(json.dumps({n:r['controls'] for n,r in selected.items()},indent=2))
if __name__=='__main__':main()
