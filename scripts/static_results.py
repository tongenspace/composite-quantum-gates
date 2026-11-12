"""Recompute the two comparison tables from the supplied pulse coefficients."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from scipy.ndimage import label
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rectangular import infidelity,matrix_infidelity

def measure(seq,n,r):
    u=(np.arange(n)+.5)*2*r/n-r
    det,eps=np.meshgrid(u,u)
    z=infidelity(seq,eps,max(seq['amplitudes'])*det)
    good=z<=1e-4;components,_=label(good);center=components[n//2,n//2]
    return dict(mean=float(z.mean()),maximum=float(z.max()),fraction=float(good.mean()),
                connected_fraction=float(np.mean(components==center)) if center else 0.)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--grid',type=int,default=801)
    ap.add_argument('--only',nargs='+');ap.add_argument('--output',type=Path,default=ROOT/'outputs/tables')
    a=ap.parse_args();assert a.grid%2==1
    reg=json.loads((ROOT/'data/controls.json').read_text())
    frozen=json.loads((ROOT/'data/static_metrics.json').read_text())
    names=list(dict.fromkeys(reg['X_table']+reg['H_table']))
    if a.only:names=[n for n in names if n in a.only]
    assert names
    result={}
    for name in names:
        seq=reg['sequences'][name]
        small=measure(seq,a.grid,.15);wide=measure(seq,a.grid,.5)
        u=np.linspace(-.15,.15,1601)
        e=np.r_[u,u,np.full(len(u),-.15),np.full(len(u),.15)]
        d=np.r_[np.full(len(u),-.15),np.full(len(u),.15),u,u]
        small['maximum']=max(small['maximum'],float(infidelity(seq,e,max(seq['amplitudes'])*d).max()))
        result[name]=dict(N=seq['N'],area_over_target=float(np.pi*np.dot(seq['amplitudes'],seq['durations_T'])/seq['theta']),
                          nominal=float(infidelity(seq,0,0)),small=small,wide=wide)
        if a.grid==801:
            old=frozen[name]
            assert abs(small['mean']-old['small']['mean'])<1e-13,name
            assert abs(wide['connected_fraction']-old['wide']['connected_fraction'])<1e-12,name
        print(name,'mean',small['mean'],'connected B0.5',wide['connected_fraction'],flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'recomputed_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    for target,ids in [('X',reg['X_table']),('H',reg['H_table'])]:
        rows=[]
        for name in ids:
            if name not in result:continue
            s=result[name]
            rows.append(dict(sequence=name,N=s['N'],normalized_area=s['area_over_target'],
                             mean_infidelity=s['small']['mean'],sampled_maximum=s['small']['maximum'],
                             coverage_B015_percent=100*s['small']['fraction'],
                             connected_coverage_B05_percent=100*s['wide']['connected_fraction']))
        if rows:
            with (a.output/f'table_{target}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
