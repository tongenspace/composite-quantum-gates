"""All-branch literature benchmark, held-out metrics and publication figures."""
from pathlib import Path
import json,sys,csv,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'studies/objective_selection')]
from rectangular import infidelity,metrics
from concatenated import baseline_controls
from evaluate import measure
from plot_paper import configure,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--no-tex',action='store_true');args=p.parse_args()
    lit=json.loads((BASE/'reconstructed_controls.json').read_text())
    reg=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
    width=json.loads((ROOT/'data/width_controls.json').read_text())
    controls={n:r['controls'] for n,r in lit.items()}|baseline_controls()|{n:reg[n] for n in ['X9a','X9b','X9c','X11c','X13c','X15c','H8','H10','H15']}|width
    chosen={}
    for target in ['X','H']:
        for family in ['AP3','PD4','AP5','PD6']:
            families=[family]+(['AP4'] if family=='PD4' else ['AP6'] if family=='PD6' else [])
            options=[n for n in lit if any(n.startswith('Cin'+f+'_') for f in families) and n.endswith('_'+target)]
            chosen[family+'_'+target]=min(options,key=lambda n:lit[n]['small_metrics']['mean'])
    results=[]
    for name,s in controls.items():
        row=metrics(s,n=801,peak=True);coarse=metrics(s,n=401,peak=True)
        wide=measure(s,n=801);wc=measure(s,n=401)
        u=np.linspace(-.15,.15,401);D,E=np.meshgrid(u,u)
        row.update(target='X' if s['theta']>2 else 'H',sampled_original_window_max=float(infidelity(s,E,max(s['amplitudes'])*D).max()),
            wide_fraction=wide['fraction_below_1e4'],wide_connected_fraction=wide['central_connected_fraction'],
            inscribed_half_width=wide['sampled_inscribed_square_half_width'],
            grid_mean_relative_change=abs(row['mean']-coarse['mean'])/row['mean'],
            grid_fraction_change=abs(wide['fraction_below_1e4']-wc['fraction_below_1e4']))
        assert row['grid_mean_relative_change']<.001
        assert row['grid_fraction_change']<.003
        results.append(row)
    record=dict(source='Low et al. PRA89,022341 (2014), Eqs.7/11/12 and Table III; Jones PRA87,052317 (2013), Eqs.62-65',
        branch_selection='For each amplitude order 3,4,5,6 and target, lowest common-peak mean over all published AP/PD branches at that order on independent 401 midpoint grid; all branches supplied',
        chosen=chosen,controls=controls,records=results,grid=801,common_peak=True,small_half_width=.15,wide_half_width=.5,threshold=1e-4)
    path=BASE/'comparison.json'
    if args.verify:
        old=json.loads(path.read_text())
        assert old['controls']==controls and old['chosen']==chosen
        for a,b in zip(old['records'],results):
            for k in ['mean','wide_fraction','wide_connected_fraction','inscribed_half_width']:
                np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-15)
        print('All higher-order and depth/width comparison metrics revalidated.',flush=True);return
    path.write_text(json.dumps(record,indent=2))
    with (BASE/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    configure(not args.no_tex)
    out=ROOT/'figures/generated';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(10.6,8.2))
    eps=np.geomspace(.005,.5,501)
    for ax,target in zip(axes[0],['X','H']):
        names=['rCinBB_'+target]+[chosen[f+'_'+target] for f in ['AP3','PD4','AP5','PD6']]
        for name,color,style in zip(names,['.35','#31688e','#7b3294','#238a8d','#d95f02'],['-','--','-.',':','-']):
            s=controls[name];label='rCinBB' if name.startswith('rCinBB') else name.split('_')[0][3:]+', branch '+name.split('_')[1]
            ax.loglog(eps,np.maximum(infidelity(s,eps,0),1e-29),style,color=color,lw=1.5,label=label)
        ax.set(xlim=(.005,.5),ylim=(1e-24,.1),xlabel=r'Amplitude error $\epsilon$',ylabel=r'Gate infidelity, $\delta_c=0$')
        ax.set_title(('(a) $X$' if target=='X' else r'(b) $R_x(\pi/2)$'),fontsize=17)
        ax.legend(fontsize=9,loc='lower right',frameon=False);ax.tick_params(labelsize=12);ax.grid(alpha=.15)
    u=np.linspace(-.5,.5,801);D,E=np.meshgrid(u,u)
    for ax,target,letter in zip(axes[1],['X','H'],'cd'):
        deep='X15c' if target=='X' else 'H15';wide='X15w' if target=='X' else 'H15w'
        higher='JonesS9' if target=='X' else chosen['AP3_H']
        for name,color,style in [(deep,'#31688e','-'),(wide,'#d95f02','-'),('rCinBB_'+target,'.35','--'),(higher,'#7b3294','-.')]:
            s=controls[name];z=infidelity(s,E,max(s['amplitudes'])*D)
            ax.contour(D,E,z,levels=[1e-4],colors=[color],linestyles=[style],linewidths=1.5)
            ax.plot([],[],color=color,ls=style,label=name.replace('_'+target,'').replace('_',' '))
        ax.plot([-.15,.15,.15,-.15,-.15],[-.15,-.15,.15,.15,-.15],':',color='.65',lw=1)
        ax.set(xlim=(-.5,.5),ylim=(-.5,.5),xlabel=r'Detuning error $\delta_c$',ylabel=r'Amplitude error $\epsilon$')
        ax.set_title(rf'({letter}) $10^{{-4}}$ contours',fontsize=17);ax.tick_params(labelsize=12)
        ax.legend(fontsize=9,loc='upper right',frameon=True,framealpha=.9)
    fig.tight_layout(pad=1.2);save(fig,out,'higher_order_objectives')
    print('Generated higher_order_objectives and all-branch metrics.',flush=True)
if __name__=='__main__':main()
