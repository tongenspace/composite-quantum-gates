from pathlib import Path
import argparse,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rectangular import infidelity
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'outputs/figures')
args=ap.parse_args();FIG=args.output;FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'cm','font.size':14,'axes.linewidth':.8,'pdf.fonttype':42})
LEVELS=[1e-8,1e-6,1e-4,1e-3,1e-2,1e-1]
COLORS=['#67001f','#c51b7d','#440154','#31688e','#238a8d','#a06400']
STYLES=[':', '--','-','-','-','-']
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=185,bbox_inches='tight');plt.close(fig)

def label(n):
    return {'Single_X':'Single X','Single_H':r'Single $R_x(\pi/2)$','B3r':'B3r','B3d':'B3d','B5':'B5','rCinSK_H':'rCinSK','rCinBB_H':'rCinBB','rCinSK_X':'rCinSK','rCinBB_X':'rCinBB','JonesS9':'Jones S9','ASBO9m':r'ASBO-9$-$','CinAP3_2_H':r'$\mathrm{CinAP3}_2$'}.get(n,n)

def landscapes(reg,names,name):
    cols=4 if len(names)==8 else 2;rows=2
    # Extra central samples resolve small inner contours without a huge full mesh.
    u=np.unique(np.r_[np.linspace(-.5,.5,801),np.linspace(-.025,.025,101),np.linspace(-.001,.001,61)])
    D,E=np.meshgrid(u,u);fig,axes=plt.subplots(rows,cols,figsize=(3.7*cols,7.8),sharex=True,sharey=True)
    for i,(ax,n) in enumerate(zip(axes.ravel(),names)):
        s=reg[n];z=infidelity(s,E,max(s['amplitudes'])*D)
        for level,color,ls in zip(LEVELS,COLORS,STYLES):
            if z.min()<level<z.max():
                cs=ax.contour(D,E,z,levels=[level],colors=[color],linestyles=[ls],linewidths=1.45 if level==1e-4 else 1.0)
                if level>=1e-4:
                    vertices=[p.vertices for p in cs.get_paths() if len(p.vertices)]
                    if vertices:
                        v=np.concatenate(vertices)
                        if np.ptp(v,axis=0).max()>.15:
                            safe=v[np.max(abs(v),axis=1)<.43]
                            if len(safe):
                                pref=np.array([-.17,.13]) if level==1e-4 else np.array([.20,.28]) if level==1e-3 else np.array([-.22,-.25]) if level==1e-2 else np.array([.19,.39])
                                pos=safe[np.argmin(np.sum((safe-pref)**2,axis=1))]
                                ax.clabel(cs,fmt={level:rf'$10^{{{int(np.log10(level))}}}$'},fontsize=11,manual=[pos],inline_spacing=2)
        ax.set_title(f'({chr(97+i)}) '+label(n),fontsize=16,pad=7);ax.set(xlim=(-.5,.5),ylim=(-.5,.5));ax.set_aspect('equal')
        ax.set_xticks([-.5,-.25,0,.25,.5]);ax.set_yticks([-.4,-.2,0,.2,.4]);ax.tick_params(labelsize=12)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5));ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        if i>=cols:ax.set_xlabel(r'Detuning error $\delta_c$',fontsize=15)
        if i%cols==0:ax.set_ylabel(r'Amplitude error $\epsilon$',fontsize=15)
    handles=[Line2D([0],[0],color=c,ls=ls,lw=1.4,label=rf'$10^{{{int(np.log10(v))}}}$') for v,c,ls in zip(LEVELS,COLORS,STYLES)]
    fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False,fontsize=12,bbox_to_anchor=(.52,.005),title='Gate infidelity',title_fontsize=12)
    fig.subplots_adjust(left=.08,right=.99,top=.94,bottom=.17,wspace=.15,hspace=.22)
    save(fig,name);print('Generated',name,flush=True)

def width_figure(reg,data):
    fig,axes=plt.subplots(1,2,figsize=(12.6,5.8),sharex=True,sharey=True);u=np.linspace(-.5,.5,1001);D,E=np.meshgrid(u,u)
    for ax,target in zip(axes,['X','H']):
        refs=[(target+'15c','#31688e','-'),(target+'15w','#d95f02','-')]
        if target=='X':refs += [('ASBO9m','.35','--'),('JonesS9','#7b3294','-.')]
        else:refs += [('rCinBB_H','.35','--'),('CinAP3_2_H','#7b3294','-.')]
        handles=[]
        for n,col,ls in refs:
            s=reg[n];ax.contour(D,E,infidelity(s,E,max(s['amplitudes'])*D),levels=[1e-4],colors=[col],linestyles=[ls],linewidths=1.4)
            handles.append(Line2D([0],[0],color=col,ls=ls,label=label(n)+rf', $\mathcal{{R}}_A={data[n]["area_over_target"]:.2f}$'))
        ax.plot([-.15,.15,.15,-.15,-.15],[-.15,-.15,.15,.15,-.15],':',color='.65',lw=.8)
        ax.set(xlim=(-.5,.5),ylim=(-.5,.5),xlabel=r'Detuning error $\delta_c$');ax.set_aspect('equal')
        ax.set_title('(a) $X$' if target=='X' else r'(b) $R_x(\pi/2)$',fontsize=16)
        ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.10),ncol=2,frameon=False,fontsize=10)
    axes[0].set_ylabel(r'Amplitude error $\epsilon$');fig.subplots_adjust(left=.08,right=.99,bottom=.12,top=.73,wspace=.14)
    save(fig,'higher_order_objectives')
reg=json.loads((ROOT/'data/controls.json').read_text())['sequences']
metrics=json.loads((ROOT/'data/static_metrics.json').read_text())
landscapes(reg,['X5c','X7c','X9c','X11c','X13c','X13w','X15c','X15w'],'combinednonsym')
landscapes(reg,['H5c','H7c','H9c','H11c'],'combinednonsymH1')
landscapes(reg,['H13c','H13w','H15c','H15w'],'combinednonsymH2')
width_figure(reg,metrics)
