"""Plot the constant-amplitude X-gate infidelity landscapes.

Run with --controls controls.json --output manuscript/Figures.
The supplied controls use radians, amplitude units Omega_0 and time units pi/Omega_0.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'src'))
from rectangular import infidelity

plt.rcParams.update({'font.family':'serif', 'font.serif':['DejaVu Serif'],
                     'mathtext.fontset':'cm', 'font.size':14,
                     'axes.linewidth':.8, 'pdf.fonttype':42})
LEVELS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
COLORS = ['#67001f', '#c51b7d', '#440154', '#31688e', '#238a8d', '#a06400']
STYLES = [':', '--', '-', '-', '-', '-']

def landscape(reg, names, filename, output, coarse=False):
    u = np.unique(np.r_[np.linspace(-.5,.5,801),
                        np.linspace(-.025,.025,101),np.linspace(-.001,.001,61)])
    D,E = np.meshgrid(u,u)
    fig, axes = plt.subplots(2,2,figsize=(7.4,7.8),sharex=True,sharey=True)
    levels, colors, styles = (LEVELS[2:],COLORS[2:],STYLES[2:]) if coarse else (LEVELS,COLORS,STYLES)
    for i,(ax,n) in enumerate(zip(axes.ravel(),names)):
        s = reg[n]
        assert np.allclose(s['amplitudes'],1), (n, s['amplitudes'])
        z = infidelity(s,E,D)
        for level,color,ls in zip(levels,colors,styles):
            if z.min()<level<z.max():
                cs = ax.contour(D,E,z,levels=[level],colors=[color],linestyles=[ls],
                                linewidths=1.45 if level==1e-4 else 1.0)
                if level>=1e-4 and (not coarse or level>=1e-2):
                    vertices = [p.vertices for p in cs.get_paths() if len(p.vertices)]
                    if vertices:
                        v = np.concatenate(vertices)
                        if np.ptp(v,axis=0).max()>.15:
                            safe = v[np.max(abs(v),axis=1)<.43]
                            if len(safe):
                                pref = np.array([-.17,.13]) if level==1e-4 else np.array([.20,.28]) if level==1e-3 else np.array([-.22,-.25]) if level==1e-2 else np.array([.19,.39])
                                pos = safe[np.argmin(np.sum((safe-pref)**2,axis=1))]
                                ax.clabel(cs,fmt={level:rf'$10^{{{int(np.log10(level))}}}$'},
                                          fontsize=11,manual=[pos],inline_spacing=2)
        name = 'Single X' if n=='Single_X' else n
        ax.set_title(f'({chr(97+i)}) '+name,fontsize=16,pad=7)
        ax.set(xlim=(-.5,.5),ylim=(-.5,.5)); ax.set_aspect('equal')
        ax.set_xticks([-.5,-.25,0,.25,.5]);ax.set_yticks([-.4,-.2,0,.2,.4]);ax.tick_params(labelsize=12)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5));ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        if i>=2: ax.set_xlabel(r'Detuning error $\delta$',fontsize=15)
        if i%2==0: ax.set_ylabel(r'Amplitude error $\epsilon$',fontsize=15)
    handles = [Line2D([0],[0],color=c,ls=ls,lw=1.4,label=rf'$10^{{{int(np.log10(v))}}}$')
               for v,c,ls in zip(levels,colors,styles)]
    fig.legend(handles=handles,loc='lower center',ncol=len(levels),frameon=False,fontsize=12,
               bbox_to_anchor=(.52,.005),title='Gate infidelity',title_fontsize=12)
    fig.subplots_adjust(left=.08,right=.99,top=.94,bottom=.17,wspace=.15,hspace=.22)
    fig.savefig(output/(filename+'.pdf'),bbox_inches='tight')
    fig.savefig(output/(filename+'.png'),dpi=185,bbox_inches='tight')
    plt.close(fig)
    print('Generated',filename,flush=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controls',type=Path,default=BASE/'data/controls.json')
    parser.add_argument('--output',type=Path,default=BASE/'outputs/figures')
    args = parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    reg = json.loads(args.controls.read_text())['sequences']
    groups = {
        'combined3_Nikolay':['Single_X','CORPSE','B3r','B3d'],
        'combined5_Nikolay':['B5','BB1','U5a','U5b'],
        'combined7_Nikolay':['U7a','U7b','X7a','X7b'],
        'combined9_Nikolay':['U9a','U9b','X9a','X9b'],
        'combined11_Nikolay':['U11a','U11b','X11a','X11b'],
    }
    for filename,names in groups.items():
        landscape(reg,names,filename,args.output,coarse=filename=='combined3_Nikolay')
