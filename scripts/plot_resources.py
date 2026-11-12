"""Figure 9 from verified fixed-control metrics, including SCORBUTUS."""
from pathlib import Path
import csv,hashlib,json,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE=Path(__file__).resolve().parents[2]
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--metrics',type=Path,default=Path(__file__).resolve().parents[1]/'data/static_metrics.json')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/figures')
args=parser.parse_args();OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
data=json.loads(args.metrics.read_text());RENAME={}
FAMILIES={'SCORBUTUS':dict(color='#6a3d9a',marker='p'),
          'Jones':dict(color='#1f4e79',marker='D'),
          'ASBO':dict(color='#b2185b',marker='h'),
          'Reduced concatenated CP':dict(color='#8c6d31',marker='P')}
NAMES={'SCORBUTUS_X':'SCORBUTUS','SCORBUTUS_H':'SCORBUTUS','rCinSK_X':'rCinSK','rCinBB_X':'rCinBB','rCinSK_H':'rCinSK','rCinBB_H':'rCinBB',
       'rSKinsC_H':'rSKinsC','Jones7a':'7a','Jones7b':'7b','Jones7c':'7c',
       'JonesS9':'S9','ASBO9p':r'9$+$','ASBO9m':r'9$-$'}
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'cm',
                     'font.size':14,'axes.linewidth':.8,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(13.5,6.4),sharey=True);rows=[]
def point(target,source,family):
    z=data[source];name=RENAME.get(source,source)
    rows.append(dict(target=target,sequence=name,source_sequence=source,family=family,
                     N=z['N'],normalized_area=z['area_over_target'],mean_infidelity=z['small']['mean']))
    return z['area_over_target'],z['small']['mean'],name
for ax,target in zip(axes,['X','H']):
    numerical=[f'{target}{n}c' for n in [5,7,9,11]]+[f'{target}{n}{v}' for n in [13,15] for v in ['c','w']]
    color='#d95f02' if target=='X' else '#238443';marker='^' if target=='X' else 's'
    offsets={f'{target}13c':(-35,10),f'{target}15c':(5,-12),
             f'{target}13w':((7,11) if target=='X' else (6,-10)),
             f'{target}15w':((13,8) if target=='X' else (9,-12)),
             f'{target}11c':((-28,-16) if target=='X' else (12,6)),f'{target}7c':(-29,5),
             f'{target}9c':((-36,-14) if target=='X' else (5,8)),f'{target}5c':((-32,8) if target=='X' else (7,7))}
    for source in numerical:
        x,y,name=point(target,source,'Numerical w' if source.endswith('w') else 'Numerical c')
        is_width=name.endswith('w')
        ax.semilogy(x,y,marker,ms=7,color=color,mfc='white' if is_width else color)
        ax.annotate(name[1:],(x,y),xytext=offsets[source],textcoords='offset points',fontsize=11,color=color)
    if target=='X':
        for branch,col,m in [('a','#31688e','o'),('b','#7b3294','v')]:
            for N in [5,7,9,11,13]:
                name=f'X{N}{branch}';x,y,_=point(target,name,f'Symmetric Xn{branch}')
                ax.semilogy(x,y,m,color=col,ms=5)
                offset=(4,-12 if branch=='b' else 5)
                if name=='X5a':offset=(4,-12)
                if name=='X11b':offset=(-13,8)
                ax.annotate(str(N),(x,y),xytext=offset,textcoords='offset points',fontsize=9,color=col)
    refs=['SCORBUTUS_'+target,'rCinSK_'+target,'rCinBB_'+target]+(['Jones7a','Jones7b','Jones7c','JonesS9','ASBO9p','ASBO9m'] if target=='X' else ['rSKinsC_H'])
    positions={'SCORBUTUS_X':(1.2,.0019,'left'),'rCinSK_X':(8.6,.0033,'left'),'rCinBB_X':(8.2,.00078,'left'),
               'Jones7a':(6.65,.00072,'right'),'Jones7b':(6.65,.000235,'right'),
               'Jones7c':(6.65,.00125,'right'),'JonesS9':(9.35,.00014,'left'),
               'ASBO9p':(9.25,.00145,'left'),'ASBO9m':(9.25,.00045,'left')}
    h_offsets={'SCORBUTUS_H':(-88,4),'rCinSK_H':(-56,15),'rCinBB_H':(9,4),'rSKinsC_H':(-58,14)}
    for source in refs:
        family='SCORBUTUS' if source.startswith('SCORBUTUS') else 'Jones' if source.startswith('Jones') else 'ASBO' if source.startswith('ASBO') else 'Reduced concatenated CP'
        style=FAMILIES[family];x,y,_=point(target,source,family)
        ax.semilogy(x,y,style['marker'],ms=6.5,color=style['color'],mfc=style['color'],mew=1.)
        if source in positions:
            tx,ty,ha=positions[source]
            ax.text(tx,ty,NAMES[source],ha=ha,va='center',fontsize=10,color=style['color'])
        else:
            ax.annotate(NAMES[source],(x,y),xytext=h_offsets[source],textcoords='offset points',fontsize=10,color=style['color'])
    x,y,_=point(target,'Single_'+target,'Single rotation');ax.semilogy(x,y,'*',color='.5',ms=8)
    ax.annotate('Single',(x,y),xytext=(6,0),textcoords='offset points',fontsize=10,color='.35')
    xmax=max(data[n]['area_over_target'] for n in numerical+refs)
    ax.set(xlim=(0,xmax*1.20),ylim=(1e-7,.04),xlabel=r'Normalized pulse area $\mathcal{R}_A=\Phi/\theta$')
    ax.set_title('(a) $X$' if target=='X' else r'(b) $R_x(\pi/2)$',fontsize=16)
    ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel(r'Mean gate infidelity on $B_{0.15}$')
handles=[Line2D([0],[0],ls='none',marker='o',color='#31688e',label=r'Symmetric $X_{na}$'),
         Line2D([0],[0],ls='none',marker='v',color='#7b3294',label=r'Symmetric $X_{nb}$')]
handles += [Line2D([0],[0],ls='none',marker=s['marker'],color=s['color'],label=n) for n,s in FAMILIES.items()]
fig.legend(handles=handles,loc='upper center',ncol=6,frameon=False,fontsize=10,bbox_to_anchor=(.51,1.0),columnspacing=1.15)
fig.text(.5,.91,'c: filled markers; w: open markers',ha='center',fontsize=11)
fig.subplots_adjust(left=.08,right=.985,bottom=.14,top=.82,wspace=.12)
fig.savefig(OUT/'normalized_resources.pdf',bbox_inches='tight')
fig.savefig(OUT/'normalized_resources.png',dpi=190,bbox_inches='tight');plt.close(fig)
with (OUT/'figure9_data.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
assert len(rows)==41
print('Saved resource comparison with compact c/w and family labels.')
