"""Four-panel appendix figure from the six-sequence edge/noise calculations."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = Path(__import__('os').environ.get('CP_FIGURE_OUTPUT',str(HERE.parents[1]/'outputs/figures')))
STYLES = {
    'X9a': ('#1f4e79', '-', '^'),
    'X13c': ('#7b3294', '-', 'o'),
    'X13w': ('#7b3294', '--', 's'),
    'H9c': ('#238443', '-', '^'),
    'H13c': ('#d95f02', '-', 'o'),
    'H13w': ('#d95f02', '--', 's'),
}


def main():
    data = json.loads((HERE/'edge_results.json').read_text())
    assert set(data) == set(STYLES)
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'cm', 'font.size':12,
                         'axes.linewidth':.8, 'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0))
    envelope, edge_ax, cutoff_ax, combined_ax = axes.ravel()
    t = np.linspace(-4, 109, 1601)
    f = np.where(t<0, 0., np.where(t<4, .5*(1-np.cos(np.pi*t/4)),
                 np.where(t<=100, 1., np.where(t<=104, .5*(1+np.cos(np.pi*(t-100)/4)), 0.))))
    envelope.plot(t, np.where((t>=0)&(t<=100), 1., 0.), '--', color='.5', lw=1.3, label='Rectangular')
    envelope.plot(t, f, color='#31688e', lw=1.8, label='Cosine edges')
    envelope.set(xlim=(-4, 109), ylim=(-.05, 1.32), xlabel=r'Local time $s$ (ns)',
                 ylabel=r'Envelope $f(s)$', title='(a) Area-preserving pulse edges')
    envelope.legend(loc='upper center', frameon=False, ncol=2, fontsize=10)
    envelope.text(50, .48, r'$T_s=100$ ns; $\tau_r=4$ ns', ha='center', fontsize=12)
    handles, rows = [], []
    for index, (name, (col, ls, marker)) in enumerate(STYLES.items()):
        result = data[name]
        edge = result['edges_only']
        edge_ax.semilogy([r['edge_ns'] for r in edge], [r['mean'] for r in edge],
                         color=col, ls=ls, marker=marker, ms=4.5, lw=1.5)
        cutoff = result['nominal_cutoffs']
        cutoff_ax.errorbar([r['fmax_Hz'] for r in cutoff], [r['mean'] for r in cutoff],
                           yerr=[r['standard_error'] for r in cutoff], color=col, ls=ls,
                           marker=marker, ms=4.5, lw=1.5, capsize=2)
        at4 = next(r for r in edge if r['edge_ns']==4)
        conditions = [('Rectangular', result['rectangular_mean'], 0., 'o', -.23),
                      ('4 ns edges', at4['mean'], 0., 's', 0.),
                      ('Edges + noise', result['combined']['mean'], result['combined']['standard_error'], '^', .23)]
        for condition, mean, se, condition_marker, offset in conditions:
            combined_ax.errorbar(index+offset, mean, yerr=se, color=col, marker=condition_marker,
                                 mfc='white' if condition_marker=='s' else col, ms=6, capsize=2,
                                 ls='none', zorder=3)
            rows.append(dict(sequence=name, condition=condition, domain_mean_infidelity=mean,
                              Monte_Carlo_standard_error=se))
        handles.append(Line2D([0],[0], color=col, ls=ls, marker=marker, ms=4.5, label=name))
    edge_ax.set(xlabel=r'Rise/fall duration $\tau_r$ (ns)', ylabel=r'Mean infidelity on $B_{0.15}$',
                title='(b) Finite-edge sensitivity')
    edge_ax.set_xticks([0,2,4,8])
    edge_ax.grid(axis='y', alpha=.17)
    combined_ax.set(yscale='log', ylabel=r'Mean infidelity on $B_{0.15}$',
                    title='(d) Combined finite-domain performance', xlim=(-.55,5.55))
    combined_ax.set_xticks(range(6), list(STYLES))
    combined_ax.tick_params(axis='x', labelsize=10)
    for tick, (col, _, _) in zip(combined_ax.get_xticklabels(), STYLES.values()):
        tick.set_color(col)
    upper = max(r['domain_mean_infidelity'] for r in rows)*8
    lower = min(r['domain_mean_infidelity'] for r in rows)/2
    combined_ax.set_ylim(lower, upper)
    condition_handles = [Line2D([0],[0], color='.25', marker=m, mfc='white' if m=='s' else '.25',
                                ls='none', label=label, ms=5)
                         for label, m in [('Rectangular','o'), ('4 ns edges','s'), ('Edges + noise','^')]]
    combined_ax.legend(handles=condition_handles, loc='upper center', ncol=3, frameon=False,
                        fontsize=9, columnspacing=.8, handletextpad=.3)
    combined_ax.grid(axis='y', alpha=.17)
    cutoff_ax.set(xscale='log', yscale='log', xlabel=r'Upper noise cutoff $f_{\max}$ (Hz)',
                  ylabel='Nominal mean infidelity', title='(c) Detuning-noise bandwidth')
    cutoff_ax.set_xticks([1e4,1e5,1e6,1e7])
    cutoff_ax.grid(axis='y', alpha=.17)
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=11,
                bbox_to_anchor=(.52,.008), columnspacing=1.3)
    fig.subplots_adjust(left=.08, right=.985, top=.94, bottom=.12, hspace=.48, wspace=.30)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'pulse_shape_sensitivity.pdf')
    fig.savefig(OUT/'pulse_shape_sensitivity.png', dpi=220)
    with (HERE/'combined_panel_data.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Generated four-panel figure for', ', '.join(STYLES))


if __name__ == '__main__':
    main()
