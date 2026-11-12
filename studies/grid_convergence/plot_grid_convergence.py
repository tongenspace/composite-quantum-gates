"""Draw the fixed-control evaluation-grid check without a TeX dependency."""
from pathlib import Path
import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'outputs/grid_convergence')
    source = parser.parse_args().input
    with (source / 'evaluation_grid_convergence.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout='constrained')
    specs = [('X5c', '#315b99', 'o'), ('H5c', '#c16d28', 's'),
             ('X15c', '#3d8568', '^'), ('H15c', '#8c5497', 'd')]
    for name, color, marker in specs:
        series = [row for row in rows if row['name'] == name]
        axes[0].plot([int(row['m']) for row in series],
                     [float(row['mean']) for row in series],
                     color=color, marker=marker, ms=4, lw=1.2, label=name)
        series = [row for row in series if int(row['m']) < 3201]
        axes[1].plot([int(row['m']) for row in series],
                     [float(row['absolute_relative_difference_percent']) for row in series],
                     color=color, marker=marker, ms=4, lw=1.2,
                     linestyle='--' if name == 'H5c' else '-')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlim(7, 3800)
        ax.set_xticks([8, 32, 128, 512, 3201], ['8', '32', '128', '512', '3201'])
        ax.set_xlabel(r'Evaluation points per axis, $m$')
        ax.grid(alpha=.16)
    axes[0].set_ylim(1e-7, 5e-3)
    axes[0].set_title(r'(a) Mean infidelity on $B_{0.15}$', fontsize=11)
    axes[0].set_ylabel(r'$\overline{\mathcal{I}}_{0.15}^{(m)}$')
    axes[0].legend(loc='center right', frameon=False)
    axes[1].set_ylim(1e-5, 100)
    axes[1].set_title(r'(b) Deviation from $3201\times3201$', fontsize=11)
    axes[1].set_ylabel('Relative difference (%)')
    axes[1].axvline(801, ls='--', color='.6', lw=.8)
    axes[1].text(700, 70, r'Manuscript: $m=801$', ha='right', va='top', fontsize=9)
    fig.savefig(source / 'evaluation_grid_convergence.png', dpi=200)
    fig.savefig(source / 'evaluation_grid_convergence.svg')
    plt.close(fig)
    print('Grid-convergence figures:', source)

if __name__ == '__main__':
    main()
