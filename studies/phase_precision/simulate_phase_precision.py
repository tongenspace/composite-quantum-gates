"""Reproduce Appendix A: rectangular pulses with phase offsets or phase rounding.

Run with Python 3, NumPy, SciPy, Matplotlib. No optimization is performed.
The input contains only the four full-precision sequences used in Appendix A.
Detuning arguments below use Delta/Omega_0; plotted squares use Delta/Omega_max.
"""
from pathlib import Path
import csv
import json
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rectangular import infidelity

HERE = Path(__file__).resolve().parent
FIGURES = Path(__import__('os').environ.get('CP_FIGURE_OUTPUT',str(HERE.parents[1]/'outputs/figures')))
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.diag([1, -1]).astype(complex)
PAULI = np.array([SX, SY, SZ])


def target_rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def averaged_transfer(seq, eps, det, sigma):
    eps, det = np.broadcast_arrays(np.asarray(eps, float), np.asarray(det, float))
    result = np.broadcast_to(np.eye(3), eps.shape + (3, 3)).copy()
    first = np.exp(-sigma**2 / 2)
    second = np.exp(-2 * sigma**2)
    for phi, amp, duration in zip(seq['phases'], seq['amplitudes'], seq['durations_T']):
        r = amp * (1 + eps)
        q = np.sqrt(r*r + det*det)
        u = np.divide(r, q, out=np.zeros_like(q), where=q != 0)
        v = np.divide(-det, q, out=np.zeros_like(q), where=q != 0)
        angle = np.pi * duration * q
        c, s = np.cos(angle), np.sin(angle)
        nn = np.empty(eps.shape + (3, 3))
        nn[..., 0, 0] = u*u*(1 + second*np.cos(2*phi))/2
        nn[..., 1, 1] = u*u*(1 - second*np.cos(2*phi))/2
        nn[..., 0, 1] = nn[..., 1, 0] = -u*u*second*np.sin(2*phi)/2
        nn[..., 0, 2] = nn[..., 2, 0] = u*v*first*np.cos(phi)
        nn[..., 1, 2] = nn[..., 2, 1] = -u*v*first*np.sin(phi)
        nn[..., 2, 2] = v*v
        cross = np.zeros_like(nn)
        cross[..., 0, 1] = -v
        cross[..., 1, 0] = v
        cross[..., 0, 2] = -u*first*np.sin(phi)
        cross[..., 2, 0] = u*first*np.sin(phi)
        cross[..., 1, 2] = -u*first*np.cos(phi)
        cross[..., 2, 1] = u*first*np.cos(phi)
        rot = c[..., None, None]*np.eye(3) + (1-c)[..., None, None]*nn + s[..., None, None]*cross
        result = rot @ result
    return result


def expected_infidelity(seq, eps, det, sigma):
    if sigma == 0:
        return infidelity(seq, eps, det)
    transfer = averaged_transfer(seq, eps, det, sigma)
    overlap = np.einsum('ij,...ij->...', target_rotation(seq['theta']), transfer)
    return np.clip((3-overlap)/6, 0, 2/3)


def grid_mean(seq, sigma=0., n=201):
    u = (np.arange(n)+.5)*.3/n-.15
    det, eps = np.meshgrid(u, u)
    return float(expected_infidelity(seq, eps, max(seq['amplitudes'])*det, sigma).mean())


def unitary_transfer(phi, amp, duration, eps, det):
    h = .5*(amp*(1+eps)*(np.cos(phi)*SX-np.sin(phi)*SY)-det*SZ)
    unitary = expm(-1j*np.pi*duration*h)
    return np.array([[np.trace(p @ unitary @ q @ unitary.conj().T).real/2
                      for q in PAULI] for p in PAULI])


def validate(seqs):
    rng = np.random.default_rng(20261026)
    nodes, weights = hermgauss(18)
    weights /= np.sqrt(np.pi)
    errors = []
    matrix_errors = []
    for seq in seqs:
        for eps, det in rng.uniform(-.2, .2, (3, 2)):
            for sigma in [1e-4, 1e-3, .03]:
                independent = np.eye(3)
                for phi, amp, duration in zip(seq['phases'], seq['amplitudes'], seq['durations_T']):
                    step = sum(w*unitary_transfer(phi+np.sqrt(2)*sigma*x, amp, duration, eps, det)
                               for x, w in zip(nodes, weights))
                    independent = step @ independent
                expected = averaged_transfer(seq, eps, det, sigma)
                errors.append(float(np.max(np.abs(independent-expected))))
            direct = np.eye(2, dtype=complex)
            for phi, amp, duration in zip(seq['phases'], seq['amplitudes'], seq['durations_T']):
                h = .5*(amp*(1+eps)*(np.cos(phi)*SX-np.sin(phi)*SY)-det*SZ)
                direct = expm(-1j*np.pi*duration*h) @ direct
            target = expm(-.5j*seq['theta']*SX)
            matrix_errors.append(abs(float(infidelity(seq, eps, det))-(4-abs(np.trace(target.conj().T @ direct))**2)/6))
    assert max(errors) < 3e-13
    assert max(matrix_errors) < 3e-14
    x9 = next(s for s in seqs if s['name'] == 'X9a')
    exact_errors = [abs(float(expected_infidelity(x9, 0, 0, s)) + np.expm1(-18*s*s)/3)
                    for s in [1e-6, 1e-4, 1e-3, .03]]
    assert max(exact_errors) < 3e-14
    return dict(gauss_hermite_transfer_max_absolute_difference=max(errors),
                matrix_product_infidelity_max_absolute_difference=max(matrix_errors),
                X9a_analytical_max_absolute_difference=max(exact_errors))


def write_csv(name, rows):
    with (HERE/name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    seqs = json.loads((HERE/'phase_controls.json').read_text())
    validation = validate(seqs)
    print('Independent checks:', validation, flush=True)
    sigmas = np.unique(np.r_[0, np.logspace(-6, np.log10(.03), 33), 1e-4, 1e-3, .03])
    records, rounding, printed, convergence = [], [], [], []
    for seq in seqs:
        for sigma in sigmas:
            records.append(dict(sequence=seq['name'], sigma_rad=float(sigma),
                                nominal_mean_infidelity=float(expected_infidelity(seq, 0, 0, sigma)),
                                domain_mean_infidelity=grid_mean(seq, sigma)))
        for bits in range(8, 25):
            step = 2*np.pi/2**bits
            rounded = {**seq, 'phases':(np.round(np.asarray(seq['phases'])/step)*step).tolist()}
            rounding.append(dict(sequence=seq['name'], bits=bits, step_rad=step,
                                 nominal_infidelity=float(infidelity(rounded, 0, 0)),
                                 domain_mean_infidelity=grid_mean(rounded)))
        rounded = {**seq, 'phases':(np.round((np.asarray(seq['phases'])/np.pi) % 2, 5)*np.pi).tolist(),
                   'amplitudes':np.round(seq['amplitudes'], 5).tolist()}
        base = grid_mean(seq, n=401)
        new = grid_mean(rounded, n=401)
        printed.append(dict(sequence=seq['name'], nominal_infidelity=float(infidelity(rounded, 0, 0)),
                            domain_mean_infidelity=new, full_precision_mean=base,
                            relative_mean_change=new/base-1))
        for sigma in [0., 1e-4, 1e-3, .03]:
            coarse = next(r for r in records if r['sequence']==seq['name'] and r['sigma_rad']==sigma)
            fine = grid_mean(seq, sigma, 401)
            convergence.append(dict(sequence=seq['name'], sigma_rad=sigma,
                                    mean_201=coarse['domain_mean_infidelity'], mean_401=fine,
                                    relative_change=fine/coarse['domain_mean_infidelity']-1))
        print('Completed', seq['name'], flush=True)
    validation['grid_convergence'] = convergence
    payload = dict(phase_model='Independent zero-mean Gaussian offsets, constant during each pulse',
                   phase_units='radians', domain='abs(epsilon),abs(Delta/Omega_max)<=0.15',
                   domain_sampling='201x201 midpoint grid; 401x401 convergence checks',
                   controls=seqs, validation=validation, gaussian=records, digital_rounding=rounding,
                   five_decimal_tables=printed)
    (HERE/'phase_precision_data.json').write_text(json.dumps(payload, indent=2)+'\n')
    write_csv('phase_precision_metrics.csv', records)
    write_csv('phase_quantization_metrics.csv', rounding)
    write_csv('five_decimal_metrics.csv', printed)
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'cm', 'font.size':11,
                         'axes.linewidth':.8, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), layout='constrained')
    colors = ['#31688e', '#d95f02', '#7b3294', '#238443']
    for seq, color in zip(seqs, colors):
        rows = [r for r in records if r['sequence']==seq['name'] and r['sigma_rad']>0]
        x = [r['sigma_rad'] for r in rows]
        label = seq['name']
        axes[0].loglog(x, [r['nominal_mean_infidelity'] for r in rows], color=color, label=label)
        axes[1].loglog(x, [r['domain_mean_infidelity'] for r in rows], color=color, label=label)
        baseline = next(r['domain_mean_infidelity'] for r in records
                        if r['sequence']==seq['name'] and r['sigma_rad']==0)
        axes[1].axhline(baseline, color=color, ls=':', lw=1)
        rows = [r for r in rounding if r['sequence']==seq['name']]
        axes[2].semilogy([r['bits'] for r in rows],
                         [max(1e-16, r['nominal_infidelity']) for r in rows],
                         '-o', color=color, ms=3, lw=1.2, label=label)
    axes[0].set_title('(a) Nominal gate')
    axes[1].set_title(r'(b) Mean over $B_{0.15}$')
    axes[2].set_title('(c) Digital phase resolution')
    for ax in axes[:2]:
        ax.set_xlabel(r'RMS phase error $\sigma_\phi$ (rad)')
        ax.set_xlim(1e-6, .03)
    axes[0].set_ylabel('Phase-averaged gate infidelity')
    axes[1].set_ylabel('Phase- and domain-averaged infidelity')
    axes[2].set_xlabel(r'Phase resolution $b$ (bits)')
    axes[2].set_ylabel('Nominal gate infidelity')
    axes[0].set_ylim(1e-13, .03)
    axes[1].set_ylim(1e-7, .03)
    axes[2].set_ylim(6e-17, .01)
    axes[2].set_xticks([8, 12, 16, 20, 24])
    axes[2].set_xlim(7.5, 24.5)
    for ax in axes:
        ax.grid(True, which='major', alpha=.17)
        ax.legend(fontsize=8, loc='upper right' if ax is axes[2] else 'upper left')
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES/'phase_precision.pdf')
    fig.savefig(FIGURES/'phase_precision.png', dpi=230)
    print('Selected results:', flush=True)
    for row in records:
        if row['sigma_rad'] in [0, 1e-4, 1e-3]:
            print(row, flush=True)
    print('Five-decimal controls:', printed, flush=True)
    print('16-bit controls:', [r for r in rounding if r['bits']==16], flush=True)
    print('Max relative grid change:', max(abs(r['relative_change']) for r in convergence), flush=True)


if __name__ == '__main__':
    main()
