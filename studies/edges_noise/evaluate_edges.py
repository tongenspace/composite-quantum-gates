"""Finite-edge and continuous-detuning tests of six fixed manuscript controls.

All domain comparisons use the same 81x81 midpoint grid and physical peak.
Run from any directory. No fitting, control selection, or reoptimization occurs.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from rectangular import infidelity
import shaped_model as model

HERE = Path(__file__).resolve().parent
NAMES = ['X9a', 'X13c', 'X13w', 'H9c', 'H13c', 'H13w']
SEED = 2026102242


def setup(seq):
    peak = max(seq['amplitudes'])
    model.T_NS = 100*peak
    model.F_REF_HZ = 5e6/peak
    return peak


def domain(seq, edge_ns=4., noise=True, n=81, samples=128, steps=16):
    peak = setup(seq)
    u = (np.arange(n)+.5)*.3/n-.15
    det, eps = np.meshgrid(u, u)
    vals, norm = model.evaluate(seq, edge_ns/(100*peak), eps, peak*det,
                               samples=samples, seed=SEED, noise_scale=float(noise),
                               edge_steps=steps, flat_steps=max(4, steps), modes=96, chunk=2048)
    means = vals.reshape(len(vals), -1).mean(axis=1)
    if noise:
        pairs = (means[:len(means)//2]+means[len(means)//2:])/2
        se = float(pairs.std(ddof=1)/np.sqrt(len(pairs)))
    else:
        se = 0.
    return dict(mean=float(means.mean()), standard_error=se, grid_n=n,
                samples=len(vals), max_unitarity_error=norm)


def checks(seq):
    peak = setup(seq)
    coeff = model.gaussian_coefficients(2, modes=96, seed=2026102244)[0]
    checks = []
    for eps, det in [(0., 0.), (.11, -.09), (-.08, .14)]:
        got, norm = model.evaluate(seq, .04/peak, eps, peak*det,
                                   coefficients=np.array([coeff, -coeff]),
                                   edge_steps=24, flat_steps=16)
        expected, unit = model.reference_matrix(seq, .04/peak, eps, peak*det, coefficients=coeff)
        checks.append(dict(epsilon=eps, delta_c=det, difference=abs(float(got[0])-expected),
                           reference_unitarity=unit))
    assert max(r['difference'] for r in checks) < 1e-8
    coarse = domain(seq, n=9, samples=16, steps=8)
    fine = domain(seq, n=9, samples=16, steps=24)
    assert abs(coarse['mean']-fine['mean']) < 1e-8
    return dict(independent_DOP853=checks, step_refinement=dict(coarse=coarse, fine=fine))


def evaluate(seq):
    peak = setup(seq)
    u = (np.arange(81)+.5)*.3/81-.15
    det, eps = np.meshgrid(u, u)
    rectangular = float(infidelity(seq, eps, peak*det).mean())
    rows = []
    for edge in [0., 2., 4., 8.]:
        row = domain(seq, edge, noise=False)
        rows.append(dict(edge_ns=edge, **row))
    assert abs(rows[0]['mean']-rectangular) < 1e-12
    print(seq['name'], 'edge sweep complete', flush=True)
    combined = domain(seq)
    print(seq['name'], 'domain noise complete', combined, flush=True)
    cutoffs = []
    for cutoff in [1e4, 1e5, 1e6, 1e7]:
        setup(seq)
        vals, norm = model.evaluate(seq, .04/peak, 0., 0., samples=512, seed=2026102243,
                                   edge_steps=24, flat_steps=16, modes=192, fmax=cutoff)
        v = vals.ravel()
        pairs = (v[:len(v)//2]+v[len(v)//2:])/2
        cutoffs.append(dict(fmax_Hz=cutoff, mean=float(v.mean()),
                            standard_error=float(pairs.std(ddof=1)/np.sqrt(len(pairs)))))
    validation = checks(seq)
    return dict(rectangular_mean=rectangular, edges_only=rows, combined=combined,
                nominal_cutoffs=cutoffs, validation=validation,
                body_duration_ns=100*peak, total_duration_ns=seq['N']*(100*peak+4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+', choices=NAMES)
    ap.add_argument('--fresh', action='store_true')
    args = ap.parse_args()
    seqs = json.loads((HERE/'edge_controls.json').read_text())
    path = HERE/'edge_results.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    for seq in seqs:
        if args.only and seq['name'] not in args.only:
            continue
        if seq['name'] in data and not args.fresh:
            continue
        data[seq['name']] = evaluate(seq)
        path.write_text(json.dumps(data, indent=2)+'\n')
        print('SAVED', seq['name'], flush=True)
    protocol = dict(peak_Rabi_Hz=5e6, edge_durations_ns=[0,2,4,8],
                    domain_halfwidth=.15, midpoint_grid_n=81, noise_A_Hz2=3.6e5,
                    noise_lower_Hz=1., principal_upper_Hz=1e6,
                    noise_RMS_Hz=float(np.sqrt(3.6e5*np.log(1e6))),
                    domain_trajectories=128, domain_independent_antithetic_pairs=64,
                    domain_seed=SEED, domain_modes=96, nominal_trajectories=512,
                    nominal_seed=2026102243, nominal_modes=192,
                    fixed_controls=True, no_optimization=True)
    (HERE/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')


if __name__ == '__main__':
    main()
