"""Fixed-control midpoint-grid convergence on the manuscript's common-peak B_0.15.

No controls are optimized. Run once to calculate data and checks; the TeX source
reads the resulting CSV files directly. NumPy and SciPy are required.
"""
from pathlib import Path
import argparse
import csv
import shutil
import hashlib
import json
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rectangular import infidelity, matrix_infidelity, metrics

NAMES = ['X5c', 'H5c', 'X15c', 'H15c']
GRIDS = [8, 16, 32, 64, 128, 256, 512, 801, 1601, 3201]
RADIUS = .15

def midpoint_mean(seq, m, chunk_rows=32):
    # The detuning fed to the stored propagator is delta=p*delta_c.
    axis = (np.arange(m) + .5) * (2 * RADIUS / m) - RADIUS
    peak = max(seq['amplitudes'])
    total = 0.
    for start in range(0, m, chunk_rows):
        eps, dc = np.meshgrid(axis, axis[start:start + chunk_rows])
        total += float(np.sum(infidelity(seq, eps, peak * dc), dtype=np.float64))
    return total / (m * m)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/grid_convergence')
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    controls_path = ROOT / 'data/controls.json'
    controls = json.loads(controls_path.read_text())['sequences']
    publication = json.loads((ROOT / 'data/static_metrics.json').read_text())
    frozen = {name: publication[name]['small']['mean'] for name in NAMES}
    saved = json.loads((HERE / 'checks.json').read_text())['checks']
    rows, checks = [], {}
    rng = np.random.default_rng(20261106)
    for name in NAMES:
        seq = controls[name]
        peak = max(seq['amplitudes'])
        # Direct matrix exponentials provide a separate convention check.
        errors = []
        for eps, dc in rng.uniform(-RADIUS, RADIUS, (5, 2)):
            value = float(infidelity(seq, eps, peak * dc))
            errors.append(abs(value - matrix_infidelity(seq, eps, peak * dc)))
        assert max(errors) < 2e-14, (name, errors)
        series = []
        for m in GRIDS:
            started = time.perf_counter()
            mean = midpoint_mean(seq, m)
            series.append({'name': name, 'm': m, 'mean': mean})
            print(f'{name} m={m}: {mean:.12e} ({time.perf_counter()-started:.1f}s)', flush=True)
        reference = series[-1]['mean']
        np.testing.assert_allclose(reference, saved[name]['reference_mean_3201'], rtol=2e-11, atol=1e-17)
        for row in series:
            row['relative_difference_percent'] = 100 * (row['mean'] / reference - 1)
            row['absolute_relative_difference_percent'] = abs(row['relative_difference_percent'])
        published_grid = next(row['mean'] for row in series if row['m'] == 801)
        assert np.isclose(published_grid, frozen[name], rtol=2e-12, atol=1e-17), (name, published_grid, frozen[name])
        small_check = metrics(seq, r=RADIUS, n=32, peak=True)['mean']
        assert np.isclose(small_check, next(row['mean'] for row in series if row['m'] == 32), rtol=2e-12)
        checks[name] = {'matrix_check_max_abs_difference': max(errors),
                        'publication_mean_801': frozen[name],
                        'calculated_mean_801': published_grid,
                        'reference_mean_3201': reference,
                        'relative_difference_801_vs_3201_percent': 100 * (published_grid / reference - 1)}
        with (output / f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(series[0]))
            writer.writeheader(); writer.writerows(series)
        with (output / f'{name}_deviation.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(series[0]))
            writer.writeheader(); writer.writerows(series[:-1])
        rows.extend(series)
    with (output / 'evaluation_grid_convergence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {'protocol': {'controls_fixed': True, 'domain': '|epsilon|, |delta_c| <= 0.15',
                          'quadrature': 'equal-weight cell midpoints in both coordinates',
                          'grids_per_axis': GRIDS, 'reference_grid': 3201,
                          'drive': 'common peak nominal Rabi frequency; native delta=p*delta_c',
                          'metric': 'average gate infidelity; ideal virtual Z completion for Hadamard',
                          'noise': 'none; deterministic rectangular pulses',
                          'controls_sha256': hashlib.sha256(controls_path.read_bytes()).hexdigest()},
              'checks': checks}
    (output / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(HERE / 'evaluation_grid_convergence.tex', output / 'evaluation_grid_convergence.tex')
    print(json.dumps(checks, indent=2), flush=True)

if __name__ == '__main__':
    main()
