"""Independent verification of the final refined controls and their trade-offs."""
from pathlib import Path
import json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from rectangular import infidelity,metrics,matrix_infidelity
data=json.loads((HERE/'selected_controls.json').read_text())
reg=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
rng=np.random.default_rng(202610145);maximum=0.
for name,row in data.items():
    s=row['controls'];assert s==reg[name]
    for e,d in rng.uniform(-.5,.5,(32,2)):
        maximum=max(maximum,abs(float(infidelity(s,e,d))-matrix_infidelity(s,e,d)))
    for peak,key in [(False,'native_mean'),(True,'peak_mean')]:
        m=metrics(s,n=401,peak=peak)
        np.testing.assert_allclose(m['mean'],row['grid_401_metrics'][key],rtol=1e-12,atol=1e-15)
        assert abs(m['mean']/row['metrics'][key]-1)<.001
    rounded={**s,'amplitudes':np.round(s['amplitudes'],8).tolist(),
             'phases':(np.round(np.asarray(s['phases'])/np.pi%2,8)*np.pi).tolist()}
    np.testing.assert_allclose(infidelity(rounded,0,0),row['printed_eight_decimal_nominal_infidelity'],atol=1e-17,rtol=1e-10)
h=data['H15'];x=data['X5c']
assert h['metrics']['nominal']<1e-12 and h['metrics']['peak_mean']<h['baseline_metrics']['peak_mean']/50
assert h['metrics']['area_over_target']<=h['baseline_metrics']['area_over_target']+1e-9
assert h['metrics']['duration_over_target']<h['baseline_metrics']['duration_over_target']
assert x['metrics']['nominal']<1e-4 and x['metrics']['native_fraction']>x['baseline_metrics']['native_fraction']
assert x['wide_domain_metrics']['fraction_below_1e4']>x['baseline_wide_domain_metrics']['fraction_below_1e4']
assert x['metrics']['area_over_target']<=x['baseline_metrics']['area_over_target']+1e-9
assert x['metrics']['duration_over_target']<=x['baseline_metrics']['duration_over_target']+1e-9
# The area-selected X5c is intentionally NOT described as a mean-error improvement.
assert x['metrics']['native_mean']>x['baseline_metrics']['native_mean']
assert x['metrics']['peak_mean']>x['baseline_metrics']['peak_mean']
for name in ['X9c','H6']:
    if name not in data:continue
    row=data[name];old=json.loads((HERE/'input_controls.json').read_text())[name]
    assert row['metrics']['nominal']<1e-12
    assert row['printed_eight_decimal_nominal_infidelity']<1e-12
    if row.get('selection_criterion')=='mean':
        assert row['metrics']['native_mean']<row['baseline_metrics']['native_mean']
        assert row['metrics']['peak_mean']<row['baseline_metrics']['peak_mean']
    else:assert row['wide_domain_metrics']['fraction_below_1e4']>row['baseline_wide_domain_metrics']['fraction_below_1e4']
    assert row['metrics']['native_mean']<=1.102*row['baseline_metrics']['native_mean']
    assert row['metrics']['peak_mean']<row['baseline_metrics']['peak_mean']
    assert sum(row['controls']['amplitudes'])<=sum(old['amplitudes'])+1e-8
    assert max(row['controls']['amplitudes'])<=max(old['amplitudes'])+1e-8
    assert row['metrics']['duration_over_target']<=row['baseline_metrics']['duration_over_target']+1e-8
for name in ['H3','H4','H5','H7','H8','H10']:
    row=data[name];old=json.loads((HERE/'input_controls.json').read_text())[name]
    assert row['metrics']['nominal']<1e-12 and row['printed_eight_decimal_nominal_infidelity']<1e-12
    if row.get('selection_criterion')=='mean':
        assert row['metrics']['native_mean']<row['baseline_metrics']['native_mean']
        assert row['metrics']['peak_mean']<row['baseline_metrics']['peak_mean']
    else:assert row['wide_domain_metrics']['fraction_below_1e4']>row['baseline_wide_domain_metrics']['fraction_below_1e4']
    assert row['metrics']['native_mean']<=1.102*row['baseline_metrics']['native_mean']
    assert row['metrics']['peak_mean']<=1.052*row['baseline_metrics']['peak_mean']
    assert sum(row['controls']['amplitudes'])<=sum(old['amplitudes'])+1e-8
    assert max(row['controls']['amplitudes'])<=max(old['amplitudes'])+1e-8
    assert row['metrics']['duration_over_target']<=row['baseline_metrics']['duration_over_target']+1e-8
assert maximum<1e-13
print('Refined controls, printed precision, independent matrices, convergence and disclosed trade-offs verified.',maximum)

for name,row in data.items():
    if row.get('selection_criterion')!='exact_robustness':continue
    from refine_exact_x import sampled_worst
    original=json.loads((HERE/'exact_x_baselines.json').read_text())[name]
    assert row['metrics']['nominal']<1e-12 and row['printed_eight_decimal_nominal_infidelity']<1e-12
    for key in ['native_mean','peak_mean','native_sobol_mean','peak_sobol_mean']:
        assert row['metrics'][key]<=row['baseline_metrics'][key]
    for key in ['native_fraction','peak_fraction']:
        assert row['metrics'][key]>=row['baseline_metrics'][key]
    assert row['wide_domain_metrics']['fraction_below_1e4']>=row['baseline_wide_domain_metrics']['fraction_below_1e4']
    assert row['peak_wide_domain_metrics']['fraction_below_1e4']>=row['baseline_peak_wide_domain_metrics']['fraction_below_1e4']
    for peak,key in [(False,'native_worst'),(True,'peak_worst')]:
        fresh=sampled_worst(row['controls'],401,peak)
        assert abs(fresh/row['metrics'][key]-1)<.005
        if row.get('maximum_is_constraint',False):assert fresh<=row['baseline_metrics'][key]*(1+1e-10)
    assert sum(row['controls']['amplitudes'])<=sum(original['amplitudes'])+1e-9
    assert max(row['controls']['amplitudes'])<=max(original['amplitudes'])+1e-9
print('Exact-X coverage, means, nominal matching, printed precision, sampled-maximum diagnostics and area caps passed.')
