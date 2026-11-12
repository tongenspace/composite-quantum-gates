"""Select independently validated X13c/X15c controls; preserve existing entries."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from scipy.linalg import expm
from scipy.stats import qmc

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT))
from rectangular import infidelity,metrics,matrix,matrix_infidelity,SX


def worst(s,peak,n=801):
    u=np.linspace(-.15,.15,n);D,E=np.meshgrid(u,u)
    return float(infidelity(s,E,D*(max(s['amplitudes']) if peak else 1.)).max())


def validate(s):
    native=metrics(s,n=801);peak=metrics(s,n=801,peak=True)
    coarse_native=metrics(s,n=401);coarse_peak=metrics(s,n=401,peak=True)
    rounded=s | dict(amplitudes=np.round(s['amplitudes'],8).tolist(),
        phases=(np.round(np.asarray(s['phases'])/np.pi%2,8)*np.pi).tolist())
    printed=float(infidelity(rounded,0,0))
    assert native['center']<1e-12 and printed<1e-12
    assert .05-1e-10<=min(s['amplitudes']) and max(s['amplitudes'])<=2+1e-10
    assert sum(s['amplitudes'])<=s['N']+1e-9
    assert all(np.linalg.norm([s['amplitudes'][i]-s['amplitudes'][i+1],
        np.angle(np.exp(1j*(s['phases'][i]-s['phases'][i+1])))])>1e-7 for i in range(s['N']-1))
    rng=np.random.default_rng(202610194+s['N']);error=0.;unitarity=0.
    G=expm(-.5j*np.pi*SX);nominal=matrix(s,0,0)
    phase=np.trace(G.conj().T@nominal);phase=phase/abs(phase)
    nominal_matrix_residual=float(np.linalg.norm(nominal-phase*G))
    assert nominal_matrix_residual<3e-6
    for e,d in rng.uniform(-.5,.5,(96,2)):
        U=matrix(s,e,d)
        error=max(error,abs(float(infidelity(s,e,d))-matrix_infidelity(s,e,d)))
        unitarity=max(unitarity,float(np.max(abs(U.conj().T@U-np.eye(2)))))
    assert error<1e-13 and unitarity<1e-13
    held=[]
    for seed in [202610194,202610195]:
        points=qmc.Sobol(2,scramble=True,seed=seed).random_base2(15)
        E,D=((2*points-1)*.15).T
        hn=float(infidelity(s,E,D).mean());hp=float(infidelity(s,E,max(s['amplitudes'])*D).mean())
        assert abs(hn-native['mean'])/native['mean']<.005
        assert abs(hp-peak['mean'])/peak['mean']<.005
        held.append(dict(seed=seed,points=len(points),native_mean=hn,peak_mean=hp))
    assert abs(native['mean']-coarse_native['mean'])/native['mean']<.001
    assert abs(peak['mean']-coarse_peak['mean'])/peak['mean']<.001
    return dict(controls=s,native=native,peak=peak,coarse_native=coarse_native,coarse_peak=coarse_peak,
        wide_native=metrics(s,r=.5,n=801),wide_peak=metrics(s,r=.5,n=801,peak=True),
        native_worst=worst(s,False),peak_worst=worst(s,True),held_out=held,
        matrix_max_difference=error,unitarity_max_error=unitarity,
        nominal_matrix_residual=nominal_matrix_residual,printed_eight_decimal_nominal_infidelity=printed)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    winners={}
    if args.verify:
        saved=json.loads((BASE/'selected_controls.json').read_text())
        registry=json.loads((ROOT/'data/pulse_controls.json').read_text())['sequences']
        for name,row in saved.items():
            assert row['controls']==registry[name]
            checked=validate(registry[name])
            np.testing.assert_allclose([checked['native']['mean'],checked['peak']['mean']],
                [row['native']['mean'],row['peak']['mean']],rtol=1e-12,atol=1e-20)
        print('X13c/X15c fine-grid, held-out, matrix, printing and resource checks passed.');return
    for name in ['X13c','X15c']:
        records=json.loads((BASE/(name.lower()+'_search_records.json')).read_text())
        candidates=[r for r in records if r['stage'] in ['polish','nominal_projection']
                    and r['success'] and r['metrics']['nominal']<1e-12]
        candidates.sort(key=lambda r:r['metrics']['native_mean']+r['metrics']['peak_mean'])
        if not candidates:raise RuntimeError('No qualified final candidate for '+name)
        for candidate in candidates:
            try:row=validate(candidate['controls'])
            except AssertionError:continue
            row['selection']=dict(criterion='Lowest equally weighted native/common-peak mean among successful polished feasible candidates',
                source_stage=candidate['stage'],start_index=candidate['start_index'],seed=candidate['seed'],
                training_grids=[25,51],validation_grids=[401,801],half_width=.15,
                area_cap_over_pi=candidate['controls']['N'],amplitude_bounds=[.05,2.],
                nominal='Full target unitary up to global phase; supplied/printed infidelity <1e-12',
                limitation='Best found under the disclosed search; no global-optimality or perturbative-order claim')
            winners[name]=row;break
        if name not in winners:raise RuntimeError('No independently valid candidate for '+name)
        print(name,json.dumps(dict(native=winners[name]['native'],peak=winners[name]['peak'],
            nominal_printed=winners[name]['printed_eight_decimal_nominal_infidelity'])),flush=True)
    (BASE/'selected_controls.json').write_text(json.dumps(winners,indent=2))
    if args.apply:
        p=ROOT/'data/pulse_controls.json';registry=json.loads(p.read_text())
        archive=BASE/'reference_controls.json'
        if not archive.exists():archive.write_bytes(p.read_bytes())
        before=json.loads(archive.read_text())
        for name,s in before['sequences'].items():assert registry['sequences'][name]==s
        registry['sequences'].update({name:r['controls'] for name,r in winners.items()})
        p.write_text(json.dumps(registry,indent=2))
        (BASE/'resource_checks.json').write_text(json.dumps(dict(
            reference_controls_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            extended_registry_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            existing_controls_preserved=True,added_controls=list(winners)),indent=2))
        print('Added X13c/X15c; all existing selected controls preserved.')


if __name__=='__main__':main()
