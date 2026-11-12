"""Independently validate exact-X candidates; apply only qualified replacements."""
from pathlib import Path
import argparse,json
import numpy as np
from search import evaluate
from rectangular import metrics,infidelity,matrix_infidelity
from refine_exact_x import sampled_worst
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true')
    ap.add_argument('--gates',nargs='+',choices=['X5c','X7c','X11c'],default=['X7c','X11c'])
    ap.add_argument('--require-maximum',action='store_true');args=ap.parse_args()
    originals=json.loads((HERE/'exact_x_baselines.json').read_text())
    report={'protocol':dict(nominal_ceiling=1e-12,grid=801,held_out='16384 scrambled Sobol points',
              acceptance='No greater native/peak means or smaller native/peak 1e-4 fractions on half-width-0.15 and half-width-0.5 squares; no larger cumulative area or peak amplitude; exact nominal unitary and eight-decimal nominal infidelity below 1e-12. Selection prioritizes a wider low-error region over the sampled maximum. Maxima are reported as diagnostics, not certified worst-case bounds.',
              require_sampled_maximum=args.require_maximum),'gates':{}}
    winners={}
    rng=np.random.default_rng(202610173)
    for name,old in originals.items():
        if name not in args.gates:continue
        baseline=evaluate(old,801)
        baseline.update(native_worst=sampled_worst(old),peak_worst=sampled_worst(old,peak=True))
        bw={k:metrics(old,r=.5,n=801,peak=p) for k,p in [('native',False),('peak',True)]}
        records=[]
        for path in sorted(HERE.glob(name.lower()+'_exact_*records.json')):
            for row in json.loads(path.read_text()):
                if row['success'] and row['metrics']['nominal']<1e-12:
                    records.append({**row,'source_record':path.name})
        pre=[r for r in records if r['metrics']['native_mean']<=1.001*baseline['native_mean']
             and r['metrics']['peak_mean']<=1.001*baseline['peak_mean']
             and r['wide_domain_metrics']['native']['fraction_below_1e4']>=.999*bw['native']['fraction_below_1e4']
             and r['wide_domain_metrics']['peak']['fraction_below_1e4']>=.999*bw['peak']['fraction_below_1e4']
             and sum(r['controls']['amplitudes'])<=sum(old['amplitudes'])+1e-9]
        qualified=[]
        pre.sort(key=lambda r:-min(r['wide_domain_metrics']['native']['fraction_below_1e4']/bw['native']['fraction_below_1e4'],
                                   r['wide_domain_metrics']['peak']['fraction_below_1e4']/bw['peak']['fraction_below_1e4']))
        for r in pre:
            s=r['controls'];m=evaluate(s,801)
            m.update(native_worst=sampled_worst(s),peak_worst=sampled_worst(s,peak=True))
            w={k:metrics(s,r=.5,n=801,peak=p) for k,p in [('native',False),('peak',True)]}
            rounded={**s,'amplitudes':np.round(s['amplitudes'],8).tolist(),
                     'phases':(np.round(np.asarray(s['phases'])/np.pi%2,8)*np.pi).tolist()}
            nominal_print=float(infidelity(rounded,0,0))
            ok=(m['nominal']<1e-12 and nominal_print<1e-12
                and m['native_mean']<=baseline['native_mean'] and m['peak_mean']<=baseline['peak_mean']
                and m['native_sobol_mean']<=baseline['native_sobol_mean']
                and m['peak_sobol_mean']<=baseline['peak_sobol_mean']
                and (not args.require_maximum or (m['native_worst']<=baseline['native_worst'] and m['peak_worst']<=baseline['peak_worst']))
                and m['native_fraction']>=baseline['native_fraction'] and m['peak_fraction']>=baseline['peak_fraction']
                and w['native']['fraction_below_1e4']>=bw['native']['fraction_below_1e4']
                and w['peak']['fraction_below_1e4']>=bw['peak']['fraction_below_1e4']
                and max(s['amplitudes'])<=max(old['amplitudes'])+1e-10)
            if not ok:continue
            err=max(abs(float(infidelity(s,e,d))-matrix_infidelity(s,e,d)) for e,d in rng.uniform(-.5,.5,(32,2)))
            assert err<1e-13
            qualified.append(dict(controls=s,metrics=m,baseline_metrics=baseline,
                  grid_401_metrics=evaluate(s,401),wide_domain_metrics=w['native'],
                  baseline_wide_domain_metrics=bw['native'],peak_wide_domain_metrics=w['peak'],
                  baseline_peak_wide_domain_metrics=bw['peak'],selection_criterion='exact_robustness',
                  maximum_is_constraint=args.require_maximum,
                  matrix_max_difference=err,printed_eight_decimal_nominal_infidelity=nominal_print,
                  source_record=r['source_record'],source_start_index=r['start_index'],source_protocol=r['protocol']))
            # Candidates are ranked on the independent coarse-grid coverage;
            # accept the first to pass every fine-grid/held-out requirement.
            break
        entry=dict(baseline_controls=old,baseline_metrics=baseline,baseline_wide=bw,
                   successful_exact_candidates=len(records),qualified_candidates=len(qualified))
        if qualified:
            winner=max(qualified,key=lambda r:min(
                r['wide_domain_metrics']['fraction_below_1e4']/bw['native']['fraction_below_1e4'],
                r['peak_wide_domain_metrics']['fraction_below_1e4']/bw['peak']['fraction_below_1e4']))
            winners[name]=winner;entry['selected']=winner
        else:entry['status']='No candidate meets all requirements; canonical controls retained. This is a numerical search result, not a nonexistence proof.'
        report['gates'][name]=entry
        print(name,'qualified',len(qualified),'of',len(records),flush=True)
        if qualified:print(json.dumps(winner['metrics']),flush=True)
    (HERE/'exact_x_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.apply:
        p=HERE/'selected_controls.json';selected=json.loads(p.read_text())
        archive=HERE/'selection_inputs.json'
        if not archive.exists():archive.write_text(json.dumps(selected,indent=2)+'\n')
        selected.update(winners);p.write_text(json.dumps(selected,indent=2)+'\n')
        p=ROOT/'data/pulse_controls.json';reg=json.loads(p.read_text());reg['sequences'].update({k:r['controls'] for k,r in winners.items()});p.write_text(json.dumps(reg,indent=2)+'\n')
        p=ROOT/'studies/nominal_resources/controls.json';seqs=json.loads(p.read_text());seqs=[winners[s['name']]['controls'] if s['name'] in winners else s for s in seqs];p.write_text(json.dumps(seqs,indent=2)+'\n')
        p=HERE/'input_controls.json';initial=json.loads(p.read_text())
        for k in winners:
            if k not in initial:initial[k]=originals[k]
        p.write_text(json.dumps(initial,indent=2)+'\n')
        print('Applied:',', '.join(winners),flush=True)

if __name__=='__main__':main()
