"""Select independently validated H refinements from saved search records.

The final policy minimizes means for H10 and prioritizes plotted 1e-4 coverage
for the other five H gates. Every candidate has nominal matching. Fine-grid
means may reach 1.102/1.052 of original native/common-peak values, allowing
the stated 1.1/1.05 training budgets and quadrature changes. --apply updates
only the canonical registry and nominal-resource inputs; figures are separate.
"""
from pathlib import Path
import argparse,json
import numpy as np
from search import evaluate
from rectangular import metrics,infidelity,matrix_infidelity
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NAMES=['H3','H4','H5','H7','H8','H10']
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gates',nargs='+',choices=NAMES,default=NAMES);parser.add_argument('--apply',action='store_true')
    parser.add_argument('--criterion',choices=['coverage','mean'],default=None);args=parser.parse_args()
    old=json.loads((HERE/'input_controls.json').read_text());selected=json.loads((HERE/'selected_controls.json').read_text())
    for name in args.gates:
        criterion=args.criterion or ('mean' if name=='H10' else 'coverage')
        original=old[name];baseline=evaluate(original,n=801);wold=metrics(original,r=.5,n=801)
        records=[]
        for tag in ['remaining','followup','native']:
            p=HERE/(name.lower()+'_'+tag+'_records.json')
            if p.exists():records.extend(json.loads(p.read_text()))
        good=[]
        for row in records:
            s=row['controls'];m=row['metrics']
            domain_ok=(m['native_mean']<baseline['native_mean'] and m['peak_mean']<baseline['peak_mean']) if criterion=='mean' else (m['native_mean']<=1.102*baseline['native_mean'] and m['peak_mean']<=1.052*baseline['peak_mean'] and row['wide_domain_metrics']['fraction_below_1e4']>wold['fraction_below_1e4'])
            if row['success'] and m['nominal']<1e-12 and domain_ok and sum(s['amplitudes'])<=sum(original['amplitudes'])+1e-8 and max(s['amplitudes'])<=max(original['amplitudes'])+1e-8:
                good.append(row)
        assert good,'No qualifying improvement for '+name
        key=(lambda r:r['metrics']['native_mean']+r['metrics']['peak_mean']) if criterion=='mean' else (lambda r:-r['wide_domain_metrics']['fraction_below_1e4'])
        finalists=sorted(good,key=key)[:8]
        for row in finalists:row['fine_wide']=metrics(row['controls'],r=.5,n=801)
        winner=min(finalists,key=key) if criterion=='mean' else max(finalists,key=lambda r:(r['fine_wide']['fraction_below_1e4'],-r['metrics']['peak_mean']))
        s=winner['controls'];m=evaluate(s,n=801);coarse=evaluate(s,n=401)
        rounded={**s,'amplitudes':np.round(s['amplitudes'],8).tolist(),'phases':(np.round(np.asarray(s['phases'])/np.pi%2,8)*np.pi).tolist()}
        center=float(infidelity(rounded,0,0))
        error=max(abs(float(infidelity(s,e,d))-matrix_infidelity(s,e,d)) for e,d in np.random.default_rng(202610170+NAMES.index(name)).uniform(-.5,.5,(40,2)))
        assert error<1e-13 and center<1e-12
        if criterion=='mean':assert m['native_mean']<baseline['native_mean'] and m['peak_mean']<baseline['peak_mean']
        else:assert winner['fine_wide']['fraction_below_1e4']>wold['fraction_below_1e4']
        assert abs(m['native_mean']/coarse['native_mean']-1)<.001 and abs(m['peak_mean']/coarse['peak_mean']-1)<.001
        selected[name]=dict(controls=s,metrics=m,baseline_metrics=baseline,grid_401_metrics=coarse,
            wide_domain_metrics=winner['fine_wide'],baseline_wide_domain_metrics=wold,
            wide_peak_domain_metrics=metrics(s,r=.5,n=801,peak=True),baseline_wide_peak_domain_metrics=metrics(original,r=.5,n=801,peak=True),
            printed_eight_decimal_nominal_infidelity=center,matrix_max_difference=error,
            selection=('Lower native/common-peak means and exact nominal matching, within the specified resource bounds; smaller wide threshold region disclosed.' if criterion=='mean' else 'Larger plotted 1e-4 region and exact nominal matching, within the specified resource bounds; native/common-peak mean budgets and trade-offs disclosed.'),
            selection_criterion=criterion,
            chosen_record=dict(stage=winner['stage'],start_index=winner['start_index'],protocol=winner['protocol']))
        print('SELECTED',name,'wide growth',winner['fine_wide']['fraction_below_1e4']/wold['fraction_below_1e4']-1,
            'mean changes',m['native_mean']/baseline['native_mean']-1,m['peak_mean']/baseline['peak_mean']-1,
            'resource changes',m['area_over_target']/baseline['area_over_target']-1,m['duration_over_target']/baseline['duration_over_target']-1,
            'printed nominal',center,flush=True)
    (HERE/'selected_controls.json').write_text(json.dumps(selected,indent=2)+'\n')
    if args.apply:
        p=ROOT/'data/pulse_controls.json';reg=json.loads(p.read_text())
        for name in args.gates:reg['sequences'][name]=selected[name]['controls']
        p.write_text(json.dumps(reg,indent=2)+'\n')
        p=ROOT/'studies/nominal_resources/controls.json';rows=json.loads(p.read_text())
        for i,s in enumerate(rows):
            if s['name'] in args.gates:
                v=selected[s['name']]['controls'];rows[i]={**s,**v,'areas':(np.pi*np.asarray(v['amplitudes'])).tolist()}
        p.write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
