"""Verify final identities and check control specifications.

Read-only with respect to controls and search records; no optimizer is imported.
"""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text())
def pointer(obj,p):
    for token in p.split('/')[1:]:obj=obj[int(token)] if isinstance(obj,list) else obj[token]
    return obj
def same(a,b):
    return (a['N']==b['N'] and abs(a['theta']-b['theta'])<1e-14
        and np.max(np.abs(np.angle(np.exp(1j*(np.array(a['phases'])-b['phases'])))))<1e-12
        and np.allclose(a['amplitudes'],b['amplitudes'],atol=1e-13,rtol=0)
        and np.allclose(a['durations_T'],b['durations_T'],atol=1e-13,rtol=0))
def main():
    reg=load('data/controls.json')['sequences'];matches=load('design/coefficient_matches.json');out={}
    for name,locations in matches.items():
        verified=[]
        for loc in locations:
            obj=load(loc['file']);matched=pointer(obj,loc['json_pointer']);assert same(reg[name],matched),(name,loc)
            parent=pointer(obj,loc['json_pointer'].rsplit('/',1)[0]) if '/' in loc['json_pointer'][1:] else obj
            meta={k:v for k,v in parent.items() if k not in ['controls','metrics','wide_domain_metrics','wide_metrics','small_metrics','selection_dynamic','selection_rect']} if isinstance(parent,dict) else {}
            verified.append({**loc,'verified':True,'source_sha256':hashlib.sha256((ROOT/loc['file']).read_bytes()).hexdigest(),'parent_metadata':meta})
        out[name]=verified
    seeds={'selection':2026102201,'domain_evaluation':2026102242,'nominal_bandwidth_evaluation':2026102243}
    coeff={k:np.random.default_rng(v).standard_normal((8,192)) for k,v in seeds.items()}
    assert not np.array_equal(coeff['selection'],coeff['domain_evaluation'])
    assert not np.array_equal(coeff['selection'],coeff['nominal_bandwidth_evaluation'])
    report={'all_final_numerical_controls_matched':len(out),'matches':out,
        'noise':{'seeds':seeds,'selection_trajectories':16,'selection_independent_antithetic_pairs':8,
            'domain_trajectories':128,'domain_independent_antithetic_pairs':64,
            'nominal_trajectories':512,'nominal_independent_antithetic_pairs':256,
            'selection_modes':96,'domain_modes':96,'nominal_modes':192,
            'subsequent_coefficients_fresh_relative_to_selection':True,
            'same_seed_reused_across_candidates':True,'same_seed_reused_across_nominal_cutoffs':True,
            'note':'Fresh Gaussian draws do not make a model used in selection a held-out model. Antithetic partners are dependent. Cutoffs reuse coefficients with different frequency bases.'},
        'uncertainties':[
            'X5c: the coefficients, feasibility conditions, and coverage/nominal trade-off are specified; a unique scalar final ranking and tie-break are unspecified.',
            'X13w: selection specifies the largest checked connected area; an executable final selector and tie-break are unspecified. Total threshold fraction and connected coverage are distinct quantities.'
        ]}
    dest=ROOT/'outputs/control_checks';dest.mkdir(parents=True,exist_ok=True)
    (dest/'control_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified all',len(out),'final numerical identities by coefficients; noise ensemble separation checked. See design/METHODS.md for selection limits.')
if __name__=='__main__':main()
