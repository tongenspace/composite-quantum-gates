"""Small, independent checks of the final model, controls and stored results."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rectangular import infidelity,matrix,matrix_infidelity

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'outputs/checks');a=ap.parse_args()
    reg=json.loads((ROOT/'data/controls.json').read_text());rng=np.random.default_rng(20261026)
    checks={}
    for name,seq in reg['sequences'].items():
        differences=[];unit=[]
        for e,dc in rng.uniform(-.3,.3,(8,2)):
            d=max(seq['amplitudes'])*dc
            differences.append(abs(float(infidelity(seq,e,d))-matrix_infidelity(seq,e,d)))
            u=matrix(seq,e,d);unit.append(float(np.max(abs(u.conj().T@u-np.eye(2)))))
        assert max(differences)<4e-13,name
        assert max(unit)<4e-13,name
        checks[name]=dict(nominal=float(infidelity(seq,0,0)),matrix_difference=max(differences),unitarity=max(unit))
    for name in reg['main_numerical']:
        if name!='X5c':assert checks[name]['nominal']<1e-12,name
    assert abs(checks['X5c']['nominal']-8.27e-5)<1e-7
    # Exact duration-error mapping, tested on unequal-amplitude and unequal-time controls.
    for name in ['X9a','X13c','H9c','rCinBB_H']:
        s=reg['sequences'][name];e=.07;d=-.09;eta=.021
        changed={**s,'durations_T':((1+eta)*np.array(s['durations_T'])).tolist()}
        direct=float(infidelity(changed,e,d));mapped=float(infidelity(s,(1+e)*(1+eta)-1,d*(1+eta)))
        assert abs(direct-mapped)<1e-13
    # Ideal virtual-Z completion preserves the primitive-level gate fidelity.
    z=np.diag([np.exp(-1j*np.pi/4),np.exp(1j*np.pi/4)])
    h=np.array([[1,1],[1,-1]])/np.sqrt(2)
    virtual=[]
    for name in ['H9c','H13c','H13w']:
        s=reg['sequences'][name]
        u=1j*z@matrix(s,.07,max(s['amplitudes'])*.11)@z
        value=float((4-abs(np.trace(h.conj().T@u))**2)/6)
        expected=float(infidelity(s,.07,max(s['amplitudes'])*.11))
        virtual.append(abs(value-expected));assert virtual[-1]<1e-13
    phase=json.loads((ROOT/'studies/phase_precision/phase_precision_data.json').read_text())
    discrepancies={}
    for row in phase['digital_rounding']:
        if row['bits']==16:
            base=next(v['domain_mean_infidelity'] for v in phase['gaussian']
                      if v['sequence']==row['sequence'] and v['sigma_rad']==0)
            discrepancies[row['sequence']]=row['domain_mean_infidelity']/base-1
    a.output.mkdir(parents=True,exist_ok=True)
    report=dict(driven_controls_tested=len(checks),controls=checks,
                duration_mapping_verified=True,virtual_Z_max_difference=max(virtual),
                digital_16bit_domain_relative_changes=discrepancies,
                phase_data_note='Digital phase resolution and five-decimal coefficient rounding are distinct tests; see docs/NUMERICAL_NOTES.md.')
    (a.output/'check_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Checked',len(checks),'controls against independent matrix exponentials.')
    print('Largest matrix difference:',max(v['matrix_difference'] for v in checks.values()))
    print('16-bit domain relative changes:',discrepancies)

if __name__=='__main__':main()
