"""Apply independently feasible finite-domain mean-search selections at N=9."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'workspace/hadamard'
registry=json.loads((OUT/'controls.json').read_text())
for gate in ['X','H']:
    selected=json.loads((OUT/f'search/{gate}9_mean_selected.json').read_text())
    assert selected['success'] and selected['nominal']<1e-12
    s=selected['controls'];assert s['N']==9
    assert all(abs(s['amplitudes'][i]-s['amplitudes'][i+1])+abs(np.angle(np.exp(1j*(s['phases'][i]-s['phases'][i+1]))))>1e-7 for i in range(8))
    registry['sequences'][gate+'9c']=s
(OUT/'controls.json').write_text(json.dumps(registry,indent=2))
print('Applied nine-pulse mean-objective selections; independent validation remains required.')
