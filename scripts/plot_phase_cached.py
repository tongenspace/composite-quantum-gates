"""Draw Figure 11 from its saved, exactly averaged phase-error data."""
from pathlib import Path
import importlib.util,json,sys,textwrap
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'studies/phase_precision'
sys.path.insert(0,str(folder))
spec=importlib.util.spec_from_file_location('phase_figure',folder/'simulate_phase_precision.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data=json.loads((folder/'phase_precision_data.json').read_text())
module.seqs=data['controls'];module.records=data['gaussian'];module.rounding=data['digital_rounding']
module.FIGURES=ROOT/'outputs/figures';module.FIGURES.mkdir(parents=True,exist_ok=True)
source=(folder/'simulate_phase_precision.py').read_text()
start=source.index('    plt.rcParams.update(')
end=source.index("    print('Selected results:'",start)
exec(compile(textwrap.dedent(source[start:end]),'cached_phase_plot','exec'),module.__dict__)
print('Generated phase_precision from frozen data.')
