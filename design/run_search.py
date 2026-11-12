"""Run a specified control-design problem in an isolated workspace.

The supplied pulse registry remains unchanged. Reproduction of fixed-control
figures and metrics uses reproduce.py and requires no optimization.
"""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('family',choices=['x-long','x13-width','x15-width','hadamard','nine-pulse','references'])
    ap.add_argument('--pulse-count',type=int,choices=[13,15],default=13)
    ap.add_argument('--target',choices=['X','H'],default='X')
    ap.add_argument('--starts',type=int,default=6)
    ap.add_argument('--run',default='design_run')
    ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args()
    run=(ROOT/'outputs'/a.run).resolve()
    assert run.is_relative_to((ROOT/'outputs').resolve())
    assert not run.exists(),'Choose a new --run name to preserve an existing run.'
    paper=run/'workspace/inputs';research=run/'research/numerical'
    for d in [paper/'data',paper/'studies/pulse_shape',research,run/'workspace/initial',
              run/'workspace/hadamard/search']:
        d.mkdir(parents=True,exist_ok=True)
    searches=ROOT/'design/searches'
    for directory in ['control_refinement','x_extension','x13_width','objective_selection','higher_order']:
        shutil.copytree(searches/directory,paper/'studies'/directory)
    for name in ['rectangular.py','concatenated.py','independent_matrices.py']:
        if (searches/name).exists():shutil.copy2(searches/name,paper/name)
    shutil.copy2(searches/'shaped_model.py',paper/'studies/pulse_shape/shaped_model.py')
    shutil.copy2(ROOT/'data/controls.json',paper/'data/pulse_controls.json')
    shutil.copy2(ROOT/'design/initial_controls.json',run/'workspace/initial/pulse_controls.json')
    current=json.loads((ROOT/'data/controls.json').read_text())
    # Map publication identifiers to the identifiers required by the search inputs.
    mapping=json.loads((ROOT/'data/sequence_name_map.json').read_text())
    inverse={v:k for k,v in mapping.items()}
    snapshot={**current,'sequences':{inverse.get(k,k):dict(v,name=inverse.get(k,k))
                                   for k,v in current['sequences'].items()}}
    (run/'workspace/hadamard/controls.json').write_text(json.dumps(snapshot,indent=2))
    for path in (searches/'hadamard_and_nine_pulse').iterdir():
        if path.suffix=='.py':shutil.copy2(path,research/path.name)
        elif path.suffix=='.json':shutil.copy2(path,run/'workspace/hadamard/search'/path.name)
    if a.family=='x-long':
        script=paper/'studies/x_extension/extend_x.py';args=['--gate',f'X{a.pulse_count}c','--starts',str(a.starts)]
    elif a.family=='x13-width':
        script=paper/'studies/x13_width/search_x13_width.py';args=['--starts',str(a.starts)]
    elif a.family=='x15-width':
        script=paper/'studies/objective_selection/search_width.py';args=['--gate','X15c','--starts',str(a.starts)]
    elif a.family=='hadamard':
        script=research/'design_h.py';args=['--N',str(a.pulse_count),'--starts',str(a.starts)]
    elif a.family=='nine-pulse':
        script=research/'refine_mean.py';args=['--gate',a.target,'--N','9','--starts',str(a.starts)]
    else:
        script=paper/'studies/higher_order/construct.py';args=[]
    command=[sys.executable,str(script),*args]
    (run/'search_command.json').write_text(json.dumps(dict(command=command,
                    current_registry_replaced=False,input_identifiers=True),indent=2)+'\n')
    print('Isolated layout:',run)
    print('Command:',script.name,*args)
    if not a.prepare_only:subprocess.run(command,cwd=run,check=True)

if __name__=='__main__':main()
