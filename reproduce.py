"""Entry point for the manuscript's figures, performance tables and checks."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent

def execute(script,*args,cwd=None,env=None):
    subprocess.run([sys.executable,str(script),*map(str,args)],cwd=cwd or ROOT,env=env,check=True)

def study(name,calculate=True):
    source=ROOT/'studies'/name;work=ROOT/'outputs'/name
    if not work.exists():shutil.copytree(source,work,ignore=shutil.ignore_patterns('__pycache__'))
    env=dict(os.environ,CP_FIGURE_OUTPUT=str(ROOT/'outputs/figures'))
    if name=='phase_precision':
        if calculate:execute(work/'simulate_phase_precision.py',env=env)
        else:
            # Cached phase data are plotted without rerunning their averages.
            execute(ROOT/'scripts/plot_phase_cached.py',env=env)
    else:
        if calculate:execute(work/'evaluate_edges.py','--fresh',env=env)
        execute(work/'plot_edges.py',env=env)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','check','tables','figures','phase','edges','derivatives','scorbutus','grid-convergence'])
    p.add_argument('--grid',type=int,default=801);p.add_argument('--only',nargs='+')
    a=p.parse_args()
    if a.action=='verify':
        manifest=json.loads((ROOT/'MANIFEST.sha256.json').read_text())
        for name,digest in manifest.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        print('Verified',len(manifest),'frozen file hashes.');return
    if a.action=='scorbutus':execute(ROOT/'scripts/verify_scorbutus.py')
    elif a.action=='grid-convergence':
        execute(ROOT/'studies/grid_convergence/evaluate_grid_convergence.py')
        execute(ROOT/'studies/grid_convergence/plot_grid_convergence.py')
    elif a.action=='check':execute(ROOT/'scripts/check_results.py')
    elif a.action=='tables':
        args=['--grid',str(a.grid)]+(['--only',*a.only] if a.only else [])
        execute(ROOT/'scripts/static_results.py',*args)
    elif a.action=='figures':
        for name in ['plot_constant.py','plot_variable.py','plot_resources.py']:
            execute(ROOT/'scripts'/name)
        study('phase_precision',calculate=False);study('edges_noise',calculate=False)
    elif a.action=='derivatives':execute(ROOT/'scripts/check_derivatives.py')
    else:study('phase_precision' if a.action=='phase' else 'edges_noise')

if __name__=='__main__':main()
