"""Reproduce this historical research round from its saved source data."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STEPS = ['reverse_audit.py', 'build_reverse_panel.py', 'reverse_mechanisms.py',
         'reverse_controls.py', 'run_reverse_rules.py', 'reverse_models.py', 'build_reverse_report.py']


def main():
    env = dict(os.environ)
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    env['MPLCONFIGDIR'] = str(Path(tempfile.gettempdir()) / 'reverse-research-matplotlib')
    for step in STEPS:
        print('RESEARCH_STEP', step, flush=True)
        subprocess.run([sys.executable, str(ROOT / 'analysis' / step)], cwd=ROOT, env=env, check=True)
    print('RESEARCH_COMPLETE', ROOT / 'analysis/results/reverse_report.html')


if __name__ == '__main__':
    main()
