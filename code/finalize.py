"""Freeze selected assets, audit interface, then score the frozen test once."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
from common import load_inputs, DATA, PROTOCOL


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    completed = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    return completed.stdout


def main():
    root = Path('runs')
    if (root/'freeze.json').exists():
        raise RuntimeError('Already frozen. Do not silently replace or rescore; inspect saved results.')
    selection = json.loads((root/'selection.json').read_text())
    selected = root / selection['winner']['name']
    dest = root/'model_bundle'
    shutil.copytree(selected, dest)
    assets = {str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in dest.iterdir() if p.is_file()}
    asset_bytes = sum(v['bytes'] for v in assets.values())
    assert asset_bytes <= 16 * 2**20
    sources = {p.name:sha(p) for p in Path('.').glob('*.py')}
    # Freeze before reading any test reference through official scoring.
    frozen = {'selected':selection['winner'], 'sources':sources, 'assets':assets,
              'asset_bytes':asset_bytes, 'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (root/'freeze.json').write_text(json.dumps(frozen,indent=2))
    # Novel input size and IDs, using only train candidate data. Original manifest untouched.
    bank, arrays = load_inputs('train')
    audit = root/'audit_inputs'
    (audit/'candidates').mkdir(parents=True,exist_ok=True)
    subset = bank[:7]
    for i, row in enumerate(subset):
        row['id'] = f'interface-check-{i}'
    names = ['candidates/gsm8k_audit.jsonl','candidates/gsm8k_audit.npz']
    (audit/names[0]).write_text(''.join(json.dumps(r)+'\n' for r in subset),encoding='utf-8')
    np.savez(audit/names[1],embeddings=arrays['embeddings'][:7].reshape(-1,1024))
    (audit/'manifest.json').write_text(json.dumps({'protocol':PROTOCOL,'splits':{'audit':7},
        'sha256':{name:sha(audit/name) for name in names}}))
    print(run(['predict.py','--method','submission.py','--checkpoint',str(dest),'--split','audit',
               '--data-dir',str(audit),'--output',str(root/'audit-predictions.json')]))
    # Official validation results for frozen model and baselines.
    for name, checkpoint, method in [('validation',dest,'submission.py'),
                                      ('baseline-validation',root/'majority','submission.py'),
                                      ('first-validation',None,'baseline_first.py')]:
        args = ['predict.py','--method',method,'--split','validation','--output',str(root/f'{name}-predictions.json')]
        if checkpoint:
            args += ['--checkpoint',str(checkpoint)]
        tick = time.perf_counter()
        print(run(args))
        print(run(['score_predictions.py','--split','validation','--predictions',str(root/f'{name}-predictions.json'),
                   '--output',str(root/f'{name}-score.json')]))
        frozen[f'{name}_seconds'] = time.perf_counter()-tick
    tick = time.perf_counter()
    print(run(['predict.py','--method','submission.py','--checkpoint',str(dest),'--split','test',
               '--output',str(root/'test-predictions.json')]))
    print(run(['score_predictions.py','--split','test','--predictions',str(root/'test-predictions.json'),
               '--output',str(root/'test-score.json')]))
    frozen['test_total_seconds'] = time.perf_counter()-tick
    frozen['environment'] = {'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
                            'cpu':platform.processor(),'logical_cpus':os.cpu_count(),'threads':4,
                            'reference_machine_verified':False}
    frozen['pip_freeze'] = run(['-m','pip','freeze']).splitlines()
    (root/'reproduction.json').write_text(json.dumps(frozen,indent=2))
    print(json.dumps({'test_total_seconds':frozen['test_total_seconds'],'asset_bytes':asset_bytes}))


if __name__ == '__main__':
    main()
