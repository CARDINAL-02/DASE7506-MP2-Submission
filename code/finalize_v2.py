"""Promote only by pre-registered validation rule, then one final v2 test. Never compare test to select."""
import hardware_probe  # Load the Windows observer before fixing child environment.
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from threadpoolctl import threadpool_info
from common import load_inputs
from features import parsed
from supplement_analysis import uncertainty
from train import labels
# train.py sets training thread variables; restore conservative inference child settings.
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def observed_run(args):
    p=subprocess.Popen([sys.executable,*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    samples=[]
    while p.poll() is None:
        samples.append(hardware_probe.thread_count(p.pid))
        time.sleep(.01)
    stdout,stderr=p.communicate()
    if p.returncode:
        raise RuntimeError(stdout+stderr)
    return {'command':[sys.executable,*args],'peak_sampled_os_threads':max(samples,default=0),
            'samples':len(samples),'stdout':stdout,'stderr':stderr}


def main():
    r=Path('runs')
    result=json.loads((r/'supplement/results.json').read_text())
    if not result['promote']:
        print('No validation improvement: v1 retained; no new test evaluation.')
        return
    if (r/'freeze-v2.json').exists():
        raise RuntimeError('V2 already frozen; do not silently retest or replace.')
    source=r/'supplement'/result['best_challenger']['name']
    dest=r/'model_bundle_v2'
    shutil.copytree(source,dest)
    vb,va=load_inputs('validation')
    y=labels(vb,'validation')
    from submission import predict
    choices=predict(vb,va,dest)
    correct=int(y[np.arange(len(vb)),choices].sum())
    assert correct==result['best_challenger']['correct'] and correct>result['original_correct']
    evidence={'name':'v2-'+result['best_challenger']['name'],'validation_correct':correct,
        'validation_n':len(vb),'validation_accuracy':100*correct/len(vb),'selection_source':'supplement/results.json',
        'post_test_exposure':True,'decision':'Keep v2 regardless of final test comparison; no test-based rollback.',
        'thread_env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        'assets':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in dest.iterdir()},
        'sources':{p.name:sha(p) for p in Path('.').glob('*.py')},
        'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    evidence['asset_bytes']=sum(v['bytes'] for v in evidence['assets'].values())
    assert evidence['asset_bytes']<=16*2**20
    (r/'freeze-v2.json').write_text(json.dumps(evidence,indent=2))
    (r/'supplement/uncertainty-v2.json').write_text(json.dumps(uncertainty(vb,va,y,dest),indent=2))
    for split in ('validation','test'):
        tick=time.perf_counter()
        pred=observed_run(['predict.py','--method','submission.py','--checkpoint',str(dest),
             '--split',split,'--output',str(r/f'{split}-v2-predictions.json')])
        score=observed_run(['score_predictions.py','--split',split,'--predictions',str(r/f'{split}-v2-predictions.json'),
             '--output',str(r/f'{split}-v2-score.json')])
        evidence[f'{split}_total_seconds']=time.perf_counter()-tick
        evidence[f'{split}_processes']=[pred,score]
        print(score['stdout'],flush=True)
    evidence['backend_info_parent']=threadpool_info()
    (r/'reproduction-v2.json').write_text(json.dumps(evidence,indent=2))
    (r/'active_submission.json').write_text(json.dumps({'version':'v2','checkpoint':'runs/model_bundle_v2',
         'method':'submission.py','score':'runs/test-v2-score.json','freeze':'runs/freeze-v2.json',
         'selection':'validation-only; known prior v1 test disclosed; no test-based rollback'},indent=2))
    print(json.dumps({'asset_bytes':evidence['asset_bytes'],'test_total_seconds':evidence['test_total_seconds'],
        'peak_sampled_threads':max(p['peak_sampled_os_threads'] for p in evidence['test_processes'])}))


if __name__=='__main__':
    main()
