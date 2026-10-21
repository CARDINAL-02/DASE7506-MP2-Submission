"""Separate v3 freeze and one final evaluation; validation determines promotion, never test."""
from finalize_v2 import observed_run,sha
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import shutil
import time


def main():
    r=Path('runs')
    result=json.loads((r/'v3/results.json').read_text())
    if not result['promote']:
        print('No validation promotion; v2 retained, no new test evaluation.')
        return
    if (r/'freeze-v3.json').exists():
        raise RuntimeError('Already frozen v3; no silent retest or replacement.')
    dest=r/'model_bundle_v3'
    shutil.copytree(r/'v3'/result['best']['name'],dest)
    assets={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in dest.rglob('*') if p.is_file()}
    size=sum(v['bytes'] for v in assets.values())
    assert size<=16*2**20
    tickval=time.perf_counter()
    val_prediction=observed_run(['predict.py','--method','v3_method.py','--checkpoint',str(dest),'--split','validation',
        '--output',str(r/'validation-v3-predictions.json')])
    val_score=observed_run(['score_predictions.py','--split','validation','--predictions',str(r/'validation-v3-predictions.json'),
        '--output',str(r/'validation-v3-score.json')])
    val_seconds=time.perf_counter()-tickval
    assert val_seconds<30, 'Conservative local validation runtime exceeds budget; do not test.'
    score=json.loads((r/'validation-v3-score.json').read_text())
    assert score['correct']==result['best']['correct'] and score['correct']>743
    audit=observed_run(['predict.py','--method','v3_method.py','--checkpoint',str(dest),'--split','audit',
        '--data-dir',str(r/'audit_inputs'),'--output',str(r/'audit-v3-predictions.json')])
    choices=json.loads((r/'audit-v3-predictions.json').read_text())['selected']
    assert len(choices)==7 and all(type(i)is int and 0<=i<4 for i in choices)
    frozen={'version':'v3','selected':result['best'],'target_test_accuracy':83.5,
        'prior_v1_v2_test_known':True,'test_policy':'Keep validation-selected v3 regardless of test; no further search or rollback.',
        'selection_source':'runs/v3/results.json','asset_bytes':size,'assets':assets,
        'sources':{p.name:sha(p) for p in Path('.').glob('*.py')},
        'thread_env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'validation_total_seconds':val_seconds,'validation_processes':[val_prediction,val_score],'audit_process':audit}
    (r/'freeze-v3.json').write_text(json.dumps(frozen,indent=2))
    tick=time.perf_counter()
    pred=observed_run(['predict.py','--method','v3_method.py','--checkpoint',str(dest),'--split','test',
        '--output',str(r/'test-v3-predictions.json')])
    scored=observed_run(['score_predictions.py','--split','test','--predictions',str(r/'test-v3-predictions.json'),
        '--output',str(r/'test-v3-score.json')])
    frozen['test_total_seconds']=time.perf_counter()-tick
    frozen['test_processes']=[pred,scored]
    t=json.loads((r/'test-v3-score.json').read_text())
    frozen['target_met']=t['accuracy']>=83.5
    (r/'reproduction-v3.json').write_text(json.dumps(frozen,indent=2))
    (r/'active_submission.json').write_text(json.dumps({'version':'v3','checkpoint':'runs/model_bundle_v3',
        'method':'v3_method.py','score':'runs/test-v3-score.json','freeze':'runs/freeze-v3.json',
        'selection':'36-candidate validation-only budget, prior tests and public leaderboard exposure disclosed; no test rollback'},indent=2))
    print(scored['stdout'])
    print(json.dumps({'target_met':frozen['target_met'],'test_total_seconds':frozen['test_total_seconds'],
        'asset_bytes':size,'peak_sampled_os_threads':max(p['peak_sampled_os_threads'] for p in frozen['test_processes'])}))


if __name__=='__main__':
    main()
