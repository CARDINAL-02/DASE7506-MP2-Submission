"""Post-test integrity/reproduction audit. Does not select, tune, or replace the frozen model."""
import hardware_probe
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from finalize_v2 import observed_run
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    r=Path('runs_linear')
    out=r/'review'
    out.mkdir(exist_ok=True)
    if (out/'integrity.json').exists():
        raise RuntimeError('Review already recorded. Preserve it rather than silently overwrite.')
    f=json.loads((r/'freeze.json').read_text())
    assert all(sha(Path(n))==h for n,h in f['sources'].items())
    assert all(sha(Path(n))==v['sha256'] for n,v in f['assets'].items())
    bundle=r/'model_bundle'
    with np.load(bundle/'weights.npz',allow_pickle=False) as w:
        array_bytes=sum(w[k].nbytes for k in w.files)
    with zipfile.ZipFile(bundle/'weights.npz') as z:
        entry_bytes=sum(v.file_size for v in z.infolist())
    config_bytes=(bundle/'config.json').stat().st_size
    stored=sum(p.stat().st_size for p in bundle.iterdir() if p.is_file())
    assert entry_bytes+config_bytes<=16*2**20
    record={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'purpose':'Post-test reproduction and integrity addendum; not a replacement pre-test freeze.',
        'original_freeze_sha256':sha(r/'freeze.json'),'original_freeze_sources_match':True,
        'original_freeze_assets_match':True,'sources':{n:sha(Path(n)) for n in
            ('submission.py','features_v2.py','features.py','train_linear_v2.py','train_v2.py','common.py',
             'predict.py','score_predictions.py','prepare_final_submission.py')},
        'assets':f['assets'],'stored_files_bytes':stored,'learned_array_payload_bytes':array_bytes,
        'configuration_bytes':config_bytes,'conservative_uncompressed_asset_bytes':entry_bytes+config_bytes,
        'measurement_rule':'Uncompressed NPZ member sizes including NPY headers, plus JSON configuration.',
        'thread_env':{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    tick=time.perf_counter()
    pred=observed_run(['predict.py','--method','submission.py','--checkpoint',str(bundle),'--split','test',
        '--output',str(out/'test-predictions.json')])
    score=observed_run(['score_predictions.py','--split','test','--predictions',str(out/'test-predictions.json'),
        '--output',str(out/'test-score.json')])
    record['test_total_seconds']=time.perf_counter()-tick
    record['test_processes']=[pred,score]
    s=json.loads((out/'test-score.json').read_text())
    assert (s['correct'],s['n'])==(391,467)
    saved=json.loads((r/'test-predictions.json').read_text())
    actual=json.loads((out/'test-predictions.json').read_text())
    assert actual['selected']==saved['selected'] and actual['ids']==saved['ids']
    record['official_score']=s
    record['saved_predictions_match']=True
    record['audit_process']=observed_run(['predict.py','--method','submission.py','--checkpoint',str(bundle),
        '--split','audit','--data-dir','runs/audit_inputs','--output',str(out/'audit-predictions.json')])
    audit=json.loads((out/'audit-predictions.json').read_text())['selected']
    assert len(audit)==7 and all(type(i)is int and 0<=i<4 for i in audit)
    record['audit_n']=7
    # Deterministic train-only reconstruction, without calling main() or overwriting saved assets/metrics.
    import train_linear_v2 as train
    from threadpoolctl import threadpool_limits
    captured={}
    actual_minimize=train.minimize
    def tracked_minimize(*args,**kwargs):
        result=actual_minimize(*args,**kwargs)
        captured.update(success=bool(result.success),iterations=int(result.nit),message=str(result.message))
        return result
    train.minimize=tracked_minimize
    tick=time.perf_counter()
    with threadpool_limits(limits=4):
        b,a=train.load_inputs('train')
        y=train.labels(b,'train')
        emb=a['embeddings']
        x=np.concatenate([train.features_v2.extract(b,emb,train.parsed(b)),emb.reshape(-1,1024)],axis=1).astype(np.float64)
        sc=train.StandardScaler().fit(x)
        x=sc.transform(x).reshape(len(b),4,-1)
        weight=train.fit(x,y,train.LAMBDA)
    with np.load(bundle/'weights.npz',allow_pickle=False) as w:
        exact={'mean':bool(np.array_equal(sc.mean_,w['mean'])),'scale':bool(np.array_equal(sc.scale_,w['scale'])),
               'weights':bool(np.array_equal(weight,w['lin_w']))}
    assert all(exact.values())
    record['train_only_reconstruction']={'exact_arrays':exact,'seconds':time.perf_counter()-tick,
        'optimizer_status_on_reconstruction':captured,'training_threads':4,'original_assets_overwritten':False}
    record['note']='Reproducing the fixed final model after test is not a new method-selection trial.'
    (out/'integrity.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:record[k] for k in ('stored_files_bytes','conservative_uncompressed_asset_bytes',
        'test_total_seconds','official_score','train_only_reconstruction')},indent=2))


if __name__=='__main__':
    main()
