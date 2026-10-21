import hashlib, json, subprocess, sys, time, platform
from pathlib import Path
root=Path('.'); runs=root/'runs_v2'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sources=['train_v2.py','features_v2.py','submission.py','features.py','common.py','predict.py','score_predictions.py','test_selection.py']
frozen={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'sources':{s:sha(root/s) for s in sources},
        'assets':{str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in (runs/'model_bundle').iterdir() if p.is_file()},
        'train_metrics':json.loads((runs/'train_metrics.json').read_text()),
        'config':json.loads((runs/'model_bundle'/'config.json').read_text())}
frozen['asset_bytes']=sum(v['bytes'] for v in frozen['assets'].values())
(runs/'freeze.json').write_text(json.dumps(frozen,indent=2))
print(json.dumps({k:v for k,v in frozen.items() if k not in ('sources','assets')},indent=2))
def run(args):
    t=time.perf_counter(); p=subprocess.run(args,capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    return time.perf_counter()-t, p.stdout
py=sys.executable
t1,o1=run([py,'predict.py','--method','submission.py','--checkpoint','runs_v2/model_bundle','--split','test','--output','runs_v2/test-predictions.json'])
t2,o2=run([py,'score_predictions.py','--split','test','--predictions','runs_v2/test-predictions.json','--output','runs_v2/test-score.json'])
print('PREDICT',o1.strip()); print('SCORE',o2.strip()); print('TOTAL_SECONDS',t1+t2)
rec={'test_total_seconds':t1+t2,'predict_seconds':t1,'score_seconds':t2,
     'environment':{'python':sys.version,'numpy':__import__('numpy').__version__,'platform':platform.platform(),'threads':4}}
(runs/'reproduction.json').write_text(json.dumps(rec,indent=2))
