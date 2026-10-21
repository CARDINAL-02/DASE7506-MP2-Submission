import hashlib,json,subprocess,sys,time,platform,numpy as np
from pathlib import Path
root=Path('.'); runs=root/'runs_linear'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sources=['train_v2.py','features_v2.py','submission.py','features.py','common.py','predict.py','score_predictions.py','test_selection.py']
frozen={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'sources':{s:sha(root/s) for s in sources},
        'assets':{str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in (runs/'model_bundle').iterdir() if p.is_file()},
        'config':json.loads((runs/'model_bundle'/'config.json').read_text())}
frozen['asset_bytes']=sum(v['bytes'] for v in frozen['assets'].values())
(runs/'freeze.json').write_text(json.dumps(frozen,indent=2))
print(json.dumps({k:v for k,v in frozen.items() if k not in ('sources','assets')},indent=2))
def run(args):
    t=time.perf_counter(); p=subprocess.run(args,capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    return time.perf_counter()-t,p.stdout
py=sys.executable
t1,o1=run([py,'predict.py','--method','submission.py','--checkpoint','runs_linear/model_bundle','--split','test','--output','runs_linear/test-predictions.json'])
t2,o2=run([py,'score_predictions.py','--split','test','--predictions','runs_linear/test-predictions.json','--output','runs_linear/test-score.json'])
print('PREDICT',o1.strip()); print('SCORE',o2.strip()); print('TOTAL_SECONDS',t1+t2)
(runs/'reproduction.json').write_text(json.dumps({'test_total_seconds':t1+t2,'predict_seconds':t1,'score_seconds':t2,
 'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'threads':4}},indent=2))
