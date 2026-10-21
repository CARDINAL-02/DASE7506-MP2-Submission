"""Pre-registered validation-only supplemental experiments. No test access."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '4'
import json
from pathlib import Path
import shutil
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import load_inputs
from features import extract, parsed, choose, majority
from supplement_features import restricted
from supplement_method import scores
from train import labels, metrics, export

OUT = Path('runs/supplement')


def objective(w,x,target,valid,penalty):
    z = x@w
    z = np.where(valid,z,-1e12)
    logp = z-logsumexp(z,axis=1,keepdims=True)
    loss = -(target*logp).sum(axis=1).mean()+penalty*np.dot(w,w)/2
    diff = np.exp(logp)-target
    grad = np.einsum('ncd,nc->d',x,diff)/len(x)+penalty*w
    return float(loss),grad


def save_linear(folder,w,scaler):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez(folder/'weights.npz',mean=scaler.mean_,scale=scaler.scale_,w0=w.reshape(-1,1),b0=np.zeros(1))
    (folder/'config.json').write_text(json.dumps({'kind':'linear','features':'combined','layers':1}))


def main():
    if (OUT/'results.json').exists():
        raise RuntimeError('Supplement already complete. Preserve results; do not silently search again.')
    OUT.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    tb,ta=load_inputs('train')
    vb,va=load_inputs('validation')
    yt,yv=labels(tb,'train'),labels(vb,'validation')
    at,av=parsed(tb),parsed(vb)
    hard=np.array([len({a for a in row if a is not None})>=2 for row in av])
    records=[]
    def record(name,folder,seconds,role,**extra):
        s=scores(vb,va,folder)
        choices=choose(s,av)
        assets=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        item={'name':name,'role':role,**metrics(choices,yv,hard),'asset_bytes':assets,'fit_seconds':seconds,**extra}
        records.append(item)
        np.savez(OUT/f'{name}-validation.npz',scores=s,selected=np.array(choices))
        (OUT/'partial.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(item),flush=True)
    for mode in ('answer','surface','answer_surface','answer_surface_embedding'):
        if mode == 'answer_surface_embedding':
            xt=np.concatenate([restricted(tb,'answer_surface'),extract(tb,ta,'embedding')],axis=1)
            xv=np.concatenate([restricted(vb,'answer_surface'),extract(vb,va,'embedding')],axis=1)
        else:
            xt,xv=restricted(tb,mode),restricted(vb,mode)
        scaler=StandardScaler().fit(xt)
        tick=time.perf_counter()
        model=LogisticRegression(C=.01,max_iter=600).fit(scaler.transform(xt),yt.ravel())
        folder=OUT/mode
        export(folder,model,scaler,mode)
        record(mode,folder,time.perf_counter()-tick,'ablation',dimensions=xt.shape[1])
    rawt,rawv=extract(tb,ta,'combined').astype(float),extract(vb,va,'combined').astype(float)
    scaler=StandardScaler().fit(rawt)
    xt,xv=scaler.transform(rawt),scaler.transform(rawv)
    valid=np.array([[a is not None for a in row] for row in at])
    target=yt/yt.sum(axis=1,keepdims=True)
    for penalty in (.001,.01,.1):
        name=f'listwise-{penalty:g}'
        tick=time.perf_counter()
        fit=minimize(objective,np.zeros(xt.shape[1]),args=(xt.reshape(-1,4,xt.shape[1]),target,valid,penalty),
                     method='L-BFGS-B',jac=True,options={'maxiter':200,'ftol':1e-10})
        save_linear(OUT/name,fit.x,scaler)
        record(name,OUT/name,time.perf_counter()-tick,'challenger',success=bool(fit.success),iterations=int(fit.nit),message=str(fit.message))
    groupx=xt.reshape(-1,4,xt.shape[1])
    differences=[]
    for i in range(len(tb)):
        for pos in np.flatnonzero(yt[i]):
            for neg in np.flatnonzero((yt[i]==0)&valid[i]):
                differences.append(groupx[i,pos]-groupx[i,neg])
    diff=np.asarray(differences)
    pairx=np.concatenate([diff,-diff])
    pairy=np.concatenate([np.ones(len(diff)),np.zeros(len(diff))])
    for c in (.01,.1,1.):
        name=f'pairwise-{c:g}'
        tick=time.perf_counter()
        model=LogisticRegression(C=c,fit_intercept=False,max_iter=600).fit(pairx,pairy)
        save_linear(OUT/name,model.coef_.ravel(),scaler)
        record(name,OUT/name,time.perf_counter()-tick,'challenger',pairs=len(diff))
    tick=time.perf_counter()
    hardt=np.array([len({a for a in row if a is not None})>=2 for row in at])
    model=LogisticRegression(C=.01,max_iter=600).fit(xt,yt.ravel(),sample_weight=np.repeat(np.where(hardt,3.,1.),4))
    export(OUT/'weighted-lr',model,scaler,'combined')
    record('weighted-lr',OUT/'weighted-lr',time.perf_counter()-tick,'challenger')
    folder=OUT/'ensemble'
    folder.mkdir(exist_ok=True)
    members=['model_bundle',*[f'combined-mlp-0.01-s{s}' for s in (17,29,43)]]
    scales=[]
    for i,m in enumerate(members):
        shutil.copytree(Path('runs')/m,folder/f'm{i}',dirs_exist_ok=True)
        scales.append(float(np.std(scores(tb,ta,folder/f'm{i}'))))
    (folder/'config.json').write_text(json.dumps({'kind':'ensemble','members':[f'm{i}' for i in range(4)],'scales':scales},indent=2))
    record('ensemble',folder,0.,'challenger',training_logit_scales=scales)
    challengers=[r for r in records if r['role']=='challenger' and r.get('success',True)]
    best=max(challengers,key=lambda r:(r['accuracy'],r['hard_accuracy'],-r['asset_bytes']))
    original=json.loads(Path('runs/selection.json').read_text())['winner']
    result={'records':records,'best_challenger':best,'original_correct':original['correct'],
        'promote':best['correct']>original['correct'],'selection_rule':'strictly higher whole-validation correct count; ties hard accuracy, smaller assets, earlier order',
        'seconds':time.perf_counter()-start,'post_test_exposure':True,'test_access':False}
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print('DECISION',json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
