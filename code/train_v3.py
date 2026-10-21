"""Fixed 36-candidate v3 grid. Fits train, selects validation, never loads test."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import load_inputs
from features import choose,parsed
from train import labels,metrics
from supplement import objective
from supplement_method import scores as previous_scores
from v3_features import v3_features
from v3_method import scores

OUT=Path('runs/v3')


def export_tree(model,path,mode):
    path.mkdir(parents=True,exist_ok=True)
    values={}
    is_hist=isinstance(model,HistGradientBoostingClassifier)
    trees=[p[0].nodes for p in model._predictors] if is_hist else [t.tree_ for t in model.estimators_]
    for i,tree in enumerate(trees):
        if is_hist:
            assert not tree['is_categorical'].any()
            fields={'feature':tree['feature_idx'].astype(np.int32),'threshold':tree['num_threshold'],
                'left':tree['left'].astype(np.int32),'right':tree['right'].astype(np.int32),
                'leaf':tree['is_leaf'].astype(bool),'value':tree['value']}
        else:
            prob=tree.value[:,0,1]/tree.value[:,0,:].sum(axis=1)
            fields={'feature':np.maximum(tree.feature,0).astype(np.int32),'threshold':tree.threshold,
                'left':tree.children_left.astype(np.int32),'right':tree.children_right.astype(np.int32),
                'leaf':tree.children_left<0,'value':prob/len(trees)}
        values.update({f'{i}_{k}':v for k,v in fields.items()})
    np.savez(path/'weights.npz',**values)
    (path/'config.json').write_text(json.dumps({'kind':'boost-v3' if is_hist else 'forest-v3',
        'features':mode,'trees':len(trees),'base':float(model._baseline_prediction.ravel()[0]) if is_hist else 0.},indent=2))


def save_linear(path,weight,bias,scaler):
    path.mkdir(parents=True,exist_ok=True)
    np.savez(path/'weights.npz',mean=scaler.mean_,scale=scaler.scale_,weight=weight,bias=bias)
    (path/'config.json').write_text(json.dumps({'kind':'linear-v3','features':'dense'}))


def main():
    if (OUT/'results.json').exists():
        raise RuntimeError('Fixed experiment complete; do not overwrite or extend by test feedback.')
    OUT.mkdir(parents=True,exist_ok=True)
    tickall=time.perf_counter()
    tb,ta=load_inputs('train')
    vb,va=load_inputs('validation')
    yt,yv=labels(tb,'train'),labels(vb,'validation')
    at,av=parsed(tb),parsed(vb)
    hard=np.array([len({v for v in a if v is not None})>=2 for a in av])
    xt={m:v3_features(tb,ta,m) for m in ('compact','dense')}
    xv={m:v3_features(vb,va,m) for m in ('compact','dense')}
    assert all(np.isfinite(v).all() for v in [*xt.values(),*xv.values()])
    oldt,oldv=previous_scores(tb,ta,Path('runs/model_bundle_v2')),previous_scores(vb,va,Path('runs/model_bundle_v2'))
    scale=lambda s:max(float(np.std(s-s.mean(axis=1,keepdims=True))),1e-8)
    old_scale=scale(oldt)
    records=[]
    plan=next(p for p in (Path(__file__).resolve().parent.parent/'V3_PLAN.md',
                          Path(__file__).resolve().parent.parent.parent/'V3_PLAN.md') if p.is_file())
    (OUT/'protocol.json').write_text(json.dumps({'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'dimensions':{k:v.shape[1] for k,v in xt.items()},
        'target_test_accuracy':83.5,'test_access':False,'prior_tests_known':True,'old_scale_train':old_scale},indent=2))
    def record(name,seconds,library_score=None,**extra):
        path=OUT/name
        sv=scores(vb,va,path,cache=xv)
        choice=choose(sv,av)
        if library_score is not None:
            np.testing.assert_allclose(sv,library_score,rtol=1e-6,atol=1e-7)
            assert choice==choose(library_score,av)
        size=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        item={'name':name,**metrics(choice,yv,hard),'asset_bytes':size,'fit_seconds':seconds,**extra}
        records.append(item)
        np.savez(OUT/f'{name}-validation.npz',scores=sv,selected=np.array(choice))
        (OUT/'partial.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(item),flush=True)
        return sv
    for mode in ('compact','dense'):
        for leaves,iterations,l2 in ((7,200,5.),(15,200,10.),(31,150,20.)):
            name=f'hgb-{mode}-{leaves}'
            tick=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                model=HistGradientBoostingClassifier(max_leaf_nodes=leaves,max_iter=iterations,l2_regularization=l2,
                    learning_rate=.08,min_samples_leaf=30,early_stopping=False,random_state=20261021).fit(xt[mode],yt.ravel())
            seconds=time.perf_counter()-tick
            export_tree(model,OUT/name,mode)
            record(name,seconds,model._raw_predict(xv[mode]).reshape(-1,4),warnings=[str(w.message) for w in caught])
    for trees,depth,leaf in ((64,8,10),(128,12,15)):
        name=f'extra-compact-{depth}'
        tick=time.perf_counter()
        model=ExtraTreesClassifier(n_estimators=trees,max_depth=depth,min_samples_leaf=leaf,
            max_features=1.,random_state=20261021,n_jobs=1).fit(xt['compact'],yt.ravel())
        export_tree(model,OUT/name,'compact')
        record(name,time.perf_counter()-tick,model.predict_proba(xv['compact'])[:,1].reshape(-1,4))
    # Fixed fusions use train-input score scale only; validation does not fit any normalization.
    trees=[r['name'] for r in records]
    for member in trees:
        news=scores(tb,ta,OUT/member,cache=xt)
        new_scale=scale(news)
        for weight in (.25,.5,.75):
            name=f'fusion-{member}-{weight:g}'
            path=OUT/name
            path.mkdir(exist_ok=True)
            shutil.copytree(OUT/member,path/'tree',dirs_exist_ok=True)
            shutil.copytree('runs/model_bundle_v2',path/'v2',dirs_exist_ok=True)
            (path/'config.json').write_text(json.dumps({'kind':'fusion-v3','weight':weight,'old_scale':old_scale,'new_scale':new_scale},indent=2))
            record(name,0.,member=member,weight=weight,new_scale_train=new_scale)
    scaler=StandardScaler().fit(xt['dense'])
    tx,vx=scaler.transform(xt['dense']),scaler.transform(xv['dense'])
    for c in (.01,.1):
        name=f'linear-dense-{c:g}'
        tick=time.perf_counter()
        model=LogisticRegression(C=c,max_iter=600).fit(tx,yt.ravel())
        save_linear(OUT/name,model.coef_.ravel(),float(model.intercept_[0]),scaler)
        record(name,time.perf_counter()-tick,model.decision_function(vx).reshape(-1,4))
    valid=np.array([[v is not None for v in a] for a in at])
    target=yt/yt.sum(axis=1,keepdims=True)
    for penalty in (.01,.1):
        name=f'listwise-dense-{penalty:g}'
        tick=time.perf_counter()
        fit=minimize(objective,np.zeros(tx.shape[1]),args=(tx.reshape(-1,4,tx.shape[1]),target,valid,penalty),
            method='L-BFGS-B',jac=True,options={'maxiter':300,'ftol':1e-10})
        save_linear(OUT/name,fit.x,0.,scaler)
        record(name,time.perf_counter()-tick,(vx@fit.x).reshape(-1,4),success=bool(fit.success),iterations=int(fit.nit),message=str(fit.message))
    assert len(records)==36
    eligible=[r for r in records if r['asset_bytes']<=16*2**20 and r.get('success',True)]
    best=max(eligible,key=lambda r:(r['correct'],r['hard_accuracy'],-r['asset_bytes']))
    result={'records':records,'best':best,'promote':best['correct']>743,'baseline_correct':743,
        'candidate_count':36,'seconds':time.perf_counter()-tickall,'test_access':False,'prior_tests_known':True,
        'selection_rule':'validation correct, hard accuracy, smaller assets, earlier order; promote strictly above743'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print('SELECTED',json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
