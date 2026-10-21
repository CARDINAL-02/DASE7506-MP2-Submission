"""Train the frozen v2 listwise linear selector. Reads only train/validation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import load_inputs, read_jsonl, DATA, parse_answer
from features import parsed
import features_v2

LAMBDA=0.01

def labels(bank, split):
    refs=read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert [r['id'] for r in refs]==[r['id'] for r in bank]
    return np.array([[int(v is not None and v==parse_answer(g['answer'])) for v in a]
                     for a,g in zip(parsed(bank),refs)],dtype=np.int64)

def fit(X,y,lam,maxiter=400):
    K=X.shape[2]; Xf=X.reshape(-1,K); mask=y>0
    def obj(w):
        s=(Xf@w).reshape(-1,4); lse=logsumexp(s,axis=1)
        sp=np.where(mask,s,-1e30); lse_pos=logsumexp(sp,axis=1)
        loss=np.mean(lse-lse_pos)+0.5*lam*w@w
        q=np.exp(s-lse[:,None]); p=np.exp(sp-lse_pos[:,None]); p=np.where(mask,p,0.)
        grad=Xf.T@(q-p).ravel()/len(y)+lam*w
        return loss,grad
    return minimize(obj,np.zeros(K),jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-10}).x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=Path('runs_linear/model_bundle')); a=ap.parse_args()
    t=time.perf_counter()
    train,ta=load_inputs('train'); val,va=load_inputs('validation')
    yt,yv=labels(train,'train'),labels(val,'validation')
    ans=parsed(val); hard=np.array([len({v for v in r if v is not None})>=2 for r in ans])
    Xt=np.concatenate([features_v2.extract(train,ta['embeddings'],parsed(train)),ta['embeddings'].reshape(-1,1024)],1).astype(np.float64)
    Xv=np.concatenate([features_v2.extract(val,va['embeddings'],ans),va['embeddings'].reshape(-1,1024)],1).astype(np.float64)
    sc=StandardScaler().fit(Xt); Xt=sc.transform(Xt).reshape(-1,4,Xt.shape[1]); Xv=sc.transform(Xv).reshape(-1,4,Xv.shape[1])
    w=fit(Xt,yt,LAMBDA)
    s=(Xv.reshape(-1,Xv.shape[2])@w).reshape(-1,4)
    valid=np.array([[v is not None for v in r] for r in ans])
    s[~valid]=-np.inf; ch=s.argmax(1); ok=yv[np.arange(len(yv)),ch].astype(bool)
    metrics={'validation_accuracy':float(ok.mean()*100),'validation_hard_accuracy':float(ok[hard].mean()*100),
             'correct':int(ok.sum()),'n':len(yv),'hard_correct':int(ok[hard].sum()),'hard_n':int(hard.sum()),
             'train_seconds':time.perf_counter()-t,'lambda':LAMBDA}
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output/'weights.npz',mean=sc.mean_,scale=sc.scale_,lin_w=w)
    (a.output/'config.json').write_text(json.dumps({'kind':'linear_v2','features':'v2emb','lambda':LAMBDA,
        'hard_subset_definition':'>=2 distinct non-None parsed answers'},indent=2))
    (Path('runs_linear')/'train_metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=4): main()
