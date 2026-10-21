"""Grouped 5-fold CV on train: compare feature sets and regularization."""
import sys, json, time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models1 import load, fit_logit, metrics, valid_mask
from models2 import build

def dataset(kind):
    if kind=='v2emb':
        Xt,yt,ht,Xv,yv,hv=build('v2emb'); return Xt,yt,ht
    Xt,yt,ht=load('train','combined'); return Xt,yt,ht

def main():
    for kind in ('combined','v2emb'):
        X,y,h=dataset(kind)
        vv=valid_mask('train')
        groups=np.repeat(np.arange(len(y)),1)  # one group per question; X is (N,4,D)
        gkf=GroupKFold(n_splits=5)
        print('===',kind,'X',X.shape,flush=True)
        lams=(0.003,0.01,0.03,0.1,0.2,0.3) if kind=='combined' else (0.003,0.01,0.03,0.1,0.2,0.3)
        res={l:[] for l in lams}
        for fold,(tr,te) in enumerate(gkf.split(X,y,groups)):
            sc=StandardScaler().fit(X[tr].reshape(-1,X.shape[2]))
            Xtr=sc.transform(X[tr].reshape(-1,X.shape[2])).reshape(-1,4,X.shape[2])
            Xte=sc.transform(X[te].reshape(-1,X.shape[2])).reshape(-1,4,X.shape[2])
            for lam in lams:
                s,_=fit_logit(Xtr,y[tr],Xte,lam,maxiter=300)
                a,ha,n,hn=metrics(s,y[te],h[te],vv[te])
                res[lam].append((a,ha,hn,int(h[te].sum())))
            print('fold',fold,'done',flush=True)
        for lam in lams:
            arr=np.array([r[1] for r in res[lam]])
            tot_hn=sum(r[3] for r in res[lam]); tot_hn_correct=sum(r[2] for r in res[lam])
            print('lam %-6g hard mean %.2f sd %.2f | pooled %d/%d %.2f'%(lam,arr.mean(),arr.std(),tot_hn_correct,tot_hn,100*tot_hn_correct/tot_hn),flush=True)
        (Path(__file__).parent/f'cv_{kind}.json').write_text(json.dumps({str(k):[[float(x) for x in r] for r in v] for k,v in res.items()},indent=1))
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
