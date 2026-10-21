import sys, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
from models2 import build
from models4 import fit_logit_w
C=Path(__file__).resolve().parent/'cache'
def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb'); X,y,h=Xt,yt,ht; vv=valid_mask('train')
    par=np.load(C/'train_parsed.npy',allow_pickle=True)
    ndist=np.array([len({v for v in r if v is not None}) for r in par])
    counts=np.array([[sum(1 for x in r if x==v) for v in r] for r in par]); maxgrp=counts.max(1)
    variants={'plain':np.ones(len(y)),'d05':1+.5*(ndist-1),'inv2':1+2*(maxgrp<2)}
    gkf=GroupKFold(n_splits=5); folds=list(gkf.split(X,y,np.arange(len(y))))
    out={}
    for name,w in variants.items():
        for lam in (0.003,0.01,0.02,0.05,0.1):
            acc=[]; tot=cor=0
            for tr,te in folds:
                sc=StandardScaler().fit(X[tr].reshape(-1,X.shape[2]))
                A=sc.transform(X[tr].reshape(-1,X.shape[2])).reshape(-1,4,X.shape[2])
                B=sc.transform(X[te].reshape(-1,X.shape[2])).reshape(-1,4,X.shape[2])
                th=fit_logit_w(A,y[tr],lam,w[tr])
                s=(B.reshape(-1,B.shape[2])@th).reshape(-1,4)
                a,ha,n,hn=metrics(s,y[te],h[te],vv[te]); acc.append(ha); tot+=int(h[te].sum()); cor+=n
            print('%-5s lam=%-5g mean %.2f sd %.2f pooled %d/%d %.2f'%(name,lam,np.mean(acc),np.std(acc),cor,tot,100*cor/tot),flush=True)
            out[f'{name}-{lam}']={(str(k),str(v)) for k,v in [(0,0)]}
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
