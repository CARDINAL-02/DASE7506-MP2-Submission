import sys, time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask, fit_logit
C=Path(__file__).resolve().parent/'cache'
d=np.load(C/'train_base.npz'); dv=np.load(C/'validation_base.npz')
Et=d['emb'].reshape(-1,1024); Ev=dv['emb'].reshape(-1,1024)
yt=d['y']; ht=d['hard']; yv=dv['y']; hv=dv['hard']
V2t=np.load(C/'train_v2.npy').astype(np.float64); V2v=np.load(C/'validation_v2.npy').astype(np.float64)
vv=valid_mask('validation')
for k in (32,64,128,256):
    t=time.time(); p=PCA(n_components=k,random_state=0,whiten=False).fit(Et)
    Pt=p.transform(Et); Pv=p.transform(Ev); print('PCA',k,'fit %.1fs'%(time.time()-t),flush=True)
    Xt=np.concatenate([V2t,Pt],1); Xv=np.concatenate([V2v,Pv],1)
    sc=StandardScaler().fit(Xt); Xt=sc.transform(Xt).reshape(-1,4,Xt.shape[1]); Xv=sc.transform(Xv).reshape(-1,4,Xv.shape[1])
    for lam in (0.003,0.01,0.03,0.1):
        s,_=fit_logit(Xt,yt,Xv,lam); a,ha,n,hn=metrics(s,yv,hv,vv)
        print('  PCA%d lam=%g hard %.2f all %.2f'%(k,lam,ha,a),flush=True)
