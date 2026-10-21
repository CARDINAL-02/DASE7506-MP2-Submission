import sys, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask, fit_logit
C=Path(__file__).resolve().parent/'cache'
d=np.load(C/'train_base.npz'); dv=np.load(C/'validation_base.npz')
Et=d['emb'].reshape(-1,1024); Ev=dv['emb'].reshape(-1,1024)
yt,ht,yv,hv=d['y'],d['hard'],dv['y'],dv['hard']
Vt=np.load(C/'train_v2.npy').reshape(-1,54); Vv=np.load(C/'validation_v2.npy').reshape(-1,54)
vv=valid_mask('validation')
groups=[('all',list(range(54))),('base22',list(range(22))),('base+agree',list(range(24))),
        ('+overlap',list(range(29))),('+eq',list(range(35))),('+answer',list(range(44))),
        ('no-lex',list(range(48))),('no-eq',[i for i in range(54) if not 29<=i<=34]),
        ('no-answer',[i for i in range(54) if not 38<=i<=43])]
for name,cols in groups:
    Xt=np.concatenate([Vt[:,cols],Et],1); Xv=np.concatenate([Vv[:,cols],Ev],1)
    sc=StandardScaler().fit(Xt); Xt=sc.transform(Xt).reshape(-1,4,Xt.shape[1]); Xv=sc.transform(Xv).reshape(-1,4,Xv.shape[1])
    best=(-1,None)
    for lam in (0.01,):
        s,_=fit_logit(Xt,yt,Xv,lam); a,ha,n,hn=metrics(s,yv,hv,vv)
        if ha>best[0]: best=(ha,lam,a)
    print('%-18s D=%4d best lam=%-5g hard %.2f all %.2f'%(name,Xt.shape[2],best[1],best[0],best[2]),flush=True)
