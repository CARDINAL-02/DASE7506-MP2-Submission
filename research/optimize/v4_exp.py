import sys, numpy as np, time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask, fit_logit
from mlp_listwise import train as train_mlp, score as mlp_score
C=Path(__file__).resolve().parent/'cache'
d=np.load(C/'train_base.npz'); dv=np.load(C/'validation_base.npz')
Et=d['emb']; Ev=dv['emb']
yt,ht,yv,hv=d['y'],d['hard'],dv['y'],dv['hard']
Vt=np.load(C/'train_v2.npy').reshape(-1,4,54); Vv=np.load(C/'validation_v2.npy').reshape(-1,4,54)
vv=valid_mask('validation')
def build(V,E):
    mean=V.mean(1,keepdims=True); rel=V-mean; rep=np.repeat(mean,4,axis=1)
    rank=np.zeros_like(V)
    for c in range(V.shape[2]):
        order=V[:,:,c].argsort(axis=1).argsort(axis=1)
        rank[:,:,c]=order/3.0
    return np.concatenate([V,rel,rep,rank,E],axis=2)
Xt=build(Vt,Et); Xv=build(Vv,Ev); print('D',Xt.shape[2],flush=True)
sc=StandardScaler().fit(Xt.reshape(-1,Xt.shape[2]))
Xt64=sc.transform(Xt.reshape(-1,Xt.shape[2])).reshape(-1,4,Xt.shape[2])
Xv64=sc.transform(Xv.reshape(-1,Xv.shape[2])).reshape(-1,4,Xv.shape[2])
Xt32=Xt64.astype(np.float32); Xv32=Xv64.astype(np.float32)
for lam in (0.003,0.01,0.03,0.1):
    s,_=fit_logit(Xt64,yt,Xv64,lam); a,ha,n,hn=metrics(s,yv,hv,vv)
    print('linear lam',lam,'hard %.2f all %.2f'%(ha,a),flush=True)
for seed in (0,1,2):
    t=time.time(); W,ep,ha=train_mlp(Xt32,yt,Xv32,yv,hv,vv,H=128,wd=3e-3,lr=1e-3,epochs=40,bs=512,seed=seed,verbose=False)
    s=mlp_score(Xv32,W); a,ha2,n,hn=metrics(s,yv,hv,vv)
    print('mlp seed',seed,'ep',ep,'hard %.2f all %.2f (%.0fs)'%(ha2,a,time.time()-t),flush=True)
