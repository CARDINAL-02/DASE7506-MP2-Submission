import sys, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import load, metrics, valid_mask, fit_logit
C=Path(__file__).resolve().parent/'cache'
S=dict(np.load(C/'ens_scores.npz'))
d=np.load(C/'validation_base.npz'); y=d['y']; h=d['hard']; vv=valid_mask('validation')
def z(s):
    s=s.reshape(-1); return ((s-s.mean())/(s.std()+1e-9)).reshape(-1,4)
def ev(name,keys,w=None):
    if w is None: w=[1]*len(keys)
    s=sum(wi*z(S[k]) for wi,k in zip(w,keys))/sum(w)
    a,ha,n,hn=metrics(s,y,h,vv); print('%-45s hard %.2f all %.2f (%d)'%(name,ha,a,hn))
# add structure and v2emb lam variants
for name,sp,lam in [('struct','structure',1e-5),('v2e003','v2emb',0.003),('v2e005','v2emb',0.005),('v2e02','v2emb',0.02),('v2e05','v2emb',0.05)]:
    A=load('train',sp)[0] if sp!='v2emb' else None
    if sp=='v2emb':
        Xt=np.load(C/'train_v2.npy').astype(np.float64); Xv=np.load(C/'validation_v2.npy').astype(np.float64)
        et=np.load(C/'train_base.npz')['emb'].reshape(-1,1024); evv=np.load(C/'validation_base.npz')['emb'].reshape(-1,1024)
        Xt=np.concatenate([Xt,et],1); Xv=np.concatenate([Xv,evv],1)
        sc=StandardScaler().fit(Xt); Xt=sc.transform(Xt).reshape(-1,4,Xt.shape[1]); Xv=sc.transform(Xv).reshape(-1,4,Xv.shape[1])
    else:
        B=load('validation',sp)[0]; sc=StandardScaler().fit(A.reshape(-1,A.shape[2]))
        Xt=sc.transform(A.reshape(-1,A.shape[2])).reshape(-1,4,A.shape[2]); Xv=sc.transform(B.reshape(-1,B.shape[2])).reshape(-1,4,B.shape[2])
    s,_=fit_logit(Xt,np.load(C/'train_base.npz')['y'],Xv,lam); S[name]=s
    print(name,'hard %.2f'%metrics(s,y,h,vv)[1])
base=['lin','mlpavg','emb']; b=[3,2,1]
ev('base 3/2/1',base,b)
for extra in [['v1'],['struct'],['v2e005'],['v1','struct'],['v1','v2e005'],['struct','v2e005'],['v1','struct','v2e005']]:
    ev('base+'+'+'.join(extra),base+extra,b+[1]*len(extra))
ev('base*2 + extra1',base+['v1','struct'], [6,4,2,1,1])
