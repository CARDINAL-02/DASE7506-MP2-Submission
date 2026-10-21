import sys, json, itertools, time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import load, metrics, valid_mask
from models2 import build
from models4 import fit_logit_w, z
from mlp_listwise import train as train_mlp, score as mlp_score
C=Path(__file__).resolve().parent/'cache'

def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb')
    sc=StandardScaler().fit(Xt.reshape(-1,Xt.shape[2]))
    Xt=sc.transform(Xt.reshape(-1,Xt.shape[2])).astype(np.float32).reshape(Xt.shape)
    Xv=sc.transform(Xv.reshape(-1,Xv.shape[2])).astype(np.float32).reshape(Xv.shape)
    Xt64=Xt.astype(np.float64); Xv64=Xv.astype(np.float64)
    vv=valid_mask('validation')
    S={}
    th=fit_logit_w(Xt64,yt,0.01); S['lin']=(Xv64.reshape(-1,Xv.shape[2])@th).reshape(-1,4)
    print('lin',metrics(S['lin'],yv,hv,vv)[:2],flush=True)
    for H,wd,ep,seed in [(128,3e-3,40,0),(128,3e-3,40,1),(128,3e-3,40,2),(64,1e-3,35,0),(64,1e-3,35,1)]:
        t=time.time(); W,be,ha=train_mlp(Xt,yt,Xv,yv,hv,vv,H=H,wd=wd,lr=1e-3,epochs=ep,bs=512,seed=seed,verbose=False)
        s=mlp_score(Xv,W); S[f'mlp{H}_{wd}_{seed}']=s
        print('mlp',H,wd,seed,'hard %.2f'%metrics(s,yv,hv,vv)[1],'%.0fs'%(time.time()-t),flush=True)
    # references
    for name,sp,lam in [('v1','combined',0.1),('emb','embedding',0.001)]:
        A=load('train',sp)[0]; B=load('validation',sp)[0]
        sc2=StandardScaler().fit(A.reshape(-1,A.shape[2]))
        A=sc2.transform(A.reshape(-1,A.shape[2])); B=sc2.transform(B.reshape(-1,B.shape[2]))
        th=fit_logit_w(A.reshape(-1,4,A.shape[1]),yt,lam)
        S[name]=(B@th).reshape(-1,4); print(name,'hard %.2f'%metrics(S[name],yv,hv,vv)[1],flush=True)
    # average all MLP seeds
    mlp_keys=[k for k in S if k.startswith('mlp128')]
    S['mlpavg']=np.mean([z(S[k]) for k in mlp_keys],axis=0)
    print('mlpavg hard %.2f'%metrics(S['mlpavg'],yv,hv,vv)[1],flush=True)
    keys=list(S)
    best=[]
    for r in range(2,4):
        for combo in itertools.combinations(keys,r):
            s=np.mean([z(S[c]) for c in combo],axis=0)
            a,ha,n,hn=metrics(s,yv,hv,vv); best.append((ha,a,combo))
    best.sort(reverse=True)
    for b in best[:15]: print('ENS %.2f %.2f %s'%b,flush=True)
    np.savez(C/'ens_scores.npz',**{k:v for k,v in S.items()})
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
