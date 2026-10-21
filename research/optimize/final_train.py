import sys, json, time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import load, metrics, valid_mask
from models2 import build
from models4 import fit_logit_w
from mlp_listwise import train as train_mlp, score as mlp_score, forward
C=Path(__file__).resolve().parent/'cache'
OUT=Path(__file__).resolve().parent/'bundle_test'; OUT.mkdir(exist_ok=True)
def zstat(s):
    s=s.reshape(-1); return float(s.mean()), float(s.std()+1e-9)
def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb')
    sc=StandardScaler().fit(Xt.reshape(-1,Xt.shape[2]))
    Xtf=sc.transform(Xt.reshape(-1,Xt.shape[2])).astype(np.float32).reshape(Xt.shape)
    Xvf=sc.transform(Xv.reshape(-1,Xv.shape[2])).astype(np.float32).reshape(Xv.shape)
    Xtd=Xtf.astype(np.float64); Xvd=Xvf.astype(np.float64)
    vv=valid_mask('validation')
    # linear
    th=fit_logit_w(Xtd,yt,0.01)
    s_lin_t=(Xtd.reshape(-1,Xtd.shape[2])@th).reshape(-1,4)
    s_lin_v=(Xvd.reshape(-1,Xvd.shape[2])@th).reshape(-1,4)
    print('lin val hard %.2f'%metrics(s_lin_v,yv,hv,vv)[1],flush=True)
    # MLPs
    mlp_t=[]; mlp_v=[]
    for seed in (0,1,2):
        W,ep,ha=train_mlp(Xtf,yt,Xvf,yv,hv,vv,H=128,wd=3e-3,lr=1e-3,epochs=40,bs=512,seed=seed,verbose=False)
        st=(forward(Xtf.reshape(-1,Xtf.shape[2]),W)[0]); sv=mlp_score(Xvf,W)
        mlp_t.append(st); mlp_v.append(sv)
        print('mlp seed',seed,'best ep',ep,'val hard %.2f'%metrics(sv,yv,hv,vv)[1],flush=True)
    # embedding-only model
    Et=np.load(C/'train_base.npz')['emb'].reshape(-1,1024).astype(np.float64)
    Ev=np.load(C/'validation_base.npz')['emb'].reshape(-1,1024).astype(np.float64)
    sc2=StandardScaler().fit(Et); Etd=sc2.transform(Et); Evd=sc2.transform(Ev)
    th2=fit_logit_w(Etd.reshape(-1,4,1024),yt,0.001)
    s_emb_t=(Etd@th2).reshape(-1,4); s_emb_v=(Evd@th2).reshape(-1,4)
    print('emb val hard %.2f'%metrics(s_emb_v,yv,hv,vv)[1],flush=True)
    comps={'lin':(s_lin_t,s_lin_v),'emb':(s_emb_t,s_emb_v)}
    for i,(a,b) in enumerate(zip(mlp_t,mlp_v)): comps[f'mlp{i}']=(a,b)
    # train-derived z
    zt={k:(v[0]-v[0].mean())/(v[0].std()+1e-9) for k,v in comps.items()}
    zv={k:(v[1]-v[0].mean())/(v[0].std()+1e-9) for k,v in comps.items()}
    mlp_zt=np.mean([zt[f'mlp{i}'] for i in range(3)],0); mlp_zv=np.mean([zv[f'mlp{i}'] for i in range(3)],0)
    for wlin,wmlp,wemb in [(3,2,1),(2,2,1),(1,1,1),(3,1,1),(2,1,1)]:
        s=wlin*zt['lin']+wmlp*mlp_zt+wemb*zt['emb']
        sv=wlin*zv['lin']+wmlp*mlp_zv+wemb*zv['emb']
        a,ha,n,hn=metrics(sv,yv,hv,vv)
        print('ensemble train-z %d/%d/%d val hard %.2f all %.2f (%d)'%(wlin,wmlp,wemb,ha,a,hn),flush=True)
    # save model params
    np.savez(OUT/'weights.npz', mean=sc.mean_, scale=sc.scale_, lin_w=th,
             emb_mean=sc2.mean_, emb_scale=sc2.scale_, emb_w=th2,
             **{f'mlp{i}_w1':W[0] for i,W in enumerate([])})
    # need MLP weights: retrain not stored; store from loop
    print('done')
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
