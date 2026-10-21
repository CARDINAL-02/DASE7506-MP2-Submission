import sys, json, time
from pathlib import Path
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
from models1 import metrics, valid_mask, fit_logit
from models4 import fit_logit_w, weights_for
C=Path(__file__).resolve().parent/'cache'

def build3(use_emb=True):
    Xt=np.load(C/'train_v3.npy').astype(np.float64); Xv=np.load(C/'validation_v3.npy').astype(np.float64)
    if use_emb:
        et=np.load(C/'train_base.npz')['emb'].reshape(-1,1024); ev=np.load(C/'validation_base.npz')['emb'].reshape(-1,1024)
        Xt=np.concatenate([Xt,et],1); Xv=np.concatenate([Xv,ev],1)
    d=np.load(C/'train_base.npz'); dv=np.load(C/'validation_base.npz')
    return Xt.reshape(-1,4,Xt.shape[1]),d['y'],d['hard'],Xv.reshape(-1,4,Xv.shape[1]),dv['y'],dv['hard']

def main():
    Xt,yt,ht,Xv,yv,hv=build3(True)
    sc=StandardScaler().fit(Xt.reshape(-1,Xt.shape[2]))
    Xt=sc.transform(Xt.reshape(-1,Xt.shape[2])).reshape(Xt.shape)
    Xv=sc.transform(Xv.reshape(-1,Xv.shape[2])).reshape(Xv.shape)
    vv=valid_mask('validation'); par=np.load(C/'train_parsed.npy',allow_pickle=True)
    print('=== v3emb D',Xt.shape[2],flush=True)
    for lam in (0.003,0.01,0.02,0.03,0.05,0.1,0.2):
        s,_=fit_logit(Xt,yt,Xv,lam); a,ha,n,hn=metrics(s,yv,hv,vv)
        print('plain lam=%g hard %.2f all %.2f (%d)'%(lam,ha,a,hn),flush=True)
    for kind in ('distinct',):
        w=weights_for(kind,yt,par)
        for lam in (0.003,0.01,0.02,0.03,0.05,0.1,0.2):
            th=fit_logit_w(Xt,yt,lam,w); s=(Xv.reshape(-1,Xv.shape[2])@th).reshape(-1,4)
            a,ha,n,hn=metrics(s,yv,hv,vv)
            print('%s lam=%g hard %.2f all %.2f (%d)'%(kind,lam,ha,a,hn),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
