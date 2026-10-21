import sys, json, itertools, time
from pathlib import Path
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
from models1 import load, metrics, valid_mask
from models2 import build
from models4 import fit_logit_w, z
from features import majority
C=Path(__file__).resolve().parent/'cache'

def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb')
    sc=StandardScaler().fit(Xt.reshape(-1,Xt.shape[2]))
    Xt=sc.transform(Xt.reshape(-1,Xt.shape[2])).reshape(Xt.shape)
    Xv=sc.transform(Xv.reshape(-1,Xv.shape[2])).reshape(Xv.shape)
    vv=valid_mask('validation'); par=np.load(C/'train_parsed.npy',allow_pickle=True)
    ndist=np.array([len({v for v in row if v is not None}) for row in par])
    counts=np.array([[sum(1 for x in row if x==v) for v in row] for row in par])
    maxgrp=counts.max(1)
    variants={
      'plain':np.ones(len(yt)),
      'd05':1+.5*(ndist-1),'d1':1+1*(ndist-1),'d2':1+2*(ndist-1),
      'inv3':1+1*(maxgrp<3),'inv2':1+2*(maxgrp<2),
      'bal':1/maxgrp,
      'mix':(1+.5*(ndist-1))*(1/maxgrp),
    }
    S={}
    for name,w in variants.items():
        for lam in (0.005,0.01,0.02,0.05,0.1):
            th=fit_logit_w(Xt,yt,lam,w); s=(Xv.reshape(-1,Xv.shape[2])@th).reshape(-1,4)
            a,ha,n,hn=metrics(s,yv,hv,vv); S[f'{name}-{lam}']=s
            print('%-5s lam=%-5g hard %.2f all %.2f'%(name,lam,ha,a),flush=True)
    # reference models
    for name,sp,lam in [('v1comb','combined',0.1),('emb','embedding',0.001)]:
        Xtr=load('train',sp)[0]; Xte=load('validation',sp)[0]
        sc2=StandardScaler().fit(Xtr.reshape(-1,Xtr.shape[2]))
        A=sc2.transform(Xtr.reshape(-1,Xtr.shape[2])).reshape(Xtr.shape); B=sc2.transform(Xte.reshape(-1,Xte.shape[2])).reshape(Xte.shape)
        th=fit_logit_w(A,yt,lam); s=(B.reshape(-1,B.shape[2])@th).reshape(-1,4)
        a,ha,n,hn=metrics(s,yv,hv,vv); S[name]=s; print(name,'hard %.2f'%ha,flush=True)
    rows=[]
    for n,s in S.items():
        a,ha,nn,hn=metrics(s,yv,hv,vv); rows.append((ha,a,n))
    rows.sort(reverse=True)
    print('TOP individuals:')
    for r in rows[:8]: print('  %.2f %.2f %s'%r)
    keys=[r[2] for r in rows[:6]]
    best=[]
    for r in range(2,4):
        for combo in itertools.combinations(keys,r):
            s=np.mean([z(S[c]) for c in combo],axis=0)
            a,ha,nn,hn=metrics(s,yv,hv,vv); best.append((ha,a,combo))
    best.sort(reverse=True)
    print('TOP ensembles:')
    for b in best[:10]: print('  %.2f %.2f %s'%b)
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
