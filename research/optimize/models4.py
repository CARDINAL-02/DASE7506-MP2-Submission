import sys, json, time
from pathlib import Path
import numpy as np
from collections import Counter
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
from models1 import load, metrics, valid_mask
from models2 import build
from features import majority

C=Path(__file__).resolve().parent/'cache'

def fit_logit_w(X,y,lam,w=None,maxiter=300):
    N,K=X.shape[1],X.shape[2]
    Xf=X.reshape(-1,K); mask=(y>0)
    if w is None: w=np.ones(len(y))
    w=w/w.mean()
    def obj(theta):
        s=(Xf@theta).reshape(-1,4)
        lse=logsumexp(s,axis=1)
        sp=np.where(mask,s,-1e30); lse_pos=logsumexp(sp,axis=1)
        per=lse-lse_pos
        loss=(w*per).mean()+0.5*lam*theta@theta
        q=np.exp(s-lse[:,None])
        p=np.exp(sp-lse_pos[:,None]); p=np.where(mask,p,0.)
        grad=Xf.T@((w[:,None]*(q-p)).ravel())/len(w)+lam*theta
        return loss,grad
    r=minimize(obj,np.zeros(K),jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-10})
    return r.x

def weights_for(kind, y, parsed):
    maj=majority([[v if v is None or isinstance(v,str) else None for v in row] for row in parsed])
    majok=y[np.arange(len(y)),maj].astype(bool)
    npos=y.sum(1)
    ndist=np.array([len({v for v in row if v is not None}) for row in parsed])
    if kind=='majwrong': return 1.0+2.0*(~majok)
    if kind=='onecorrect': return 1.0+2.0*(npos==1)
    if kind=='distinct': return 1.0+0.5*(ndist-1)
    return np.ones(len(y))

def z(scores):
    s=scores.reshape(-1); return ((s-s.mean())/(s.std()+1e-9)).reshape(-1,4)

def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb')
    sc=StandardScaler().fit(Xt.reshape(-1,Xt.shape[2]))
    Xt=sc.transform(Xt.reshape(-1,Xt.shape[2])).reshape(Xt.shape)
    Xv=sc.transform(Xv.reshape(-1,Xv.shape[2])).reshape(Xv.shape)
    vv=valid_mask('validation')
    par=np.load(C/'train_parsed.npy',allow_pickle=True)
    print('=== v2emb weighted',flush=True)
    scores={}
    for kind in ('none','majwrong','onecorrect','distinct'):
        w=weights_for(kind,yt,par)
        for lam in (0.003,0.01,0.03,0.1,0.2):
            t=time.time(); th=fit_logit_w(Xt,yt,lam,w)
            s=(Xv.reshape(-1,Xv.shape[2])@th).reshape(-1,4)
            a,ha,n,hn=metrics(s,yv,hv,vv)
            print('%s lam=%g hard %.2f all %.2f (%.0fs)'%(kind,lam,ha,a,time.time()-t),flush=True)
            scores[f'{kind}-{lam}']=s
    # individual reference models
    Xc,yc,hc,Xcv,ycv,hcv=load('train','combined'); Xce=load('train','embedding')[0]; Xcv_e=load('validation','embedding')[0]
    for name,Xtr,Xte,lams in [('combined',Xc,Xcv,(0.03,0.1,0.3)),('embedding',Xce,Xcv_e,(0.0003,0.001,0.003,0.01))]:
        sc2=StandardScaler().fit(Xtr.reshape(-1,Xtr.shape[2]))
        A=sc2.transform(Xtr.reshape(-1,Xtr.shape[2])).reshape(Xtr.shape)
        B=sc2.transform(Xte.reshape(-1,Xte.shape[2])).reshape(Xte.shape)
        for lam in lams:
            th=fit_logit_w(A,yc,lam)
            s=(B.reshape(-1,B.shape[2])@th).reshape(-1,4)
            a,ha,n,hn=metrics(s,ycv,hcv,vv)
            print('%s lam=%g hard %.2f all %.2f'%(name,lam,ha,a),flush=True)
            scores[f'{name}-{lam}']=s
    # ensembles of best individual configs
    cand=['none-0.01','majwrong-0.01','onecorrect-0.01','distinct-0.01','combined-0.1','embedding-0.001']
    cand=[c for c in cand if c in scores]
    best={}
    for c in cand:
        a,ha,n,hn=metrics(scores[c],yv,hv,vv); best[c]=(ha,a)
        print('ind',c,'hard %.2f all %.2f'%(ha,a),flush=True)
    import itertools
    for r in range(2,len(cand)+1):
        for combo in itertools.combinations(cand,r):
            s=np.mean([z(scores[c]) for c in combo],axis=0)
            a,ha,n,hn=metrics(s,yv,hv,vv)
            print('ENS','+'.join(combo),'hard %.2f all %.2f'%(ha,a),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
