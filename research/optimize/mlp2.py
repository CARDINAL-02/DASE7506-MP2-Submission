import sys, time, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
from models2 import build
C=Path(__file__).resolve().parent/'cache'

def init(layers,rng):
    layers=list(layers)+[1]
    W=[]; prev=layers[0]
    for nxt in layers[1:]:
        W.append([ (rng.standard_normal((prev,nxt)).astype(np.float32)*np.sqrt(2.0/prev)),
                   np.zeros(nxt,dtype=np.float32) ])
        prev=nxt
    return W

def forward(X,W,drop=0.0,rng=None):
    H=X; masks=[]
    for i,(w,b) in enumerate(W):
        Z=H@w+b
        if i<len(W)-1:
            H=np.maximum(Z,0)
            if drop>0:
                m=(rng.random(H.shape)>=drop).astype(np.float32)/(1-drop); masks.append(m); H*=m
        else:
            return Z.reshape(-1,4),masks,H
    return Z,masks,H

def loss_grad(X,y,W,wd,drop,rng):
    B=X.shape[0]//4
    s,masks,last=forward(X,W,drop,rng)
    mx=s.max(1,keepdims=True); e=np.exp(s-mx); q=e/e.sum(1,keepdims=True)
    mask=y>0; sp=np.where(mask,s,-1e30); mp=sp.max(1,keepdims=True); ep=np.exp(sp-mp)
    p=np.where(mask,ep/ep.sum(1,keepdims=True),0.)
    loss=float(np.mean(np.log(e.sum(1))+mx[:,0]-np.log(ep.sum(1))-mp[:,0])+0.5*wd*sum(np.sum(w*w)+np.sum(b*b) for w,b in W))
    d=(q-p)/B
    grads=[None]*len(W)
    # recompute activations forward with masks recorded
    Hs=[X]; Zs=[]
    H=X; midx=0
    for i,(w,b) in enumerate(W):
        Z=H@w+b; Zs.append(Z)
        if i<len(W)-1:
            H=np.maximum(Z,0)
            if drop>0: H=H*masks[midx]; midx+=1
            Hs.append(H)
    dz=d.reshape(-1,1)
    for i in range(len(W)-1,-1,-1):
        w,b=W[i]
        Hprev=Hs[i]
        gw=Hprev.T@dz+wd*w; gb=dz.sum(0)+wd*b
        grads[i]=[gw,gb]
        if i>0:
            dz=dz@w.T
            dz=dz*(Zs[i-1]>0)
            if drop>0: dz=dz*masks[i-1]
    return loss,grads

def train(X,y,layers,seed,epochs=40,bs=512,lr=1e-3,wd=3e-3,drop=0.1):
    rng=np.random.default_rng(seed); W=init(layers,rng)
    m=[[np.zeros_like(w),np.zeros_like(b)] for w,b in W]; v=[[np.zeros_like(w),np.zeros_like(b)] for w,b in W]
    n=len(y); idx=np.arange(n)
    for ep in range(1,epochs+1):
        rng.shuffle(idx)
        for st in range(0,n,bs):
            b=idx[st:st+bs]; _,g=loss_grad(X[b].reshape(-1,X.shape[2]),y[b],W,wd,drop,rng)
            for i in range(len(W)):
                for j in range(2):
                    m[i][j]=0.9*m[i][j]+0.1*g[i][j]; v[i][j]=0.999*v[i][j]+0.001*g[i][j]**2
                    mh=m[i][j]/(1-0.9**ep); vh=v[i][j]/(1-0.999**ep)
                    W[i][j]-=lr*mh/(np.sqrt(vh)+1e-8)
    return W

def score(X,W):
    return forward(X.reshape(-1,X.shape[2]),W,0.0)[0]

def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb')
    sc=StandardScaler().fit(Xt.reshape(-1,Xt.shape[2]))
    Xt=sc.transform(Xt.reshape(-1,Xt.shape[2])).astype(np.float32).reshape(Xt.shape)
    Xv=sc.transform(Xv.reshape(-1,Xv.shape[2])).astype(np.float32).reshape(Xv.shape)
    vv=valid_mask('validation')
    D=Xt.shape[2]
    for layers,wd,drop in [([D,128],3e-3,.1),([D,256],3e-3,.1),([D,128,64],3e-3,.1),([D,256,128],1e-2,.15)]:
        for seed in (0,1):
            t=time.time(); W=train(Xt,yt,layers,seed,drop=drop,wd=wd); s=score(Xv,W)
            a,ha,n,hn=metrics(s,yv,hv,vv)
            print('layers',layers[1:],'wd',wd,'drop',drop,'seed',seed,'hard %.2f all %.2f (%.0fs)'%(ha,a,time.time()-t),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4): main()
