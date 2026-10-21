import sys, time, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
from models2 import build
C=Path(__file__).resolve().parent/'cache'
def init(D,H,rng):
    return [rng.standard_normal((D,H)).astype(np.float32)*np.sqrt(2/D), np.zeros(H,np.float32),
            rng.standard_normal((H,1)).astype(np.float32)*np.sqrt(1/H), np.zeros(1,np.float32)]
def fwd(X,W):
    W1,b1,W2,b2=W; Z=X@W1+b1; H=np.maximum(Z,0); return (H@W2+b2).ravel(),Z,H
def train(X,y,seed,H=128,epochs=15,bs=512,lr=1e-3,wd=1e-4,drop=0.1):
    rng=np.random.default_rng(seed); W=init(X.shape[1],H,rng)
    m=[np.zeros_like(w) for w in W]; v=[np.zeros_like(w) for w in W]; n=len(y); idx=np.arange(n)
    for ep in range(1,epochs+1):
        rng.shuffle(idx)
        for st in range(0,n,bs):
            b=idx[st:st+bs]; Xb=X[b]; yb=y[b]
            s,Z,Hh=fwd(Xb,W)
            p=1/(1+np.exp(-s)); g=(p-yb)/len(b)
            gW2=Hh.T@g.reshape(-1,1)+wd*W[2]; gb2=np.float32(g.sum()).reshape(1)
            dH=g.reshape(-1,1)@W[2].T; dZ=dH*(Z>0)
            gW1=Xb.T@dZ+wd*W[0]; gb1=dZ.sum(0)
            gr=[gW1,gb1,gW2,gb2]
            for i in range(4):
                m[i]=.9*m[i]+.1*gr[i]; v[i]=.999*v[i]+.001*gr[i]**2
                W[i]-=lr*(m[i]/(1-.9**ep))/(np.sqrt(v[i]/(1-.999**ep))+1e-8)
    return W
def pair_scores(X,W):
    N=X.shape[0]; s=np.zeros((N,4),np.float64)
    for j in range(4):
        for k in range(4):
            if j==k: continue
            s[:,j]+=fwd(X[:,j]-X[:,k],W)[0]
    return s
def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb'); D=Xt.shape[2]; vv=valid_mask('validation')
    sc=StandardScaler().fit(Xt.reshape(-1,D))
    A=sc.transform(Xt.reshape(-1,D)).astype(np.float32).reshape(-1,4,D)
    B=sc.transform(Xv.reshape(-1,Xv.shape[2])).astype(np.float32).reshape(-1,4,Xv.shape[2])
    # build pairwise differences
    ds=[]; ls=[]
    for x,y in zip(A,yt):
        pos=np.flatnonzero(y>0); neg=np.flatnonzero(y==0)
        for c in pos:
            for i in neg:
                ds.append(x[c]-x[i]); ls.append(1.); ds.append(x[i]-x[c]); ls.append(0.)
    ds=np.asarray(ds); ls=np.asarray(ls,np.float32); print('pairs',ds.shape,flush=True)
    for seed in (0,1):
        t=time.time(); W=train(ds,ls,seed); s=pair_scores(B,W)
        a,ha,n,hn=metrics(s,yv,hv,vv); print('pair mlp seed',seed,'hard %.2f all %.2f (%.0fs)'%(ha,a,time.time()-t),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
