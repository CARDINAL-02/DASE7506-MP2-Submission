"""Listwise MLP trained with multi-positive softmax; NumPy only."""
import sys, time, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
from models2 import build

C=Path(__file__).resolve().parent/'cache'

def init(D,H,rng):
    return [ (rng.standard_normal((D,H)).astype(np.float32)*np.sqrt(2.0/D)),
             np.zeros(H,dtype=np.float32),
             (rng.standard_normal((H,1)).astype(np.float32)*np.sqrt(1.0/H)),
             np.zeros(1,dtype=np.float32) ]

def forward(X,W):
    W1,b1,W2,b2=W
    Z=X@W1+b1; H=np.maximum(Z,0); s=(H@W2+b2).reshape(-1,4)
    return s,Z,H

def loss_grad(X,y,W,wd):
    W1,b1,W2,b2=W; B=X.shape[0]//4
    s,Z,H=forward(X,W)
    mx=s.max(1,keepdims=True); e=np.exp(s-mx); q=e/e.sum(1,keepdims=True)
    mask=y>0; sp=np.where(mask,s,-1e30); mp=sp.max(1,keepdims=True)
    ep=np.exp(sp-mp); p=np.where(mask,ep/ep.sum(1,keepdims=True),0.0)
    loss=float(np.mean(np.log(e.sum(1))+mx[:,0]-np.log(ep.sum(1))-mp[:,0])+0.5*wd*(np.sum(W1*W1)+np.sum(W2*W2)))
    ds=(q-p)/B
    gW2=H.T@ds.reshape(-1,1)+wd*W2; gb2=np.float32(ds.sum()).reshape(1)
    dH=ds.reshape(-1,1)@W2.T
    dZ=dH*(Z>0)
    gW1=X.T@dZ+wd*W1; gb1=dZ.sum(0)
    return loss,[gW1,gb1,gW2,gb2]

def train(X,y,Xv,yv,hv,vv,H=64,wd=1e-3,lr=1e-3,epochs=40,bs=512,seed=0,verbose=True):
    rng=np.random.default_rng(seed); W=init(X.shape[2],H,rng)
    m=[np.zeros_like(w) for w in W]; v=[np.zeros_like(w) for w in W]
    n=len(y); idx=np.arange(n); best=(-1,None,-1)
    for ep in range(1,epochs+1):
        rng.shuffle(idx)
        tot=0.
        for st in range(0,n,bs):
            b=idx[st:st+bs]; Xb=X[b].reshape(-1,X.shape[2]); yb=y[b]
            loss,g=loss_grad(Xb,yb,W,wd); tot+=loss*len(b)
            for i,(p_,g_) in enumerate(zip(W,g)):
                m[i]=0.9*m[i]+0.1*g_; v[i]=0.999*v[i]+0.001*(g_*g_)
                mh=m[i]/(1-0.9**ep); vh=v[i]/(1-0.999**ep)
                p_-=lr*mh/(np.sqrt(vh)+1e-8)
        if ep%5==0 or ep==1:
            sv=score(Xv,W); a,ha,nn,hn=metrics(sv,yv,hv,vv)
            if ha>best[0]: best=(ha,[w.copy() for w in W],ep)
            if verbose: print('ep %d loss %.4f val hard %.2f all %.2f'%(ep,tot/n,ha,a),flush=True)
    return best[1],best[2],best[0]

def score(X,W):
    return forward(X.reshape(-1,X.shape[2]),W)[0]

def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb')
    sc=StandardScaler().fit(Xt.reshape(-1,Xt.shape[2]))
    Xt=sc.transform(Xt.reshape(-1,Xt.shape[2])).astype(np.float32).reshape(Xt.shape)
    Xv=sc.transform(Xv.reshape(-1,Xv.shape[2])).astype(np.float32).reshape(Xv.shape)
    vv=valid_mask('validation')
    res=[]
    for H in (32,64,128):
        for wd in (3e-4,1e-3,3e-3):
            t=time.time(); W,ep,ha=train(Xt,yt,Xv,yv,hv,vv,H=H,wd=wd,lr=1e-3,epochs=40,seed=0)
            sv=score(Xv,W); a,ha2,n,hn=metrics(sv,yv,hv,vv)
            print('H%d wd%g best_ep%d hard %.2f all %.2f (%.0fs)'%(H,wd,ep,ha2,a,time.time()-t),flush=True)
            res.append({'H':H,'wd':wd,'ep':ep,'hard':ha2,'all':a})
    (Path(__file__).parent/'results_mlp.json').write_text(json.dumps(res,indent=1))
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
