import sys, time, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models1 import load, metrics, valid_mask, fit_lr, fit_pairwise, fit_logit
from models1 import C as CACHE

def build(mode):
    Xt=np.load(CACHE/f'train_v2.npy').astype(np.float64)
    Xv=np.load(CACHE/'validation_v2.npy').astype(np.float64)
    if 'emb' in mode:
        et=np.load(CACHE/'train_base.npz')['emb'].reshape(-1,1024)
        ev=np.load(CACHE/'validation_base.npz')['emb'].reshape(-1,1024)
        Xt=np.concatenate([Xt,et],1); Xv=np.concatenate([Xv,ev],1)
    if 'rel' in mode:
        et=np.load(CACHE/'train_base.npz')['emb']; ev=np.load(CACHE/'validation_base.npz')['emb']
        rt=(et-et.mean(1,keepdims=True)).reshape(-1,1024); rv=(ev-ev.mean(1,keepdims=True)).reshape(-1,1024)
        Xt=np.concatenate([Xt,rt],1); Xv=np.concatenate([Xv,rv],1)
    d=np.load(CACHE/'train_base.npz'); dv=np.load(CACHE/'validation_base.npz')
    return Xt.reshape(-1,4,Xt.shape[1]), d['y'], d['hard'], Xv.reshape(-1,4,Xv.shape[1]), dv['y'], dv['hard']

def run(mode):
    Xt,yt,ht,Xv,yv,hv=build(mode)
    sc=StandardScaler().fit(Xt.reshape(-1,Xt.shape[2]))
    Xt=sc.transform(Xt.reshape(-1,Xt.shape[2])).reshape(Xt.shape)
    Xv=sc.transform(Xv.reshape(-1,Xv.shape[2])).reshape(Xv.shape)
    vv=valid_mask('validation')
    print('===',mode,'D',Xt.shape[2],flush=True)
    rows=[]
    for Cval in (0.003,0.01,0.03,0.1):
        s,m=fit_lr(Xt,yt,Xv,Cval); a,ha,n,hn=metrics(s,yv,hv,vv)
        print('LR C=%g all %.2f hard %.2f (%d)'%(Cval,a,ha,hn),flush=True); rows.append(('lr',Cval,a,ha))
    for Cval in (0.001,0.003,0.01):
        s,w,npairs=fit_pairwise(Xt,yt,Xv,Cval); a,ha,n,hn=metrics(s,yv,hv,vv)
        print('PAIR C=%g all %.2f hard %.2f (%d)'%(Cval,a,ha,hn),flush=True); rows.append(('pair',Cval,a,ha))
    lams = (1e-4,1e-3,1e-2,0.03,0.05,0.1,0.2,0.3,0.5,1.0) if Xt.shape[2]<100 else (0.001,0.01,0.03,0.1,0.2,0.3,0.5,1.0)
    for lam in lams:
        s,w=fit_logit(Xt,yt,Xv,lam); a,ha,n,hn=metrics(s,yv,hv,vv)
        print('LOGIT lam=%g all %.2f hard %.2f (%d)'%(lam,a,ha,hn),flush=True); rows.append(('logit',lam,a,ha))
    best=max(rows,key=lambda r:(r[3],r[2])); print('BEST',best,flush=True)
    (Path(__file__).parent/f'results2_{mode}.json').write_text(json.dumps(rows,indent=1))

if __name__=='__main__':
    with threadpool_limits(limits=8):
        for mode in sys.argv[1:] or ['v2']:
            run(mode)
