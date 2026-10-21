"""Validation experiments: linear, listwise, pairwise, trees on cached features."""
import sys, time, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

C = Path(__file__).resolve().parent / 'cache'
TR = C/'train_base.npz'; VA = C/'validation_base.npz'

def load(split, mode):
    X = np.load(C/f'{split}_{mode}.npy').astype(np.float64)
    d = np.load(C/f'{split}_base.npz')
    return X.reshape(-1,4,X.shape[1]), d['y'], d['hard']

def choose(scores, y):
    # mask invalid candidates (all parsed answers are non-None except ~5%; y is whether correct)
    # We do not know parse validity here; use a mask built from parsed text in caller.
    return scores.argmax(1)

def metrics(scores, y, hard, valid):
    s = scores.copy(); s[~valid] = -np.inf
    ch = s.argmax(1)
    ok = y[np.arange(len(y)), ch].astype(bool)
    return ok.mean()*100, ok[hard].mean()*100, ok.sum(), ok[hard].sum()

def prep(mode):
    Xt, yt, ht = load('train', mode)
    Xv, yv, hv = load('validation', mode)
    scaler = StandardScaler().fit(Xt.reshape(-1, Xt.shape[2]))
    Xt = scaler.transform(Xt.reshape(-1, Xt.shape[2])).reshape(Xt.shape)
    Xv = scaler.transform(Xv.reshape(-1, Xv.shape[2])).reshape(Xv.shape)
    return Xt, yt, ht, Xv, yv, hv, scaler

def valid_mask(split):
    p = np.load(C/f'{split}_parsed.npy', allow_pickle=True)
    return np.array([[v is not None for v in row] for row in p])

def fit_lr(Xt, yt, Xv, Cval):
    m = LogisticRegression(C=Cval, max_iter=1000, solver='lbfgs')
    m.fit(Xt.reshape(-1,Xt.shape[2]), yt.ravel())
    return m.decision_function(Xv.reshape(-1,Xv.shape[2])).reshape(-1,4), m

def fit_pairwise(Xt, yt, Xv, Cval, max_pairs=None):
    # build differences correct vs incorrect
    diffs=[]; labs=[]
    for x,y in zip(Xt,yt):
        pos=np.flatnonzero(y>0); neg=np.flatnonzero(y==0)
        for c in pos:
            for i in neg:
                diffs.append(x[c]-x[i]); labs.append(1.0)
                diffs.append(x[i]-x[c]); labs.append(0.0)
    if not diffs: raise ValueError('no pairs')
    diffs=np.asarray(diffs); labs=np.asarray(labs)
    if max_pairs and len(diffs)>max_pairs:
        rng=np.random.default_rng(0); idx=rng.choice(len(diffs),max_pairs,replace=False)
        diffs,labs=diffs[idx],labs[idx]
    m=LogisticRegression(C=Cval, max_iter=1000, solver='lbfgs')
    m.fit(diffs,labs)
    w=m.coef_.ravel()
    return Xv @ w, w, len(diffs)

def fit_logit(Xt, yt, Xv, lam, maxiter=200):
    N,D4,K=Xt.shape
    mask=yt>0
    def obj(w):
        s=Xt@w
        lse=logsumexp(s,axis=1)
        # multi-positive: log sum exp over positives
        sp=np.where(mask,s,-1e30)
        lse_pos=logsumexp(sp,axis=1)
        loss=np.mean(lse-lse_pos)+0.5*lam*np.dot(w,w)
        q=np.exp(s-lse[:,None])
        p=np.exp(sp-lse_pos[:,None]); p=np.where(mask,p,0.0)
        Xf=Xt.reshape(-1,K)
        grad=(Xf.T@q.ravel()-Xf.T@p.ravel())/N+lam*w
        return loss,grad
    t=time.time()
    r=minimize(obj,np.zeros(K),jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-7})
    print('  logit',lam,'nit',r.nit,'loss %.4f'%r.fun,'%.1fs'%(time.time()-t),flush=True)
    return Xv@r.x, r.x

def run(mode='combined'):
    Xt,yt,ht,Xv,yv,hv,sc=prep(mode)
    vt=valid_mask('train'); vv=valid_mask('validation')
    print('===',mode,'D',Xt.shape[2],'train hard',ht.sum(),'val hard',hv.sum(),flush=True)
    rows=[]
    for Cval in (0.003,0.01,0.03,0.1):
        s,m=fit_lr(Xt,yt,Xv,Cval)
        a,ha,n,hn=metrics(s,yv,hv,vv)
        print('LR C=%g all %.2f hard %.2f (%d/%d)'%(Cval,a,ha,hn,hv.sum()),flush=True)
        rows.append(('lr',Cval,a,ha))
    for Cval in (0.003,0.01,0.03,0.1):
        s,w,npairs=fit_pairwise(Xt,yt,Xv,Cval)
        a,ha,n,hn=metrics(s,yv,hv,vv)
        print('PAIR C=%g pairs %d all %.2f hard %.2f (%d/%d)'%(Cval,npairs,a,ha,hn,hv.sum()),flush=True)
        rows.append(('pair',Cval,a,ha))
    for lam in (1e-5,1e-4,1e-3,1e-2,1e-1):
        s,w=fit_logit(Xt,yt,Xv,lam)
        a,ha,n,hn=metrics(s,yv,hv,vv)
        print('LOGIT lam=%g all %.2f hard %.2f (%d/%d)'%(lam,a,ha,hn,hv.sum()),flush=True)
        rows.append(('logit',lam,a,ha))
    best=max(rows,key=lambda r:(r[3],r[2]))
    print('BEST',best,flush=True)
    (Path(__file__).parent/f'results1_{mode}.json').write_text(json.dumps(rows,indent=1))

if __name__=='__main__':
    with threadpool_limits(limits=8):
        for mode in sys.argv[1:] or ['structure','embedding','combined']:
            run(mode)
