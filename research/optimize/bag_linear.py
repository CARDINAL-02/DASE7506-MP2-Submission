import sys, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
from models2 import build
from models4 import fit_logit_w
C=Path(__file__).resolve().parent/'cache'
def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb'); D=Xt.shape[2]; vv=valid_mask('validation')
    sc=StandardScaler().fit(Xt.reshape(-1,D))
    A=sc.transform(Xt.reshape(-1,D)).reshape(-1,4,D); B=sc.transform(Xv.reshape(-1,Xv.shape[2])).reshape(-1,4,Xv.shape[2])
    rng=np.random.default_rng(0); scores=[]
    for b in range(10):
        idx=rng.choice(len(yt),len(yt),replace=True)
        w=fit_logit_w(A[idx],yt[idx],0.01); s=(B.reshape(-1,D)@w).reshape(-1,4); scores.append(s)
        a,ha,n,hn=metrics(s,yv,hv,vv); print('bag',b,'hard %.2f'%ha,flush=True)
    avg=np.mean(scores,0); a,ha,n,hn=metrics(avg,yv,hv,vv); print('BAG lin hard %.2f all %.2f'%(ha,a))
    np.save(C/'bag_lin.npy',avg)
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
