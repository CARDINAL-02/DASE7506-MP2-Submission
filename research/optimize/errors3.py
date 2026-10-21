import sys
from pathlib import Path
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
from models1 import metrics, valid_mask, fit_logit
from models2 import build
from features import majority
C=Path(__file__).resolve().parent/'cache'
def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb')
    sc=StandardScaler().fit(Xt.reshape(-1,Xt.shape[2]))
    Xt=sc.transform(Xt.reshape(-1,Xt.shape[2])).reshape(Xt.shape)
    Xv=sc.transform(Xv.reshape(-1,Xv.shape[2])).reshape(Xv.shape)
    s,_=fit_logit(Xt,yt,Xv,0.01)
    vv=valid_mask('validation'); s=s.copy(); s[~vv]=-np.inf; ch=s.argmax(1)
    ok=yv[np.arange(len(yv)),ch].astype(bool)
    p=np.load(C/'validation_parsed.npy',allow_pickle=True)
    maj=majority([[v if v is None or isinstance(v,str) else None for v in row] for row in p])
    mug=yv[np.arange(len(yv)),maj].astype(bool)
    print('hard %.2f'%(ok[hv].mean()*100))
    for k in (2,3,4):
        mask=np.array([len({v for v in row if v is not None})==k for row in p])&hv
        print('distinct',k,'n',mask.sum(),'acc %.2f'%(ok[mask].mean()*100),'maj %.2f'%(mug[mask].mean()*100))
    print('winonly',int((ok&~mug&hv).sum()),'majonly',int((~ok&mug&hv).sum()),'both',int((~ok&~mug&hv).sum()))
    # how often majority correct among hard by group size
    for k in (2,3,4):
        mask=np.array([len({v for v in row if v is not None})==k for row in p])&hv
        print('maj correct rate k%d %.2f'%(k,mug[mask].mean()*100))
    print('errors',int((~ok&hv).sum()))
if __name__=='__main__': main()
