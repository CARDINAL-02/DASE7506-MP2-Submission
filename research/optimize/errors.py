import sys, json
from pathlib import Path
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
from models1 import load, fit_logit, valid_mask
from features import majority

C=Path(__file__).resolve().parent/'cache'
def main():
    Xt,yt,ht=load('train','combined'); Xv,yv,hv=load('validation','combined')
    sc=StandardScaler().fit(Xt.reshape(-1,Xt.shape[2]))
    Xt=sc.transform(Xt.reshape(-1,Xt.shape[2])).reshape(Xt.shape)
    Xv=sc.transform(Xv.reshape(-1,Xv.shape[2])).reshape(Xv.shape)
    s,_=fit_logit(Xt,yt,Xv,0.1)
    vv=valid_mask('validation')
    s= s.copy(); s[~vv]=-np.inf; ch=s.argmax(1)
    ok=yv[np.arange(len(yv)),ch].astype(bool)
    p=np.load(C/'validation_parsed.npy',allow_pickle=True)
    maj=majority([[v if isinstance(v,str) or v is None else None for v in row] for row in p])
    mug=yv[np.arange(len(yv)),maj].astype(bool)
    print('overall hard',ok[hv].mean()*100,'n',hv.sum())
    print('winner_only',int((ok&~mug&hv).sum()),'maj_only',int((~ok&mug&hv).sum()),'both_wrong',int((~ok&~mug&hv).sum()))
    # by distinct count
    for k in (2,3,4):
        mask=np.array([len({v for v in row if v is not None})==k for row in p])&hv
        if mask.sum():
            print('distinct',k,'n',mask.sum(),'acc %.2f'%(ok[mask].mean()*100),'maj %.2f'%(mug[mask].mean()*100))
    # among both wrong: is correct candidate singleton? is model picking singleton?
    idx=np.flatnonzero((~ok)&(~mug)&hv)
    print('\nBOTH WRONG examples',len(idx))
    for i in idx[:15]:
        vals=list(p[i]); cnt=Counter(v for v in vals if v is not None)
        print('--- id',i,'answers',vals,'chosen',int(ch[i]),'correct',np.flatnonzero(yv[i]).tolist(),'maj',int(maj[i]),'counts',dict(cnt))
    # feature correlation with correctness (hard only)
    X2=np.load(C/'validation_v2.npy').reshape(-1,54)
    yf=yv.reshape(-1); hf=np.repeat(hv,4)
    print('\nfeature corr (hard):')
    names=['valid','samecnt','distinct','maxgrp','validfrac','loglen','rellen','zlen','lognum','loglines','hash','boxed','logeq','logqlen','logabs','neg','integer','zero','simmean','simmax','same_mean','same_max','same_sum','diff_mean','qword_ov','qnum_ov','ans_occ','ans_last','numdens','neq','eq_frac','all_bad','fracok','fracbad','nfrac','pct','dollar','markerpos','distmean','distmed','rank','ismin','ismax','isqnum','lognorm','selfcorr','worddens','connective','there','how','wait','check','incorrect','approx']
    for j,n in enumerate(names):
        x=X2[hf,j]; y=yf[hf]
        if x.std()>1e-9:
            c=np.corrcoef(x,y)[0,1]
            if abs(c)>0.05: print('%-12s %+.3f'%(n,c))
    np.save(C/'val_scores_logit.npy',s); np.save(C/'val_choice_logit.npy',ch)
if __name__=='__main__': main()
