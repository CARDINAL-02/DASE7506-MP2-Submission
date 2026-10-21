import sys
from pathlib import Path
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
from models1 import metrics, valid_mask, fit_logit
from models2 import build
from features import majority

C=Path(__file__).resolve().parent/'cache'
names=['valid','samecnt','distinct','maxgrp','validfrac','loglen','rellen','zlen','lognum','loglines','hash','boxed','logeq','logqlen','logabs','neg','integer','zero','simmean','simmax','same_mean','same_max','same_sum','diff_mean','qword_ov','qnum_ov','ans_occ','ans_last','numdens','neq','eq_frac','all_bad','fracok','fracbad','nfrac','pct','dollar','markerpos','distmean','distmed','rank','ismin','ismax','isqnum','lognorm','selfcorr','worddens','connective','there','how','wait','check','incorrect','approx']
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
    print('hard %.2f'%(ok[hv].mean()*100),'winonly',int((ok&~mug&hv).sum()),'majonly',int((~ok&mug&hv).sum()),'both',int((~ok&~mug&hv).sum()))
    X2=np.load(C/'validation_v2.npy').reshape(798,4,54)
    idx=np.flatnonzero((~ok)&(~mug)&hv)
    for i in idx[:12]:
        print('\n--- id',i,'answers',list(p[i]),'chosen',int(ch[i]),'correct',np.flatnonzero(yv[i]).tolist(),'maj',int(maj[i]))
        for j in range(4):
            f=X2[i,j]
            print('  c%d len%.0f qov%.2f eq%.2f occ%.1f simS%.2f simD%.2f int%.0f'%(j, f[5],f[25],f[30],f[26],f[20],f[23],f[16]))
    # diff between correct and chosen on features
    Xf=X2.reshape(-1,54); yf=yv.reshape(-1); hf=np.repeat(hv,4)
    print('\nCorrect-vs-incorrect standardized mean diff (hard):')
    m1=Xf[yf.reshape(-1)==1][:, :].mean(0); m0=Xf[yf.reshape(-1)==0].mean(0); sd=Xf.std(0)+1e-9
    for j,n in enumerate(names):
        d=(m1[j]-m0[j])/sd[j]
        if abs(d)>0.15: print('%-12s %+.2f'%(n,d))
if __name__=='__main__': main()
