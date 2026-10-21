import sys, numpy as np, re
from pathlib import Path
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
from models1 import metrics, valid_mask
from common import load_inputs
C=Path(__file__).resolve().parent/'cache'
S=np.load(C/'ens_scores.npz'); d=np.load(C/'validation_base.npz'); y=d['y']; h=d['hard']; vv=valid_mask('validation')
def z(s):
    s=s.reshape(-1); return ((s-s.mean())/(s.std()+1e-9)).reshape(-1,4)
ens=(3*z(S['lin'])+2*np.mean([z(S[f'mlp128_0.003_{i}']) for i in range(3)],0)+1*z(S['emb']))/6
ens=ens.copy(); ens[~vv]=-np.inf
ch=ens.argmax(1)
bank,_=load_inputs('validation'); p=np.load(C/'validation_parsed.npy',allow_pickle=True)
q=[r['question'].lower() for r in bank]
ok=y[np.arange(len(y)),ch].astype(bool); print('base %.2f'%(ok[h].mean()*100))
def val(a):
    try: return float(Fraction(a)) if a is not None else None
    except: return None
for rule in ['howmany_nonint','howmany_neg','howmany_either']:
    c=ch.copy()
    for i in range(len(c)):
        if 'how many' not in q[i]: continue
        v=[val(a) for a in p[i]]
        if rule=='howmany_nonint': bad=lambda x: x is not None and abs(x-round(x))>1e-9
        elif rule=='howmany_neg': bad=lambda x: x is not None and x<0
        else: bad=lambda x: x is not None and (abs(x-round(x))>1e-9 or x<0)
        if bad(v[c[i]]):
            cand=[j for j in range(4) if v[j] is not None and not bad(v[j])]
            if cand:
                c[i]=max(cand,key=lambda j:ens[i,j])
    o=y[np.arange(len(y)),c].astype(bool)
    print(rule,'%.2f changed %d'%(o[h].mean()*100,(c!=ch).sum()))
