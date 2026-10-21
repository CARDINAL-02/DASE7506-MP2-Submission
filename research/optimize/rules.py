import sys, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
from models1 import metrics, valid_mask
from features import majority
C=Path(__file__).resolve().parent/'cache'
S=np.load(C/'ens_scores.npz'); d=np.load(C/'validation_base.npz'); y=d['y']; h=d['hard']; vv=valid_mask('validation')
def z(s):
    s=s.reshape(-1); return ((s-s.mean())/(s.std()+1e-9)).reshape(-1,4)
ens=(3*z(S['lin'])+2*np.mean([z(S[f'mlp128_0.003_{i}']) for i in range(3)],0)+1*z(S['emb']))/6
ens=ens.copy(); ens[~vv]=-np.inf
ch=ens.argmax(1)
p=np.load(C/'validation_parsed.npy',allow_pickle=True)
maj=np.array(majority([[v if v is None or isinstance(v,str) else None for v in row] for row in p]))
ok=y[np.arange(len(y)),ch].astype(bool); mok=y[np.arange(len(y)),maj].astype(bool)
srt=np.sort(ens,axis=1); margin=srt[:,-1]-srt[:,-2]
print('base hard %.2f'%(ok[h].mean()*100))
for rule in ['maj_if_small','maj_if_diff_small','maj_if_model_singleton_small']:
    for t in (0,0.05,0.1,0.2,0.3,0.5,0.8,1.0,1.5,2.0):
        c=ch.copy()
        if rule=='maj_if_small':
            m=margin<t
        elif rule=='maj_if_diff_small':
            m=(margin<t)&(c!=maj)
        else:
            m=(margin<t)&(c!=maj)&(np.array([sum(1 for k in range(4) if p[i][k]==p[i][c[i]] and p[i][k] is not None)<=1 for i in range(len(c))]))
        c[m]=maj[m]
        o=y[np.arange(len(y)),c].astype(bool)
        print('%-28s t=%.2f hard %.2f'%(rule,t,o[h].mean()*100))
