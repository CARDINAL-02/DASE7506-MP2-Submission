import sys, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
C=Path(__file__).resolve().parent/'cache'
S=np.load(C/'ens_scores.npz'); d=np.load(C/'validation_base.npz'); y=d['y']; h=d['hard']; vv=valid_mask('validation')
p=np.load(C/'validation_parsed.npy',allow_pickle=True)
def z(s):
    s=s.reshape(-1); return ((s-s.mean())/(s.std()+1e-9)).reshape(-1,4)
ens=(3*z(S['lin'])+2*np.mean([z(S[f'mlp128_0.003_{i}']) for i in range(3)],0)+1*z(S['emb']))/6
for k in (2,3,4):
    mask=np.array([len({v for v in row if v is not None})==k for row in p])&h
    print('distinct',k,'n',mask.sum())
    for name,s in [('lin',S['lin']),('mlp0',S['mlp128_0.003_0']),('mlp2',S['mlp128_0.003_2']),('emb',S['emb']),('ens',ens)]:
        ss=s.copy(); ss[~vv]=-np.inf; ch=ss.argmax(1); ok=y[np.arange(len(y)),ch].astype(bool)
        print('  %-5s %.2f'%(name,ok[mask].mean()*100))
