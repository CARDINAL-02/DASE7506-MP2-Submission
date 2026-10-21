import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
C=Path(__file__).resolve().parent/'cache'
S=np.load(C/'ens_scores.npz'); d=np.load(C/'validation_base.npz'); y=d['y']; h=d['hard']; vv=valid_mask('validation')
def z(s):
    s=s.reshape(-1); return ((s-s.mean())/(s.std()+1e-9)).reshape(-1,4)
def wz(s):
    s=s.reshape(-1,4); return ((s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-9)).reshape(-1,4)
ens=3*z(S['lin'])+2*np.mean([z(S[f'mlp128_0.003_{i}']) for i in range(3)],0)+1*z(S['emb'])
ens=ens/6
def choice(s):
    s=s.copy(); s[~vv]=-np.inf; return s.argmax(1)
c_ens=choice(ens); c_lin=choice(S['lin'])
# current frozen model choices from official validation predictions
cur=np.array(json.load(open('DASE7506-MP2-main/code/runs/validation-predictions.json',encoding='utf-8'))['selected'])
ok=lambda c: y[np.arange(len(y)),c].astype(bool)
oe,oc=ok(c_ens),ok(cur)
print('ensemble hard %.2f'%(oe[h].mean()*100),'current hard %.2f'%(oc[h].mean()*100))
print('ens_only',int((oe&~oc&h).sum()),'cur_only',int((~oe&oc&h).sum()),'both_wrong',int((~oe&~oc&h).sum()))
delta=(oe.astype(float)-oc.astype(float))
rng=np.random.default_rng(1); idx=np.flatnonzero(h)
bs=np.array([delta[rng.choice(idx,len(idx),replace=True)].mean()*100 for _ in range(10000)])
print('paired delta %.2f pp, 95%% [%.2f, %.2f]'%(delta[h].mean()*100,np.quantile(bs,.025),np.quantile(bs,.975)))
