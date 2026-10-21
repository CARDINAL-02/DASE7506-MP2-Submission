import sys, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
C=Path(__file__).resolve().parent/'cache'
S=np.load(C/'ens_scores.npz')
d=np.load(C/'validation_base.npz'); yv=d['y']; hv=d['hard']; vv=valid_mask('validation')
def z(s):
    s=s.reshape(-1); return ((s-s.mean())/(s.std()+1e-9)).reshape(-1,4)
def ev(name,keys,w=None):
    if w is None: w=[1]*len(keys)
    s=sum(wi*z(S[k]) for wi,k in zip(w,keys))/sum(w)
    a,ha,n,hn=metrics(s,yv,hv,vv); print('%-35s hard %.2f all %.2f (%d/%d)'%(name,ha,a,hn,int(hv.sum())))
def wz(s):
    s=s.reshape(-1,4); return ((s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-9)).reshape(-1,4)
def ev2(name,keys,w=None):
    if w is None: w=[1]*len(keys)
    s=sum(wi*wz(S[k]) for wi,k in zip(w,keys))/sum(w)
    a,ha,n,hn=metrics(s,yv,hv,vv); print('WZ %-30s hard %.2f all %.2f (%d/%d)'%(name,ha,a,hn,int(hv.sum())))
print('individuals:')
for k in S.files: ev(k,[k])
print('robust combos:')
ev('lin+mlpavg',['lin','mlpavg'])
ev('lin+mlpavg+emb',['lin','mlpavg','emb'])
ev('lin+mlp128_0+mlp64_0+emb',['lin','mlp128_0.003_0','mlp64_0.001_0','emb'])
ev('lin+3seeds+emb',['lin','mlp128_0.003_0','mlp128_0.003_1','mlp128_0.003_2','emb'])
ev('all 7',['lin','mlp128_0.003_0','mlp128_0.003_1','mlp128_0.003_2','mlp64_0.001_0','mlp64_0.001_1','emb'])
ev('weighted 3/2/1',['lin','mlpavg','emb'],[3,2,1])
ev('weighted 2/2/1',['lin','mlpavg','emb'],[2,2,1])
ev('weighted 1/2/1',['lin','mlpavg','emb'],[1,2,1])
ev('mlpavg+emb',['mlpavg','emb'])

print('within-question z:')
ev2('lin+mlpavg+emb 3/2/1',['lin','mlpavg','emb'],[3,2,1])
ev2('lin+mlpavg+emb 1/1/1',['lin','mlpavg','emb'])
ev2('lin+mlp128_0+mlp64_0+emb',['lin','mlp128_0.003_0','mlp64_0.001_0','emb'])
ev2('mlpavg+emb',['mlpavg','emb'])
