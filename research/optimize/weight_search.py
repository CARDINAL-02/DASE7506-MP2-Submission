import sys, itertools, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
C=Path(__file__).resolve().parent/'cache'
S=np.load(C/'ens_scores.npz'); d=np.load(C/'validation_base.npz'); y=d['y']; h=d['hard']; vv=valid_mask('validation')
def z(s):
    s=s.reshape(-1); return ((s-s.mean())/(s.std()+1e-9)).reshape(-1,4)
Z={k:z(S[k]) for k in S.files}
best=[]
for wl in range(0,7):
  for w0 in range(0,7):
    for w1 in range(0,5):
      for w2 in range(0,5):
        for we in range(0,4):
          if wl+w0+w1+w2+we==0: continue
          s=(wl*Z['lin']+w0*Z['mlp128_0.003_0']+w1*Z['mlp128_0.003_1']+w2*Z['mlp128_0.003_2']+we*Z['emb'])
          a,ha,n,hn=metrics(s,y,h,vv)
          best.append((ha,a,(wl,w0,w1,w2,we),hn))
best.sort(reverse=True)
print('top:')
for b in best[:20]: print('hard %.2f all %.2f w=%s n=%d'%b)
# plateau: count configs >= 87.6 and their weight distribution
import collections
sel=[b for b in best if b[0]>=87.6]
print('configs >=87.6:',len(sel))
print('weight sums example',sel[:5])
