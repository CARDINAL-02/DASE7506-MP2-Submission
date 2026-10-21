import sys
from pathlib import Path
import numpy as np
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
from features import majority
C=Path(__file__).resolve().parent/'cache'
X=np.load(C/'validation_v2.npy').reshape(798,4,54)
y=np.load(C/'validation_base.npz')['y']; h=np.load(C/'validation_base.npz')['hard']
p=np.load(C/'validation_parsed.npy',allow_pickle=True)
ch=np.load(C/'val_choice_logit.npy')  # v1 model choices, placeholder
# load v2emb choices by recomputing? use saved? not saved. Use v1 for rough.
maj=majority([[v if v is None or isinstance(v,str) else None for v in row] for row in p])
ok=y[np.arange(798),ch].astype(bool); mug=y[np.arange(798),maj].astype(bool)
idx=np.flatnonzero((~ok)&(~mug)&h)
print('both wrong',len(idx))
fields={'len':5,'eq':30,'qov':25,'occ':26,'simS':20,'simD':23,'integer':16,'marker':37,'distmed':39,'valid':0,'samecnt':1}
for i in idx:
    cor=np.flatnonzero(y[i]); sel=int(ch[i])
    print('\nid',i,'answers',list(p[i]),'correct',cor.tolist(),'chosen',sel,'maj',int(maj[i]))
    for name,j in fields.items():
        print('  %s corr %.2f sel %.2f'%(name, X[i,cor,j].mean(), X[i,sel,j]), end=';')
    print()
