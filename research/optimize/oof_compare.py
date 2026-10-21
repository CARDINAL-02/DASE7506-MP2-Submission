import sys, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
C=Path(__file__).resolve().parent/'cache'
z=np.load(C/'stack_data.npz'); oof=z['oof']; Gt=z['Gt']
d=np.load(C/'train_base.npz'); y=d['y']; h=d['hard']; vv=valid_mask('train')
def ev(name,s):
    s=s.copy(); s[~vv]=-np.inf; ch=s.argmax(1); ok=y[np.arange(len(y)),ch].astype(bool)
    print('%-20s hard %.2f all %.2f'%(name,ok[h].mean()*100,ok.mean()*100))
ev('lin OOF',oof[:,:,0]); ev('mlp OOF',oof[:,:,1]); ev('emb OOF',oof[:,:,2])
ev('ens 3/2/1 OOF',3*oof[:,:,0]+2*oof[:,:,1]+1*oof[:,:,2])
ev('ens 2/2/1 OOF',2*oof[:,:,0]+2*oof[:,:,1]+1*oof[:,:,2])
ev('ens 1/1/1 OOF',oof.sum(2))
