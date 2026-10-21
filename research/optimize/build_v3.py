import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load_inputs
from features import parsed
from feats3 import extract
C=Path(__file__).resolve().parent/'cache'
for split in ('train','validation'):
    bank, arrays = load_inputs(split)
    ans=parsed(bank)
    t=time.time(); X=extract(bank, arrays['embeddings'], ans)
    print(split,X.shape,'%.1fs'%(time.time()-t),'finite',np.isfinite(X).all(),flush=True)
    np.save(C/f'{split}_v3.npy',X)
