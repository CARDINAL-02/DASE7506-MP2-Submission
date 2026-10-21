import sys, time, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
from common import load_inputs
from features import parsed
from models1 import metrics, valid_mask
C=Path(__file__).resolve().parent/'cache'
d=np.load(C/'train_base.npz'); dv=np.load(C/'validation_base.npz')
yt,ht,yv,hv=d['y'],d['hard'],dv['y'],dv['hard']; vv=valid_mask('validation')
tb,_=load_inputs('train'); vb,_=load_inputs('validation')
def texts(bank): return [c['text'] for r in bank for c in r['candidates']]
vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=50000,sublinear_tf=True,strip_accents='unicode')
t=time.time(); Xt=vec.fit_transform(texts(tb)); Xv=vec.transform(texts(vb)); print('vec %.0fs'%(time.time()-t),Xt.shape,flush=True)
for Cval in (0.1,0.3,1.0,3.0):
    t=time.time(); m=LogisticRegression(C=Cval,max_iter=2000,solver='liblinear')
    m.fit(Xt,yt.ravel()); s=m.decision_function(Xv).reshape(-1,4)
    a,ha,n,hn=metrics(s,yv,hv,vv); print('C=%g hard %.2f all %.2f (%.0fs)'%(Cval,ha,a,time.time()-t),flush=True)
