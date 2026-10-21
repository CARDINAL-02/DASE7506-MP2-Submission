import sys, json, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models1 import load, metrics, valid_mask
from models2 import build

def run(mode='v2emb'):
    Xt,yt,ht,Xv,yv,hv=build(mode)
    Xf=Xt.reshape(-1,Xt.shape[2]); Xvf=Xv.reshape(-1,Xv.shape[2])
    vv=valid_mask('validation')
    print('===',mode,'D',Xf.shape[1],flush=True)
    rows=[]
    def ev(name,s):
        a,ha,n,hn=metrics(s.reshape(-1,4),yv,hv,vv)
        print(name,'all %.2f hard %.2f (%d)'%(a,ha,hn),flush=True); rows.append((name,a,ha))
        return s
    for it,lr,leaf in [(200,0.05,31),(400,0.03,31),(300,0.05,63)]:
        t=time.time()
        m=HistGradientBoostingClassifier(max_iter=it,learning_rate=lr,max_leaf_nodes=leaf,
            min_samples_leaf=30,l2_regularization=1.0,early_stopping=False,random_state=0)
        m.fit(Xf,yt.ravel())
        s=m.predict_proba(Xvf)[:,1]
        ev(f'HGB it{it} lr{lr} leaf{leaf} (%.0fs)'%(time.time()-t),s)
    for n,leaf,mf in [(400,5,'sqrt'),(600,10,0.3)]:
        t=time.time()
        m=ExtraTreesClassifier(n_estimators=n,min_samples_leaf=leaf,max_features=mf,n_jobs=8,random_state=0)
        m.fit(Xf,yt.ravel()); s=m.predict_proba(Xvf)[:,1]
        ev(f'ET n{n} leaf{leaf} mf{mf} (%.0fs)'%(time.time()-t),s)
    (Path(__file__).parent/f'results3_{mode}.json').write_text(json.dumps(rows,indent=1))
if __name__=='__main__':
    with threadpool_limits(limits=8):
        for mode in sys.argv[1:] or ['v2']:
            run(mode)
