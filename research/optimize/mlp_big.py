import sys, time, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
from models2 import build
from mlp2 import train, score
C=Path(__file__).resolve().parent/'cache'
def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb'); D=Xt.shape[2]
    sc=StandardScaler().fit(Xt.reshape(-1,D))
    Xt=sc.transform(Xt.reshape(-1,D)).astype(np.float32).reshape(-1,4,D)
    Xv=sc.transform(Xv.reshape(-1,Xv.shape[2])).astype(np.float32).reshape(-1,4,Xv.shape[2])
    vv=valid_mask('validation')
    for H,wd,drop in [(256,3e-3,0.0),(256,1e-2,0.0),(512,1e-2,0.2),(512,3e-2,0.2)]:
        for seed in (0,1):
            t=time.time(); W=train(Xt,yt,[D,H],seed,drop=drop,wd=wd,epochs=40); s=score(Xv,W)
            a,ha,n,hn=metrics(s,yv,hv,vv)
            print('H',H,'wd',wd,'drop',drop,'seed',seed,'hard %.2f all %.2f (%.0fs)'%(ha,a,time.time()-t),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
