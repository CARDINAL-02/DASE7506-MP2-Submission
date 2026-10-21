"""Stacking meta-model trained on train OOF predictions only."""
import sys, time, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from models1 import metrics, valid_mask
from models4 import fit_logit_w
from models2 import build
from mlp2 import train as mlp_train, score as mlp_score
C=Path(__file__).resolve().parent/'cache'

def z(s):
    s=s.reshape(-1); return ((s-s.mean())/(s.std()+1e-9)).reshape(-1,4)

def group_feats(V):
    # V: (N,4,54) v2 features
    return np.stack([V[:,:,1],V[:,:,2],V[:,:,3],V[:,:,0],V[:,:,5],V[:,:,25],
                     V[:,:,30],V[:,:,26],V[:,:,37],V[:,:,20],V[:,:,23],
                     (V[:,:,1]==0.25).astype(np.float32)],axis=2)

def main():
    Xt,yt,ht,Xv,yv,hv=build('v2emb'); D=Xt.shape[2]
    vv=valid_mask('validation')
    groups=np.arange(len(yt)); gkf=GroupKFold(n_splits=5)
    oof=np.zeros((len(yt),4,3),dtype=np.float64)
    for k,(tr,te) in enumerate(gkf.split(Xt,yt,groups)):
        t=time.time()
        sc=StandardScaler().fit(Xt[tr].reshape(-1,D))
        A=sc.transform(Xt[tr].reshape(-1,D)).reshape(-1,4,D)
        B=sc.transform(Xt[te].reshape(-1,D)).reshape(-1,4,D)
        w=fit_logit_w(A,yt[tr],0.01)
        sl=(B.reshape(-1,D)@w).reshape(-1,4); mu,sd=sl.mean(),sl.std()+1e-9
        mlp=mlp_train(A.astype(np.float32),yt[tr],[D,128],seed=0,epochs=30,drop=0.0,wd=3e-3)
        sm=mlp_score(B.astype(np.float32),mlp); mm,ms=sm.mean(),sm.std()+1e-9
        Et=Xt[tr][:,:,54:].reshape(-1,1024); Ee=Xt[te][:,:,54:].reshape(-1,1024)
        es=StandardScaler().fit(Et); Et2=es.transform(Et); Ee2=es.transform(Ee)
        we=fit_logit_w(Et2.reshape(-1,4,1024),yt[tr],0.001)
        se=(Ee2@we).reshape(-1,4); em,esd=se.mean(),se.std()+1e-9
        oof[te,:,0]=(sl-mu)/sd; oof[te,:,1]=(sm-mm)/ms; oof[te,:,2]=(se-em)/esd
        print('fold',k,'%.0fs'%(time.time()-t),flush=True)
    Gt=group_feats(Xt[:,:,:54]); Gv=group_feats(Xv[:,:,:54])
    Ftr=np.concatenate([oof,Gt],axis=2); Fva_feat=Gv
    # validation base z from saved full-train scores
    S=np.load(C/'ens_scores.npz')
    Zv=np.stack([z(S['lin']),z(S['mlp128_0.003_0']),z(S['emb'])],axis=2)
    Fva=np.concatenate([Zv,Gv],axis=2)
    sc2=StandardScaler().fit(Ftr.reshape(-1,Ftr.shape[2])); A=sc2.transform(Ftr.reshape(-1,Ftr.shape[2])).reshape(-1,4,Ftr.shape[2]); B=sc2.transform(Fva.reshape(-1,Fva.shape[2])).reshape(-1,4,Fva.shape[2])
    print('meta D',A.shape[2],flush=True)
    for Cval in (0.003,0.01,0.03,0.1,0.3):
        w=fit_logit_w(A,yt,Cval); s=(B.reshape(-1,B.shape[2])@w).reshape(-1,4)
        a,ha,n,hn=metrics(s,yv,hv,vv); print('meta listwise C=%g hard %.2f all %.2f'%(Cval,ha,a),flush=True)
    for Cval in (0.003,0.01,0.03,0.1):
        m=LogisticRegression(C=Cval,max_iter=1000).fit(A.reshape(-1,A.shape[2]),yt.ravel())
        s=m.decision_function(B.reshape(-1,B.shape[2])).reshape(-1,4)
        a,ha,n,hn=metrics(s,yv,hv,vv); print('meta point C=%g hard %.2f all %.2f'%(Cval,ha,a),flush=True)
    np.savez(C/'stack_data.npz',oof=oof,Gt=Gt,Gv=Gv)
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
