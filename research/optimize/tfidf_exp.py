import sys, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
from models1 import metrics, valid_mask, fit_logit
from common import load_inputs
from features import parsed
from features import majority
C=Path(__file__).resolve().parent/'cache'

def tf_feats(bank, vec):
    qs=[r['question'] for r in bank]
    cs=[[c['text'] for c in r['candidates']] for r in bank]
    Q=vec.transform(qs); Qn=normalize(Q)
    out=np.zeros((len(bank),4,3),dtype=np.float32)
    for i in range(len(bank)):
        M=vec.transform(cs[i]); Mn=normalize(M)
        qc=(Qn[i]@Mn.T).toarray().ravel()
        cc=(Mn@Mn.T).toarray()
        for j in range(4):
            others=[k for k in range(4) if k!=j]
            out[i,j,0]=qc[j]
            out[i,j,1]=cc[j,others].mean()
            a=parsed([bank[i]])[0]
            majidx=int(np.argmax(np.bincount([0 if v is None else 0 for v in a],minlength=4))) if False else 0
            out[i,j,2]=cc[j,others].max()
    return out

def main():
    tb,_=load_inputs('train'); vb,_=load_inputs('validation')
    docs=[]
    for b in tb:
        docs.append(b['question']); docs += [c['text'] for c in b['candidates']]
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=20000,sublinear_tf=True,stop_words='english')
    t=time.time(); vec.fit(docs); print('fit %.1fs vocab %d'%(time.time()-t,len(vec.vocabulary_)),flush=True)
    Ft=tf_feats(tb,vec); Fv=tf_feats(vb,vec); print('features',Ft.shape,flush=True)
    yt=np.load(C/'train_base.npz')['y']; ht=np.load(C/'train_base.npz')['hard']
    yv=np.load(C/'validation_base.npz')['y']; hv=np.load(C/'validation_base.npz')['hard']
    Vt=np.load(C/'train_v2.npy'); Vv=np.load(C/'validation_v2.npy')
    Et=np.load(C/'train_base.npz')['emb']; Ev=np.load(C/'validation_base.npz')['emb']
    vv=valid_mask('validation')
    Vt2=Vt.reshape(-1,54); Vv2=Vv.reshape(-1,54); Ft2=Ft.reshape(-1,3); Fv2=Fv.reshape(-1,3)
    Et2=Et.reshape(-1,1024); Ev2=Ev.reshape(-1,1024)
    for name,Xt,Xv in [('v2tf',np.concatenate([Vt2,Ft2],1),np.concatenate([Vv2,Fv2],1)),
                       ('v2tfemb',np.concatenate([Vt2,Ft2,Et2],1),np.concatenate([Vv2,Fv2,Ev2],1)),
                       ('v2emb',np.concatenate([Vt2,Et2],1),np.concatenate([Vv2,Ev2],1))]:
        sc=StandardScaler().fit(Xt); Xt=sc.transform(Xt).reshape(-1,4,Xt.shape[1]); Xv=sc.transform(Xv).reshape(-1,4,Xv.shape[1])
        for lam in (0.003,0.01,0.03,0.1):
            s,_=fit_logit(Xt,yt,Xv,lam); a,ha,n,hn=metrics(s,yv,hv,vv)
            print('%s lam=%g hard %.2f all %.2f'%(name,lam,ha,a),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
