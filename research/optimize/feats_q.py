import sys, re, numpy as np
from pathlib import Path
from fractions import Fraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DASE7506-MP2-main'/'code'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load_inputs
from features import parsed
from models1 import metrics, valid_mask, fit_logit
C=Path(__file__).resolve().parent/'cache'
NUM=re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
def nums(t):
    out=[]
    for s in NUM.findall(t):
        try:
            if '/' in s:
                a,b=s.split('/',1); v=float(a)/float(b) if float(b) else None
            else: v=float(s.replace(',',''))
            if v is not None: out.append(v)
        except: pass
    return out
def parse_float(a):
    if a is None: return None
    try: return float(Fraction(a))
    except: return None
def arith_feats(bank, ans):
    rows=[]
    for i,r in enumerate(bank):
        qn=nums(r['question']); qn=list(dict.fromkeys(qn))[:12]
        for j in range(4):
            a=parse_float(ans[i][j]); f=[0.]*11
            if a is not None and qn:
                matches=[0,0,0,0]
                for x in qn:
                    for y in qn:
                        for k,val in enumerate((x+y,x-y,x*y,(x/y if y else None))):
                            if val is not None and abs(val-a)<=1e-6*max(1,abs(a)): matches[k]+=1
                f[:4]=[min(1,m/max(1,len(qn))) for m in matches]
                f[4]=float(any(abs(a-x)<=1e-9*max(1,abs(a)) for x in qn))
                f[5]=float(any(abs(a-k*x)<=1e-9*max(1,abs(a)) for x in qn for k in range(2,11)))
                f[6]=float(abs(a-sum(qn))<=1e-6*max(1,abs(a)))
                f[7]=np.log1p(abs(a)/max(1,abs(max(qn))))
                f[8]=np.log1p(abs(a)/max(1,abs(min(qn))))
                f[9]=float(a<0); f[10]=float(abs(a)<1e-9)
            rows.append(f)
    return np.asarray(rows,dtype=np.float32)
def main():
    tr,tra=load_inputs('train'); va,vaa=load_inputs('validation')
    qtr=[r['question'] for r in tr]; qva=[r['question'] for r in va]
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=5000,sublinear_tf=True,stop_words='english')
    A=vec.fit_transform(qtr); B=vec.transform(qva)
    svd=TruncatedSVD(n_components=128,random_state=0); At=svd.fit_transform(A); Bt=svd.transform(B)
    print('svd explained',svd.explained_variance_ratio_.sum(),flush=True)
    at=arith_feats(tr,parsed(tr)); av=arith_feats(va,parsed(va))
    Vt=np.load(C/'train_v2.npy'); Vv=np.load(C/'validation_v2.npy')
    Et=np.load(C/'train_base.npz')['emb'].reshape(-1,1024); Ev=np.load(C/'validation_base.npz')['emb'].reshape(-1,1024)
    Qs_t=np.repeat(At,4,axis=0); Qs_v=np.repeat(Bt,4,axis=0)
    Xt=np.concatenate([Vt,at,Qs_t,Et],1); Xv=np.concatenate([Vv,av,Qs_v,Ev],1)
    d=np.load(C/'train_base.npz'); dv=np.load(C/'validation_base.npz'); yt,ht,yv,hv=d['y'],d['hard'],dv['y'],dv['hard']
    vv=valid_mask('validation')
    for name,cols in [('all',np.arange(Xt.shape[1])),('noqsvd',np.r_[np.arange(64),np.arange(192,Xt.shape[1])]),('onlyarith',np.r_[np.arange(54,65),np.arange(Xt.shape[1]-1024,Xt.shape[1])])]:
        A2=Xt[:,cols]; B2=Xv[:,cols]; sc=StandardScaler().fit(A2)
        A3=sc.transform(A2).reshape(-1,4,A2.shape[1]); B3=sc.transform(B2).reshape(-1,4,B2.shape[1])
        for lam in (0.003,0.01,0.03,0.1):
            s,_=fit_logit(A3,yt,B3,lam); a,ha,n,hn=metrics(s,yv,hv,vv)
            print(name,'D',A2.shape[1],'lam',lam,'hard %.2f all %.2f'%(ha,a),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=8): main()
