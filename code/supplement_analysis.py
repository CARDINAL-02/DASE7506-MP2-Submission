"""Validation-only uncertainty and disagreements for unchanged v1; later resource observations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from common import load_inputs
from features import choose, parsed, majority
from supplement_method import scores
from train import labels


def uncertainty(bank,arrays,y,checkpoint):
    a=parsed(bank)
    s=scores(bank,arrays,checkpoint)
    c=np.array(choose(s,a))
    b=np.array(majority(a))
    correct=y[np.arange(len(y)),c].astype(bool)
    bc=y[np.arange(len(y)),b].astype(bool)
    valid=np.array([[v is not None for v in row] for row in a])
    masked=np.where(valid,s,-np.inf)
    ranked=np.sort(masked,axis=1)
    multiple=valid.sum(axis=1)>=2
    margins=ranked[multiple,-1]-ranked[multiple,-2]
    edges=np.quantile(margins,[0,.25,.5,.75,1])
    bins=np.searchsorted(edges[1:-1],margins,side='right')
    table=[]
    for i in range(4):
        mask=bins==i
        table.append({'quartile':i+1,'n':int(mask.sum()),'lower':float(edges[i]),'upper':float(edges[i+1]),
                      'accuracy':float(correct[multiple][mask].mean()*100) if mask.any() else None})
    def stats(mask):
        return {'n':int(mask.sum()),'model_accuracy':float(correct[mask].mean()*100) if mask.any() else None,
                'majority_accuracy':float(bc[mask].mean()*100) if mask.any() else None}
    answer_disagree=np.array([a[i][c[i]]!=a[i][b[i]] for i in range(len(bank))])
    return {'margin_definition':'highest minus second-highest valid candidate raw logit; no probability calibration',
            'quartile_edges':edges.tolist(),'quartiles':table,'multi_valid_n':int(multiple.sum()),
            'single_valid':stats(valid.sum(axis=1)==1),'all_invalid':stats(valid.sum(axis=1)==0),
            'index_disagreement':stats(c!=b),'answer_disagreement':stats(answer_disagree),
            'answer_agreement':stats(~answer_disagree),
            'caveat':'Descriptive validation bins after model selection; duplicate final answers can create near-zero candidate margins; no test calibration or threshold tuning.'}


def main():
    bank,arrays=load_inputs('validation')
    y=labels(bank,'validation')
    result=uncertainty(bank,arrays,y,Path('runs/model_bundle'))
    Path('runs/supplement/uncertainty-v1.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
