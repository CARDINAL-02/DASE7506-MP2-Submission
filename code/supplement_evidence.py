"""Validation-only paired analysis after the fixed experiment budget; no model selection."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from common import load_inputs
from features import parsed,choose
from supplement_method import scores
from train import labels


def main():
    b,a=load_inputs('validation')
    y=labels(b,'validation')
    ans=parsed(b)
    paths={'v1':Path('runs/model_bundle'),'v2':Path('runs/model_bundle_v2'),
           **{name:Path('runs/supplement')/name for name in ('answer','surface','answer_surface','answer_surface_embedding')}}
    correct={}
    for name,path in paths.items():
        c=choose(scores(b,a,path),ans)
        correct[name]=y[np.arange(len(b)),c].astype(bool)
    result={}
    for first,second in (('answer_surface','answer'),('answer_surface_embedding','answer_surface'),('v2','v1')):
        d=correct[first].astype(float)-correct[second].astype(float)
        rng=np.random.default_rng(20261021)
        boot=np.array([d[rng.integers(0,len(d),len(d))].mean()*100 for _ in range(10000)])
        result[first+'-vs-'+second]={'only_first_correct':int((d>0).sum()),'only_second_correct':int((d<0).sum()),
            'delta_pp':float(d.mean()*100),'paired_bootstrap_95_pp':np.quantile(boot,[.025,.975]).tolist()}
    result['bootstrap']={'seed':20261021,'resamples':10000,'unit':'question','selection_corrected':False}
    Path('runs/supplement/paired.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
