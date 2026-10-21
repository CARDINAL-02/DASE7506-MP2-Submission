"""Validation-only descriptive evidence for the completed v3 study; no refitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from common import load_inputs
from train import labels


def main():
    r=Path('runs')
    b,_=load_inputs('validation')
    y=labels(b,'validation')
    c2=json.loads((r/'validation-v2-predictions.json').read_text())['selected']
    c3=json.loads((r/'validation-v3-predictions.json').read_text())['selected']
    ok2=y[np.arange(len(b)),c2].astype(bool)
    ok3=y[np.arange(len(b)),c3].astype(bool)
    d=ok3.astype(float)-ok2.astype(float)
    rng=np.random.default_rng(20261021)
    boot=np.array([d[rng.integers(0,len(d),len(d))].mean()*100 for _ in range(10000)])
    result={'v3_only_correct':int((d>0).sum()),'v2_only_correct':int((d<0).sum()),
        'delta_validation_pp':float(d.mean()*100),'paired_bootstrap_95_pp':np.quantile(boot,[.025,.975]).tolist(),
        'bootstrap_seed':20261021,'resamples':10000,'selection_corrected':False,
        'interpretation':'Descriptive validation comparison; no causal feature attribution or test-driven follow-up.'}
    (r/'v3/analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
