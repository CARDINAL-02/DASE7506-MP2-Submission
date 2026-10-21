"""Recompute saved training OOF counts; no fitting, test access, or selection changes."""
from pathlib import Path
import hashlib
import json
import numpy as np
from common import load_bank
from features import parsed
from train_linear_v2 import labels


def main():
    here=Path(__file__).resolve().parent
    cache=next(p for p in (here.parent.parent/'optimize/cache/stack_data.npz',
                          here.parent/'research/optimize/cache/stack_data.npz') if p.is_file())
    bank=load_bank('train')
    y=labels(bank,'train')
    a=parsed(bank)
    valid=np.array([[v is not None for v in row] for row in a])
    hard=np.array([len({v for v in row if v is not None})>=2 for row in a])
    with np.load(cache,allow_pickle=False) as z:
        oof=z['oof']
    assert oof.shape==(len(bank),4,3) and np.isfinite(oof).all()
    values={}
    for name,s in [('linear',oof[:,:,0]),('proxy_ensemble',3*oof[:,:,0]+2*oof[:,:,1]+oof[:,:,2])]:
        c=np.where(valid,s,-np.inf).argmax(axis=1)
        ok=y[np.arange(len(y)),c].astype(bool)
        values[name]={'hard_correct':int(ok[hard].sum()),'hard_n':int(hard.sum()),
                      'hard_accuracy':float(ok[hard].mean()*100)}
    result={'purpose':'Post-review verification of saved train-only OOF scores; not a pre-test selection record.',
        'data_split':'train','folds':5,'grouping':'whole questions, including all four candidates',
        'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),'results':values,
        'proxy_definition':'Linear + one MLP seed 0 at 30 epochs + embedding-only, weights 3/2/1.',
        'tested_ensemble_definition':'Linear + average of three MLP seeds with validation-selected epochs + embedding-only.',
        'limitation':'Proxy differs from tested ensemble; seven-question edge is small, not established superiority.',
        'history':'Ensemble test preceded completed OOF analysis, which preceded final linear freeze/test; prior test exposure is disclosed.'}
    (here/'runs_linear/review/selection-provenance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(values,indent=2))


if __name__=='__main__':
    main()
