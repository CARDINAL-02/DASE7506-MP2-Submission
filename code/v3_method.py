"""Portable NumPy inference for tree/linear v3 models and fixed fusions."""
import json
from pathlib import Path
import numpy as np
from features import choose,parsed
from v3_features import v3_features
from supplement_method import scores as old_scores


def tree_predict(x,tree):
    node=np.zeros(len(x),dtype=np.int32)
    active=np.ones(len(x),dtype=bool)
    while active.any():
        ids=np.flatnonzero(active)
        n=node[ids]
        leaves=tree['leaf'][n]
        active[ids[leaves]]=False
        ids=ids[~leaves]
        n=node[ids]
        if not len(ids):
            break
        left=x[ids,tree['feature'][n]]<=tree['threshold'][n]
        node[ids]=np.where(left,tree['left'][n],tree['right'][n])
    return tree['value'][node]


def scores(bank,arrays,checkpoint,cache=None):
    p=Path(checkpoint)
    c=json.loads((p/'config.json').read_text())
    if c['kind']=='fusion-v3':
        old=old_scores(bank,arrays,p/'v2')
        new=scores(bank,arrays,p/'tree',cache)
        return (1-c['weight'])*old/c['old_scale']+c['weight']*new/c['new_scale']
    cache={} if cache is None else cache
    mode=c['features']
    if mode not in cache:
        cache[mode]=v3_features(bank,arrays,mode)
    x=cache[mode]
    if c['kind']=='forest-v3':
        x=x.astype(np.float32)
    with np.load(p/'weights.npz',allow_pickle=False) as w:
        if c['kind']=='linear-v3':
            result=((x-w['mean'])/w['scale'])@w['weight']+w['bias']
        else:
            result=np.full(len(x),float(c['base']),dtype=np.float64)
            for i in range(c['trees']):
                tree={k:w[f'{i}_{k}'] for k in ('feature','threshold','left','right','leaf','value')}
                result+=tree_predict(x,tree)
    return result.reshape(len(bank),4)


def predict(bank,arrays,checkpoint):
    if not bank:
        return []
    return choose(scores(bank,arrays,checkpoint),parsed(bank))
