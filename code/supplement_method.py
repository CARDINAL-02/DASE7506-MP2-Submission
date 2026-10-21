"""Portable inference for supplementary ablations and a possible separately frozen v2."""
import json
from pathlib import Path
import numpy as np
from features import extract, parsed, choose
from supplement_features import restricted


def scores(bank, arrays, checkpoint):
    checkpoint = Path(checkpoint)
    c = json.loads((checkpoint/'config.json').read_text())
    if c['kind'] == 'ensemble':
        return sum(scores(bank,arrays,checkpoint/m)/s for m,s in zip(c['members'],c['scales']))/len(c['members'])
    mode = c['features']
    if mode == 'answer_surface_embedding':
        x = np.concatenate([restricted(bank,'answer_surface'),extract(bank,arrays,'embedding')],axis=1)
    else:
        x = restricted(bank,mode) if mode in ('answer','surface','answer_surface') else extract(bank,arrays,mode)
    with np.load(checkpoint/'weights.npz',allow_pickle=False) as w:
        x = (x-w['mean'])/w['scale']
        for i in range(c['layers']):
            x = x@w[f'w{i}']+w[f'b{i}']
            if i<c['layers']-1:
                x = np.maximum(x,0)
    return x.reshape(len(bank),4)


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    return choose(scores(bank,arrays,checkpoint),parsed(bank))
