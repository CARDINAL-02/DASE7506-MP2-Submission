"""Offline NumPy inference; no references, fitting, or executable pickle assets."""
import json
from pathlib import Path
import numpy as np
from features import extract, parsed, choose, majority
import features_v2


def _mlp_scores(X, w, i):
    z = np.maximum(X @ w[f'mlp{i}_w1'] + w[f'mlp{i}_b1'], 0)
    return (z @ w[f'mlp{i}_w2'] + w[f'mlp{i}_b2']).reshape(-1, 4)


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    if not bank:
        return []
    answers = parsed(bank)
    if checkpoint is None:
        return majority(answers)
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text())
    if config['kind'] == 'majority':
        return majority(answers)
    if config['kind'] == 'linear_v2':
        with np.load(checkpoint / 'weights.npz', allow_pickle=False) as w:
            emb = arrays['embeddings'].astype(np.float32, copy=False)
            f = features_v2.extract(bank, emb, answers).astype(np.float64)
            X = np.concatenate([f, emb.reshape(-1, 1024).astype(np.float64)], axis=1)
            X = (X - w['mean']) / w['scale']
            scores = (X @ w['lin_w']).reshape(-1, 4)
        return choose(scores, answers)
    if config['kind'] == 'ensemble_v2':
        with np.load(checkpoint / 'weights.npz', allow_pickle=False) as w:
            emb = arrays['embeddings'].astype(np.float32, copy=False)
            f = features_v2.extract(bank, emb, answers).astype(np.float64)
            X = np.concatenate([f, emb.reshape(-1, 1024).astype(np.float64)], axis=1)
            X = (X - w['mean']) / w['scale']
            lin = (X @ w['lin_w']).reshape(-1, 4)
            z_lin = (lin - float(w['lin_mu'])) / float(w['lin_sd'])
            X32 = X.astype(np.float32)
            z_mlp = np.mean([(_mlp_scores(X32, w, i) - float(w[f'mlp{i}_mu'])) / float(w[f'mlp{i}_sd'])
                             for i in range(len(config['seeds']))], axis=0)
            E = (emb.reshape(-1, 1024).astype(np.float64) - w['emb_mean']) / w['emb_scale']
            emb_s = (E @ w['emb_w']).reshape(-1, 4)
            z_emb = (emb_s - float(w['emb_mu'])) / float(w['emb_sd'])
            wl, wm, we = config['ensemble_weights']
            scores = (wl * z_lin + wm * z_mlp + we * z_emb) / (wl + wm + we)
        return choose(scores, answers)
    x = extract(bank, arrays, config['features'])
    with np.load(checkpoint / 'weights.npz', allow_pickle=False) as w:
        x = (x - w['mean']) / w['scale']
        for i in range(config['layers']):
            x = x @ w[f'w{i}'] + w[f'b{i}']
            if i < config['layers'] - 1:
                x = np.maximum(x, 0)
    return choose(x.reshape(len(bank), 4), answers)
