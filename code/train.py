"""Fixed train/validation experiment grid. Never loads test inputs or references."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import load_inputs, read_jsonl, DATA, parse_answer
from features import parsed, majority, choose, extract

ROOT = Path('runs')


def labels(bank, split):
    refs = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    assert [r['id'] for r in refs] == [r['id'] for r in bank]
    return np.array([[int(v is not None and v == parse_answer(g['answer'])) for v in a]
                     for a, g in zip(parsed(bank), refs)], dtype=np.int64)


def metrics(choices, y, hard):
    ok = y[np.arange(len(y)), choices].astype(bool)
    return {'accuracy': float(ok.mean()*100), 'hard_accuracy': float(ok[hard].mean()*100),
            'correct': int(ok.sum()), 'n': len(y), 'hard_n': int(hard.sum())}


def export(folder, model, scaler, mode):
    folder.mkdir(parents=True, exist_ok=True)
    if hasattr(model, 'coefs_'):
        weights, biases = model.coefs_, model.intercepts_
    else:
        weights, biases = [model.coef_.T], [model.intercept_]
    values = {'mean': scaler.mean_.astype(np.float64), 'scale': scaler.scale_.astype(np.float64)}
    for i, (w, b) in enumerate(zip(weights, biases)):
        values[f'w{i}'], values[f'b{i}'] = w.astype(np.float64), b.astype(np.float64)
    np.savez(folder / 'weights.npz', **values)
    (folder / 'config.json').write_text(json.dumps({'kind':'neural' if len(weights)>1 else 'linear',
              'features':mode, 'layers':len(weights)}, indent=2))
    return sum(v.nbytes for v in values.values()) + (folder / 'config.json').stat().st_size


def main():
    ROOT.mkdir(exist_ok=True)
    start = time.perf_counter()
    train, ta = load_inputs('train')
    val, va = load_inputs('validation')
    yt, yv = labels(train, 'train'), labels(val, 'validation')
    a = parsed(val)
    hard = np.array([len({v for v in row if v is not None}) >= 2 for row in a])
    records = []
    def record(name, choices, **kwargs):
        item = {'name':name, **metrics(choices, yv, hard), **kwargs}
        records.append(item)
        (ROOT / 'experiments.json').write_text(json.dumps(records, indent=2))
        (ROOT / f'{name}-validation.json').write_text(json.dumps({'selected':choices}))
        print(json.dumps(item), flush=True)
    record('first', [0]*len(val), asset_bytes=0, train_seconds=0.)
    record('majority', majority(a), asset_bytes=0, train_seconds=0.)
    majority_dir = ROOT/'majority'
    majority_dir.mkdir(exist_ok=True)
    (majority_dir/'config.json').write_text('{"kind":"majority"}')
    for mode in ('embedding', 'structure', 'combined'):
        xt, xv = extract(train, ta, mode).astype(np.float64), extract(val, va, mode).astype(np.float64)
        scaler = StandardScaler().fit(xt)
        xt, xv = scaler.transform(xt), scaler.transform(xv)
        specs = [('lr', c, 0) for c in (0.01, 0.1, 1., 10.)]
        if mode == 'combined':
            specs += [('mlp', 0.01, seed) for seed in (17, 29, 43)]
        for kind, parameter, seed in specs:
            name = f'{mode}-{kind}-{parameter:g}-s{seed}'
            tick = time.perf_counter()
            model = (LogisticRegression(C=parameter, max_iter=600, solver='lbfgs') if kind == 'lr'
                     else MLPClassifier(hidden_layer_sizes=(32,), alpha=parameter, max_iter=100,
                         random_state=seed, early_stopping=False, batch_size=256))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(xt, yt.ravel())
            seconds = time.perf_counter()-tick
            if kind == 'lr':
                scores = model.decision_function(xv)
            else:
                # Rank logits directly: sigmoid rounding can turn confident scores into ties.
                scores = np.maximum(xv @ model.coefs_[0] + model.intercepts_[0], 0)
                scores = (scores @ model.coefs_[1] + model.intercepts_[1]).ravel()
            choices = choose(scores.reshape(-1,4), a)
            size = export(ROOT/name, model, scaler, mode)
            from submission import predict
            portable = predict(val, va, ROOT/name)
            if choices != portable:
                raise RuntimeError(f'Portable inference disagrees: {name}')
            record(name, choices, asset_bytes=size, train_seconds=seconds, seed=seed,
                   features=mode, parameter=parameter, kind=kind,
                   warnings=[str(w.message) for w in caught])
    eligible = [r for r in records if r['name'] != 'first']
    winner = max(eligible, key=lambda r:(r['accuracy'], r['hard_accuracy'], -r['asset_bytes']))
    (ROOT/'selection.json').write_text(json.dumps({'criterion':'validation accuracy, hard accuracy, smaller assets, earlier run',
             'winner':winner, 'total_seconds':time.perf_counter()-start}, indent=2))
    print('SELECTED', winner['name'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
