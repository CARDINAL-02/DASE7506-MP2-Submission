"""Reference-free features shared by training and inference."""
from collections import Counter
from fractions import Fraction
import re
import numpy as np
from common import parse_answer


def parsed(bank):
    return [[parse_answer(c['text']) for c in row['candidates']] for row in bank]


def majority(answers):
    result = []
    for a in answers:
        counts = Counter(v for v in a if v is not None)
        result.append(int(max(range(4), key=lambda j: counts.get(a[j], 0))))
    return result


def choose(scores, answers):
    scores = np.array(scores, copy=True)
    for i, a in enumerate(answers):
        valid = [v is not None for v in a]
        if any(valid):
            scores[i, np.logical_not(valid)] = -np.inf
    return [int(i) for i in scores.argmax(axis=1)]


def extract(bank, arrays, mode='combined'):
    e = arrays['embeddings'].astype(np.float32, copy=False)
    n = len(bank)
    if mode == 'embedding':
        return e.reshape(n * 4, 1024)
    unit = e / np.maximum(np.linalg.norm(e, axis=2, keepdims=True), 1e-8)
    sim = unit @ unit.transpose(0, 2, 1)
    answers = parsed(bank)
    rows = []
    for i, row in enumerate(bank):
        a = answers[i]
        counts = Counter(v for v in a if v is not None)
        texts = [c['text'] for c in row['candidates']]
        lengths = np.array([len(t) for t in texts])
        for j, t in enumerate(texts):
            nums = re.findall(r'[-+]?\d+(?:\.\d+)?', t)
            same = [k for k in range(4) if k != j and a[j] is not None and a[k] == a[j]]
            other = [k for k in range(4) if k != j]
            try:
                value = float(Fraction(a[j])) if a[j] is not None else 0.
                value = np.clip(value, -1e30, 1e30)
            except (ValueError, OverflowError):
                value = 0.
            rows.append([
                float(a[j] is not None), counts.get(a[j], 0) / 4,
                len(counts) / 4, max(counts.values(), default=0) / 4,
                sum(v is not None for v in a) / 4,
                np.log1p(len(t)), len(t) / max(lengths.mean(), 1),
                (len(t) - lengths.mean()) / max(lengths.std(), 1),
                np.log1p(len(nums)), np.log1p(t.count('\n')),
                float('####' in t), float('\\boxed' in t),
                np.log1p(t.count('=')), np.log1p(len(row['question'])),
                np.log1p(abs(value)), float(value < 0), float(value % 1 == 0),
                float(value == 0), sim[i, j, other].mean(), sim[i, j, other].max(),
                sim[i, j, same].mean() if same else 0.,
                *[float(word in t.lower()) for word in ('therefore','however','wait','check','incorrect','approximately')],
            ])
    structure = np.asarray(rows, dtype=np.float32)
    if mode == 'structure':
        return structure
    if mode != 'combined':
        raise ValueError(f'Unknown features: {mode}')
    relative = e - e.mean(axis=1, keepdims=True)
    return np.concatenate([e.reshape(n*4, 1024), relative.reshape(n*4, 1024), structure], axis=1)
