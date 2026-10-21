"""Restricted ablation features. No learned state or embedding access."""
from collections import Counter
from fractions import Fraction
import re
import numpy as np
from common import parse_answer

WORDS = ('therefore','however','wait','check','incorrect','approximately')


def restricted(bank, mode):
    rows = []
    for row in bank:
        texts = [c['text'] for c in row['candidates']]
        answers = [parse_answer(t) for t in texts]
        counts = Counter(a for a in answers if a is not None)
        lengths = np.array([len(t) for t in texts],dtype=float)
        qnums = [float(n.replace(',','')) for n in re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',row['question'])]
        qscale = sum(abs(v) for v in qnums)
        for j, t in enumerate(texts):
            a = answers[j]
            value = float(Fraction(a)) if a is not None else 0.
            value = float(np.clip(value,-1e30,1e30))
            answer = [float(a is not None),counts.get(a,0)/4,len(counts)/4,
                max(counts.values(),default=0)/4,sum(v is not None for v in answers)/4,
                np.log1p(abs(value)),float(value<0),float(value%1==0),float(value==0),
                np.log1p(len(row['question'])),np.log1p(len(qnums)),np.log1p(qscale),
                np.log1p(abs(value)/(1+qscale)),float(value in qnums),
                np.log1p(min([abs(value-q) for q in qnums],default=abs(value))),
                float('/' in a) if a is not None else 0.]
            surface = [np.log1p(len(t)),len(t)/max(lengths.mean(),1),
                (len(t)-lengths.mean())/max(lengths.std(),1),
                np.log1p(len(re.findall(r'[-+]?\d+(?:\.\d+)?',t))),
                np.log1p(t.count('\n')),float('####' in t),float('\\boxed' in t),
                np.log1p(t.count('=')),*[float(w in t.lower()) for w in WORDS]]
            if mode == 'answer':
                rows.append(answer)
            elif mode == 'surface':
                rows.append(surface)
            elif mode == 'answer_surface':
                rows.append(answer+surface)
            else:
                raise ValueError(mode)
    return np.asarray(rows,dtype=np.float64)
