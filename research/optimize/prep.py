"""Prepare train/validation feature caches. Reads only train/validation."""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'DASE7506-MP2-main' / 'code'))
from common import load_inputs, read_jsonl, DATA, parse_answer
from features import parsed, extract

OUT = Path(__file__).resolve().parent / 'cache'
OUT.mkdir(exist_ok=True)

def labels(bank, split):
    refs = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    assert [r['id'] for r in refs] == [r['id'] for r in bank]
    return np.array([[int(v is not None and v == parse_answer(g['answer'])) for v in a]
                     for a, g in zip(parsed(bank), refs)], dtype=np.int8)

for split in ('train','validation'):
    bank, arrays = load_inputs(split)
    y = labels(bank, split)
    answers = parsed(bank)
    hard = np.array([len({v for v in a if v is not None}) >= 2 for a in answers])
    np.savez_compressed(OUT/f'{split}_base.npz',
        emb=arrays['embeddings'].astype(np.float32),
        y=y, hard=hard)
    np.save(OUT/f'{split}_parsed.npy', np.array(answers, dtype=object), allow_pickle=True)
    print(split, len(bank), 'hard', int(hard.sum()), 'positives', int(y.sum()), flush=True)

# base features for model experiments
for split in ('train','validation'):
    bank, arrays = load_inputs(split)
    for mode in ('embedding','structure','combined'):
        t=time.time(); X=extract(bank, arrays, mode); print(split,mode,X.shape,'%.1fs'%(time.time()-t),flush=True)
        np.save(OUT/f'{split}_{mode}.npy', X)
