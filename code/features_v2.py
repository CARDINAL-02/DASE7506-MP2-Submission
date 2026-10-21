"""Extended reference-free candidate features (v2) for the MP2 selector.

Only candidate text, question text and supplied embeddings are used.
"""
import re, math
from collections import Counter
from fractions import Fraction
import numpy as np

STOP = set("""a an the and or but if then than that this these those of to in on for from with without by as is are was were be been being it its he she they them his her their there here we you i not no do does did can could would should may might will shall about into over under again further once only own same so such both each few more most other some any all very just also because while during before after above below up down out off between through""".split())
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
WORD_RE = re.compile(r"[A-Za-z]+")
EQ_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d[\d,]*(?:\.\d+)?)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)")
FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def to_float(ans):
    if ans is None:
        return None
    try:
        v = float(Fraction(ans))
        if not np.isfinite(v) or abs(v) > 1e30:
            return None
        return v
    except Exception:
        return None


def num_tokens(t):
    out = []
    for s in NUM_RE.findall(t):
        try:
            if '/' in s:
                a, b = s.split('/', 1)
                v = float(a) / float(b) if float(b) != 0 else None
            else:
                v = float(s.replace(',', ''))
            if v is not None and math.isfinite(v):
                out.append(v)
        except Exception:
            pass
    return out


def words(t):
    return [w for w in WORD_RE.findall(t.lower()) if w not in STOP and len(w) > 1]


def eq_stats(t):
    checked = correct = 0
    for m in EQ_RE.finditer(t):
        a = float(m.group(1).replace(',', ''))
        b = float(m.group(3).replace(',', ''))
        c = float(m.group(4).replace(',', ''))
        op = m.group(2)
        try:
            if op == '+':
                r = a + b
            elif op == '-':
                r = a - b
            elif op in ('*', '×'):
                r = a * b
            else:
                r = a / b if b != 0 else None
        except ZeroDivisionError:
            r = None
        if r is None:
            continue
        checked += 1
        if abs(r - c) <= 1e-6 * max(1, abs(c)):
            correct += 1
    return checked, correct


def frac_stats(t):
    ok = bad = 0
    for m in FRAC_RE.finditer(t):
        try:
            a = float(Fraction(m.group(1).strip()))
            b = float(Fraction(m.group(2).strip()))
            if b != 0:
                ok += 1
            else:
                bad += 1
        except Exception:
            pass
    return ok, bad


def extract(bank, emb, answers):
    e = emb.astype(np.float32, copy=False)
    norm = np.linalg.norm(e, axis=2, keepdims=True)
    unit = e / np.maximum(norm, 1e-8)
    sim = unit @ unit.transpose(0, 2, 1)
    rows = []
    for i, row in enumerate(bank):
        a = answers[i]
        q = row['question']
        qnums = num_tokens(q)
        qwords = set(words(q))
        texts = [c['text'] for c in row['candidates']]
        lengths = np.array([len(t) for t in texts])
        counts = Counter(v for v in a if v is not None)
        av = [to_float(v) for v in a]
        valid_vals = [v for v in av if v is not None]
        for j, t in enumerate(texts):
            tnums = num_tokens(t)
            tw = set(words(t))
            ov = len(qwords & tw) / max(1, len(qwords))
            hits = 0
            for qv in set(qnums):
                if any(abs(qv - x) <= 1e-9 * max(1, abs(qv)) for x in tnums):
                    hits += 1
            uniq_q = len(set(qnums))
            same = [k for k in range(4) if k != j and a[j] is not None and a[k] == a[j]]
            diff = [k for k in range(4) if k != j and a[k] is not None and a[k] != a[j]]
            allother = [k for k in range(4) if k != j]
            val = av[j]
            checked, correct = eq_stats(t)
            fok, fbad = frac_stats(t)
            ans_occ = 0
            ans_in_last = False
            if a[j] is not None:
                ans_occ = t.count(a[j])
                if val is not None:
                    ans_occ = max(ans_occ, len(re.findall(r'(?<![\d.])' + re.escape(f'{val:g}') + r'(?![\d])', t.replace(',', ''))))
                lines = [ln.strip() for ln in t.strip().splitlines() if ln.strip()]
                if lines:
                    last = lines[-1]
                    ans_in_last = (a[j] in last) or (val is not None and f'{val:g}' in last)
            if val is None:
                dist_mean = dist_med = 0.
                rank = 0.
                is_min = is_max = is_qnum = 0.
            else:
                others = [x for x in valid_vals if x is not None]
                dist_mean = float(np.mean([abs(val - x) for x in others])) if others else 0.
                dist_med = float(abs(val - np.median(others))) if others else 0.
                s = sorted(others)
                rank = (sum(1 for x in s if x < val) + 0.5 * sum(1 for x in s if x == val)) / max(1, len(s))
                is_min = float(val <= min(others)) if others else 0.
                is_max = float(val >= max(others)) if others else 0.
                is_qnum = float(any(abs(val - x) <= 1e-9 * max(1, abs(val)) for x in qnums))
            selfcorr = sum(t.lower().count(w) for w in ('wait', 'actually', 'mistake', 'error', 'incorrect', 'recheck', 'reconsider', 'alternatively'))
            marker_pos = 0.
            if '####' in t:
                marker_pos = t.rfind('####') / max(1, len(t))
            elif '\\boxed' in t:
                marker_pos = t.rfind('\\boxed') / max(1, len(t))
            rows.append([
                float(a[j] is not None), counts.get(a[j], 0) / 4, len(counts) / 4, max(counts.values(), default=0) / 4,
                sum(v is not None for v in a) / 4, np.log1p(len(t)), len(t) / max(lengths.mean(), 1),
                (len(t) - lengths.mean()) / max(lengths.std(), 1), np.log1p(len(tnums)), np.log1p(t.count('\n')),
                float('####' in t), float('\\boxed' in t), np.log1p(t.count('=')), np.log1p(len(q)),
                np.log1p(abs(val)) if val is not None else 0., float(val is not None and val < 0),
                float(val is not None and float(val).is_integer()), float(val is not None and val == 0),
                sim[i, j, allother].mean(), sim[i, j, allother].max(),
                sim[i, j, same].mean() if same else 0., sim[i, j, same].max() if same else 0.,
                sim[i, j, same].sum() if same else 0., sim[i, j, diff].mean() if diff else 0.,
                ov, hits / max(1, uniq_q), np.log1p(ans_occ), float(ans_in_last),
                np.log1p(len(tnums) / max(1, t.count('\n') + 1)), np.log1p(checked),
                correct / max(1, checked), float(checked > 0 and correct == 0),
                np.log1p(fok), float(fbad > 0), np.log1p(len(FRAC_RE.findall(t))),
                np.log1p(t.count('%')), np.log1p(t.count('$')), marker_pos,
                np.log1p(dist_mean), np.log1p(dist_med), rank, is_min, is_max, is_qnum,
                np.log1p(norm[i, j, 0]), np.log1p(selfcorr),
                np.log1p(len(tw) / max(1, len(t.split()))),
                np.log1p(t.count(' therefore ') + t.count(' thus ') + t.count(' hence ') + t.count(' so ')),
                *[float(w in t.lower()) for w in ('therefore', 'however', 'wait', 'check', 'incorrect', 'approximately')],
            ])
    return np.asarray(rows, dtype=np.float32)
