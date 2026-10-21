"""v3 features: v2 + group structure, within-question gaps/ranks, interactions,
question-condition matching and answer-plausibility features."""
import re
from collections import Counter
import numpy as np
from feats2 import extract as extract2, to_float, num_tokens, EQ_RE, FRAC_RE

COND = ('each', 'every', 'per', 'both', 'twice', 'double', 'half', 'remaining',
        'total', 'altogether', 'including', 'excluding', 'left', 'more than', 'less than')
BASE = (5, 30, 25, 26, 18, 14, 37, 1)  # len, eq_frac, qnum_ov, ans_occ, simmean, logabs, marker, samecnt


def rank_pct(vals):
    a = np.asarray(vals, dtype=float)
    order = a.argsort()
    ranks = np.empty(4, dtype=float)
    ranks[order] = np.arange(4, dtype=float) / 3.0
    return ranks


def extract(bank, emb, answers):
    base = extract2(bank, emb, answers)
    extra = []
    for i, row in enumerate(bank):
        a = answers[i]
        q = row['question'].lower()
        texts = [c['text'] for c in row['candidates']]
        counts = Counter(v for v in a if v is not None)
        sizes = list(counts.values()) if counts else [0]
        maxsize = max(sizes)
        ngroups = len(counts)
        ent = -sum((c / 4.0) * np.log(c / 4.0) for c in sizes) if counts else 0.0
        qnums = num_tokens(q)
        qn_max = max(qnums) if qnums else 0.0
        qn_min = min(qnums) if qnums else 0.0
        qn_sum = sum(qnums) if qnums else 0.0
        qn_mean = qn_sum / max(1, len(qnums))
        qconds = [w for w in COND if w in q]
        qmany = float('how many' in q)
        qmuch = float('how much' in q or 'cost' in q or 'price' in q)
        qpct = float('%' in q or 'percent' in q)
        qfrac = float('fraction' in q or 'ratio' in q)
        # ranks of base columns
        ranks = [rank_pct(base[i * 4:i * 4 + 4, c]) for c in BASE]
        for j in range(4):
            val = to_float(a[j])
            g = counts.get(a[j], 0) if a[j] is not None else 0
            same = [k for k in range(4) if k != j and a[j] is not None and a[k] == a[j]]
            diff = [k for k in range(4) if k != j and a[k] is not None and a[k] != a[j]]
            x = base[i * 4 + j]
            gap_same = []
            gap_diff = []
            for c in BASE:
                if same:
                    gap_same.append(float(x[c] - base[i * 4 + np.array(same), c].mean()))
                else:
                    gap_same.append(0.0)
                if diff:
                    gap_diff.append(float(x[c] - base[i * 4 + np.array(diff), c].mean()))
                else:
                    gap_diff.append(0.0)
            # text-derived
            t = texts[j]
            tl = t.lower()
            tconds = [w for w in qconds if w in tl]
            nbox = t.count('\\boxed')
            boxed_vals = re.findall(r'\\boxed\{([^{}]+)\}', t)
            boxed_nums = []
            for b in boxed_vals:
                m = re.search(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', b.replace(',', ''))
                if m:
                    v = to_float(m.group(0))
                    if v is not None:
                        boxed_nums.append(v)
            lastbox = boxed_nums[-1] if boxed_nums else None
            eqs = list(EQ_RE.finditer(t))
            lasteq = eqs[-1].group(4) if eqs else ''
            try:
                lasteq = float(lasteq.replace(',', ''))
            except Exception:
                lasteq = None
            tail = ''
            if '####' in t:
                tail = t.rsplit('####', 1)[1]
            elif '\\boxed' in t:
                tail = t.rsplit('\\boxed', 1)[1]
            tnums = num_tokens(t)
            uniq_ratio = len(set(np.round(tnums, 8))) / max(1, len(tnums))
            extra.append([
                float(g == 1), float(g == 2), float(g == 3), float(g == 4),
                float(ngroups == 2), float(ngroups == 3), float(ngroups == 4),
                float(g == maxsize and g > 0), ent, ngroups / 4.0, maxsize / 4.0,
                *[float(r[j]) for r in ranks],
                *gap_same, *gap_diff,
                float(g == 1) * gap_same[0], float(g == 1) * gap_same[1],
                float(g == 1) * gap_same[2], float(g == 1) * gap_same[4],
                float(g == maxsize) * gap_diff[0], float(g == maxsize) * gap_diff[1],
                float(g == maxsize) * gap_diff[2], float(g == maxsize) * gap_diff[4],
                x[1] * x[30], x[1] * x[25], x[1] * x[18], ngroups / 4.0 * x[1],
                ngroups / 4.0 * x[30],
                len(tconds) / max(1, len(qconds)) if qconds else 0.0,
                float(bool(qconds) and len(tconds) == len(qconds)),
                len(qconds) / 8.0,
                qmany, qmuch, qpct, qfrac,
                float(qmany > 0 and x[16] < 0.5), float(qmany > 0 and (val is not None and val < 0)),
                float(qpct > 0 and val is not None and 0 <= val <= 100),
                np.log1p(abs(val) / max(1.0, qn_max)) if val is not None else 0.0,
                np.log1p(abs(val) / max(1.0, abs(qn_mean))) if val is not None else 0.0,
                float(val is not None and val > qn_max), float(val is not None and val < qn_min),
                float(val is not None and abs(val - qn_sum) <= 1e-6 * max(1, abs(qn_sum))),
                np.log1p(nbox), float(nbox > 1),
                float(lastbox is not None and val is not None and abs(lastbox - val) <= 1e-9 * max(1, abs(val))),
                float(lasteq is not None and val is not None and abs(lasteq - val) <= 1e-9 * max(1, abs(val))),
                np.log1p(len(tail)), uniq_ratio,
                float('answer is' in tl or 'the answer' in tl),
                float('####' in t and ('\\boxed' in t)),
            ])
    return np.concatenate([base, np.asarray(extra, dtype=np.float32)], axis=1)
