"""Train the v2 listwise ensemble. Reads only train and validation; never test."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import load_inputs, read_jsonl, DATA, parse_answer
from features import parsed
import features_v2

ROOT = Path('runs_v2')
SEEDS = (0, 1, 2)
HIDDEN = 128
WD = 3e-3
LR = 1e-3
EPOCHS = 40
BATCH = 512
LINEAR_C = 0.01
EMB_C = 0.001
WEIGHTS = (3.0, 2.0, 1.0)  # linear, MLP average, embedding-only


def labels(bank, split):
    refs = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    assert [r['id'] for r in refs] == [r['id'] for r in bank]
    return np.array([[int(v is not None and v == parse_answer(g['answer'])) for v in a]
                     for a, g in zip(parsed(bank), refs)], dtype=np.int64)


def fit_logit(X, y, lam, maxiter=300):
    """Multi-positive listwise logistic model: score = X @ w."""
    K = X.shape[2]
    Xf = X.reshape(-1, K)
    mask = y > 0

    def obj(w):
        s = (Xf @ w).reshape(-1, 4)
        lse = logsumexp(s, axis=1)
        sp = np.where(mask, s, -1e30)
        lse_pos = logsumexp(sp, axis=1)
        loss = np.mean(lse - lse_pos) + 0.5 * lam * w @ w
        q = np.exp(s - lse[:, None])
        p = np.exp(sp - lse_pos[:, None])
        p = np.where(mask, p, 0.0)
        grad = Xf.T @ (q - p).ravel() / len(y) + lam * w
        return loss, grad

    r = minimize(obj, np.zeros(K), jac=True, method='L-BFGS-B',
                 options={'maxiter': maxiter, 'ftol': 1e-10})
    return r.x


def mlp_init(D, H, rng):
    return [rng.standard_normal((D, H)).astype(np.float32) * np.sqrt(2.0 / D),
            np.zeros(H, dtype=np.float32),
            rng.standard_normal((H, 1)).astype(np.float32) * np.sqrt(1.0 / H),
            np.zeros(1, dtype=np.float32)]


def mlp_forward(X, W):
    W1, b1, W2, b2 = W
    Z = X @ W1 + b1
    H = np.maximum(Z, 0)
    return (H @ W2 + b2).reshape(-1, 4), Z, H


def mlp_loss_grad(X, y, W, wd):
    W1, b1, W2, b2 = W
    B = X.shape[0] // 4
    s, Z, H = mlp_forward(X, W)
    mx = s.max(1, keepdims=True)
    e = np.exp(s - mx)
    q = e / e.sum(1, keepdims=True)
    mask = y > 0
    sp = np.where(mask, s, -1e30)
    mp = sp.max(1, keepdims=True)
    ep = np.exp(sp - mp)
    p = np.where(mask, ep / ep.sum(1, keepdims=True), 0.0)
    loss = float(np.mean(np.log(e.sum(1)) + mx[:, 0] - np.log(ep.sum(1)) - mp[:, 0])
                 + 0.5 * wd * (np.sum(W1 * W1) + np.sum(W2 * W2)))
    ds = (q - p) / B
    gW2 = H.T @ ds.reshape(-1, 1) + wd * W2
    gb2 = np.float32(ds.sum()).reshape(1)
    dH = ds.reshape(-1, 1) @ W2.T
    dZ = dH * (Z > 0)
    gW1 = X.T @ dZ + wd * W1
    gb1 = dZ.sum(0)
    return loss, [gW1, gb1, gW2, gb2]


def mlp_train(X, y, seed, epochs=EPOCHS, select=None):
    rng = np.random.default_rng(seed)
    W = mlp_init(X.shape[2], HIDDEN, rng)
    m = [np.zeros_like(w) for w in W]
    v = [np.zeros_like(w) for w in W]
    best = (-1.0, None, 0)
    n = len(y)
    idx = np.arange(n)
    for ep in range(1, epochs + 1):
        rng.shuffle(idx)
        for st in range(0, n, BATCH):
            b = idx[st:st + BATCH]
            _, g = mlp_loss_grad(X[b].reshape(-1, X.shape[2]), y[b], W, WD)
            for i, (p_, g_) in enumerate(zip(W, g)):
                m[i] = 0.9 * m[i] + 0.1 * g_
                v[i] = 0.999 * v[i] + 0.001 * (g_ * g_)
                mh = m[i] / (1 - 0.9 ** ep)
                vh = v[i] / (1 - 0.999 ** ep)
                p_ -= LR * mh / (np.sqrt(vh) + 1e-8)
        if select is not None and (ep % 5 == 0 or ep == 1):
            Xv, yv, hv, vv = select
            sv = mlp_score(Xv, W)
            ch = sv.copy(); ch[~vv] = -np.inf; ch = ch.argmax(1)
            ha = float(yv[np.arange(len(yv)), ch].astype(bool)[hv].mean())
            if ha > best[0]:
                best = (ha, [w.copy() for w in W], ep)
    if select is not None and best[1] is not None:
        return best[1], best[2], best[0]
    return W, epochs, None


def mlp_score(X, W):
    return mlp_forward(X.reshape(-1, X.shape[2]), W)[0]


def valid_mask(bank):
    return np.array([[v is not None for v in row] for row in parsed(bank)])


def score_stats(s):
    return float(s.mean()), float(s.std() + 1e-9)


def main():
    ROOT.mkdir(exist_ok=True)
    start = time.perf_counter()
    train, ta = load_inputs('train')
    val, va = load_inputs('validation')
    yt, yv = labels(train, 'train'), labels(val, 'validation')
    a = parsed(val)
    hard = np.array([len({v for v in row if v is not None}) >= 2 for row in a])

    Xt = np.concatenate([features_v2.extract(train, ta['embeddings'], parsed(train)),
                         ta['embeddings'].reshape(-1, 1024)], axis=1).astype(np.float64)
    Xv = np.concatenate([features_v2.extract(val, va['embeddings'], a),
                         va['embeddings'].reshape(-1, 1024)], axis=1).astype(np.float64)
    scaler = StandardScaler().fit(Xt)
    Xt = scaler.transform(Xt).reshape(-1, 4, Xt.shape[1])
    Xv = scaler.transform(Xv).reshape(-1, 4, Xv.shape[1])
    Xt32 = Xt.astype(np.float32)
    Xv32 = Xv.astype(np.float32)

    lin_w = fit_logit(Xt, yt, LINEAR_C)
    lin_t = (Xt.reshape(-1, Xt.shape[2]) @ lin_w).reshape(-1, 4)
    lin_v = (Xv.reshape(-1, Xv.shape[2]) @ lin_w).reshape(-1, 4)
    lin_mu, lin_sd = score_stats(lin_t)

    vv = valid_mask(val)
    mlps, epochs_used = [], []
    for seed in SEEDS:
        W, ep, ha = mlp_train(Xt32, yt, seed, select=(Xv32, yv, hard, vv))
        mlps.append(W); epochs_used.append(ep)
        print('mlp seed', seed, 'selected epoch', ep, 'val hard %.2f' % (ha * 100), flush=True)

    Et = ta['embeddings'].reshape(-1, 1024).astype(np.float64)
    Ev = va['embeddings'].reshape(-1, 1024).astype(np.float64)
    esc = StandardScaler().fit(Et)
    Et = esc.transform(Et)
    Ev = esc.transform(Ev)
    emb_w = fit_logit(Et.reshape(-1, 4, 1024), yt, EMB_C)
    emb_t = (Et @ emb_w).reshape(-1, 4)
    emb_v = (Ev @ emb_w).reshape(-1, 4)
    emb_mu, emb_sd = score_stats(emb_t)

    mlp_mu, mlp_sd = [], []
    for W in mlps:
        st = mlp_score(Xt32, W)
        mlp_mu.append(score_stats(st)[0])
        mlp_sd.append(score_stats(st)[1])

    def ensemble(s_lin, s_mlp, s_emb):
        zl = (s_lin - lin_mu) / lin_sd
        zm = np.mean([(s - mu) / sd for s, mu, sd in zip(s_mlp, mlp_mu, mlp_sd)], axis=0)
        ze = (s_emb - emb_mu) / emb_sd
        return (WEIGHTS[0] * zl + WEIGHTS[1] * zm + WEIGHTS[2] * ze) / sum(WEIGHTS)

    s_mlp_v = [mlp_score(Xv32, W) for W in mlps]
    ens_v = ensemble(lin_v, s_mlp_v, emb_v)
    choice = ens_v.argmax(1)
    ok = yv[np.arange(len(yv)), choice].astype(bool)
    metrics = {'validation_accuracy': float(ok.mean() * 100),
               'validation_hard_accuracy': float(ok[hard].mean() * 100),
               'correct': int(ok.sum()), 'n': len(yv),
               'hard_correct': int(ok[hard].sum()), 'hard_n': int(hard.sum()),
               'train_seconds': time.perf_counter() - start}

    bundle = ROOT / 'model_bundle'
    bundle.mkdir(parents=True, exist_ok=True)
    arrays = {'mean': scaler.mean_, 'scale': scaler.scale_, 'lin_w': lin_w,
              'lin_mu': lin_mu, 'lin_sd': lin_sd,
              'emb_mean': esc.mean_, 'emb_scale': esc.scale_, 'emb_w': emb_w,
              'emb_mu': emb_mu, 'emb_sd': emb_sd}
    for i, W in enumerate(mlps):
        arrays[f'mlp{i}_w1'], arrays[f'mlp{i}_b1'], arrays[f'mlp{i}_w2'], arrays[f'mlp{i}_b2'] = W
        arrays[f'mlp{i}_mu'], arrays[f'mlp{i}_sd'] = mlp_mu[i], mlp_sd[i]
    np.savez_compressed(bundle / 'weights.npz', **arrays)
    config = {'kind': 'ensemble_v2', 'features': 'v2emb', 'hidden': HIDDEN, 'seeds': list(SEEDS),
              'linear_lambda': LINEAR_C, 'emb_lambda': EMB_C, 'epochs_per_seed': epochs_used,
              'weight_decay': WD, 'learning_rate': LR, 'batch': BATCH,
              'ensemble_weights': list(WEIGHTS), 'hard_subset_definition': '>=2 distinct non-None parsed answers'}
    (bundle / 'config.json').write_text(json.dumps(config, indent=2))
    (ROOT / 'train_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
