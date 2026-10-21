"""Validation-only evidence, paired comparisons and errors. No test tuning."""
import json
from pathlib import Path
import numpy as np
from common import load_inputs
from features import parsed
from train import labels


def main():
    root = Path('runs')
    records = json.loads((root/'experiments.json').read_text())
    winner = json.loads((root/'selection.json').read_text())['winner']['name']
    bank, _ = load_inputs('validation')
    y = labels(bank,'validation')
    a = parsed(bank)
    def choices(name):
        return np.array(json.loads((root/f'{name}-validation.json').read_text())['selected'])
    c = choices(winner)
    b = choices('majority')
    ok, base = y[np.arange(len(y)),c].astype(bool), y[np.arange(len(y)),b].astype(bool)
    rng = np.random.default_rng(20261021)
    delta = ok.astype(float)-base.astype(float)
    samples = np.array([delta[rng.integers(0,len(y),len(y))].mean()*100 for _ in range(10000)])
    examples = []
    for kind, mask in [('improvement',ok & ~base),('regression',base & ~ok),('both_wrong',~ok & ~base)]:
        for i in np.flatnonzero(mask)[:3]:
            examples.append({'kind':kind,'id':bank[i]['id'],'question':bank[i]['question'],
                'answers':a[i], 'selected':int(c[i]), 'majority':int(b[i]),
                'correct_indices':np.flatnonzero(y[i]).tolist(),
                'selected_text':bank[i]['candidates'][c[i]]['text']})
    mlp = [r['accuracy'] for r in records if r.get('kind')=='mlp']
    result = {'winner':winner,'winner_only_correct':int((ok & ~base).sum()),
              'majority_only_correct':int((base & ~ok).sum()),'both_wrong':int((~ok & ~base).sum()),
              'paired_delta_pp':float(delta.mean()*100),'paired_bootstrap_95':np.quantile(samples,[.025,.975]).tolist(),
              'bootstrap_seed':20261021,'bootstrap_samples':10000,
              'mlp_seed_accuracy_mean':float(np.mean(mlp)),'mlp_seed_accuracy_std':float(np.std(mlp)),
              'examples':examples}
    (root/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))


if __name__ == '__main__':
    main()
