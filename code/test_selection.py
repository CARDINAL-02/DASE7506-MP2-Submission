"""Contract tests: catch invalid-answer selection, tie errors and fixed-bank assumptions."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from submission import predict


def row(texts, name='unseen'):
    return {'id': name, 'question': 'Synthetic arithmetic question',
            'candidates': [{'index': i, 'text': t} for i, t in enumerate(texts)]}


class SelectionTests(unittest.TestCase):
    def run_bank(self, bank):
        choices = predict(bank, {'embeddings': np.zeros((len(bank), 4, 1024), dtype=np.float32)}, None)
        self.assertEqual(len(choices), len(bank))
        self.assertTrue(all(type(i) is int and 0 <= i < 4 for i in choices))
        return choices

    def test_fraction_equivalence_and_invalid_majority(self):
        bank = [row(['#### 1/2', '#### 0.5', '#### 8', 'No final marker']),
                row(['invalid', 'invalid', '#### 7', 'invalid'], 'another')]
        self.assertEqual(self.run_bank(bank), [0, 2])

    def test_tie_all_invalid_and_empty(self):
        self.assertEqual(self.run_bank([row(['#### 2', '#### 3', '#### 4', '#### 5'])]), [0])
        self.assertEqual(self.run_bank([row(['x'] * 4)]), [0])
        self.assertEqual(self.run_bank([]), [])

    def test_checkpoint_ranks_confident_logits_without_sigmoid_ties(self):
        # A sigmoid in float32 collapses logits 20 and 30 to 1, wrongly choosing index 0.
        bank = [row(['#### 2', '#### 3', 'invalid', '#### 4'])]
        e = np.zeros((1, 4, 1024), dtype=np.float32)
        e[0, :, 0] = [20, 30, 100, 10]
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            p = Path(tmp)
            (p/'config.json').write_text(json.dumps({'kind':'linear','features':'embedding','layers':1}))
            w = np.zeros((1024,1))
            w[0] = 1
            np.savez(p/'weights.npz',mean=np.zeros(1024),scale=np.ones(1024),w0=w,b0=np.zeros(1))
            self.assertEqual(predict(bank,{'embeddings':e},p), [1])


if __name__ == '__main__':
    unittest.main()
