# DASE7506 MP2 - prepared final submission

**v4 linear: official full-test accuracy 83.72591006423983% (391/467).** Validation: 748/798 (93.7343%); answer-diverse validation subset: 305/355 (85.9155%). Local online submission has not been performed.

The model uses 54 reference-free features plus 1,024 supplied embedding coordinates, with a multi-positive probability-mass listwise loss (lambda=0.01). Inference is offline, CPU, NumPy only. Disk assets are 25,442 bytes; conservative uncompressed assets are **26,395 bytes**, safely below 16 MiB.

- [Final report](report/REPORT.pdf) / [editable report](report/REPORT.md)
- [Frozen checkpoint](code/runs_linear/model_bundle/) / [official score JSON](code/runs_linear/test-score.json)
- [Exact reproduction commands](code/REPRODUCE.md) / [submission checklist](SUBMISSION.md)
- [Original pre-test freeze](code/runs_linear/freeze.json) / [post-review integrity addendum](code/runs_linear/review/integrity.json)
- [Training implementation](code/train_linear_v2.py) / [feature implementation](code/features_v2.py)

## Three commands to reproduce without retraining

From this repository root, use Python 3.12 in an activated virtual environment. In PowerShell set:

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
```

```powershell
python -m pip install -r code/requirements.txt
python code/predict.py --method code/submission.py --checkpoint code/runs_linear/model_bundle --split test --output code/runs_linear/review-reproduced-predictions.json
python code/score_predictions.py --split test --predictions code/runs_linear/review-reproduced-predictions.json --output code/runs_linear/review-reproduced-score.json
```

The submitted value is the official scorer's `accuracy`, not a rounded or independently computed substitute. Source/checkpoint must match this version.

## Data, resources and evidence

The small submission/source archives omit supplied bank JSONL/NPZ files. Restore `code/data/` from the [original course release](https://github.com/xudongwu-0/DASE7506-MP2), matching the retained `manifest.json`, then run `python code/check_data.py`. Preserve [LICENSE](code/data/LICENSE.txt) and [NOTICE](code/data/NOTICE.txt). The complete archive contains the data already.

Original recorded prediction+scoring was 1.323 seconds with library caps of four. Later independent reproduction took 1.756 seconds with single-thread library caps, sequential children and sampled process-thread peaks of 3. Sampling does not continuously certify thread counts. Reference Xeon timing is unverified. Exact train-only reconstruction reproduced every learned array, with successful optimizer convergence; details are in the addendum.

The original pre-test freeze omits the final training script. Its hash is supplied transparently in the post-test addendum rather than retroactively changing history. OOF comparison favored linear by 7/2,114 hard training questions but compared a proxy one-seed MLP ensemble, not the final three-seed ensemble. Prior test results and public leaderboard exposure are disclosed in the report.

Neither `.gitignore` excludes `code/runs_linear/`. Check global ignore rules. Publish the small final checkpoint and evidence with the source; place large ZIPs in release assets. Historical research archives have original desktop path assumptions and are not required to reproduce the final model.

## AI and reuse

Substantive assistance from OpenAI Codex and another assistant identified by the student as DeepSeek covered design, implementation, experimentation and reporting. The student must confirm this disclosure and understand the work. No external dataset or additional pretrained weights were used. Publication, student ID and actual course submission remain the account holder's responsibility.
