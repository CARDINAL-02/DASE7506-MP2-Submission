# Final v4 reproduction

Prepared model: `submission.py` + `runs_linear/model_bundle`. Official score: **391/467 = 83.72591006423983%**. All following commands run from `code/`. On Linux/macOS use `.venv/bin/python` instead of `.venv/Scripts/python.exe`. In the current desktop workspace the existing interpreter is `../../.venv/Scripts/python.exe`.

## 1. New environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Inference needs only pinned NumPy. For training, install `-r requirements-training.txt`; for PDF generation, install `-r requirements-report.txt`. No network is used by inference after dependencies and course data have been installed.

## 2. Restore and check data

Complete archive users already have the course data. Source-only users must restore the exact `data/` bank from the original course release, retaining its manifest, license and notice. Do not change the manifest to accept different data.

```powershell
.\.venv\Scripts\python.exe check_data.py
.\.venv\Scripts\python.exe -m unittest test_selection.py
```

## 3. Reproduce official score without training

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$elapsed=Measure-Command {
  .\.venv\Scripts\python.exe predict.py --method submission.py --checkpoint runs_linear/model_bundle --split test --output runs_linear/review-reproduced-predictions.json
  if ($LASTEXITCODE -ne 0) { throw 'Prediction failed' }
  .\.venv\Scripts\python.exe score_predictions.py --split test --predictions runs_linear/review-reproduced-predictions.json --output runs_linear/review-reproduced-score.json
  if ($LASTEXITCODE -ne 0) { throw 'Scoring failed' }
}
$elapsed.TotalSeconds
Get-Content runs_linear/review-reproduced-score.json
```

Do not overwrite frozen `test-predictions.json` or `test-score.json`. Expected accuracy is 83.72591006423983. The observed post-review full two-process time was 1.756 seconds on the local i7-10870H; reference-machine eligibility is unverified. Caps are one per compute library, with sequential children and sampled OS-thread peak 3; samples are not continuous proof.

## 4. Reproduce training without replacing the submitted model

In a separate checkout copy, first back up that copy's `runs_linear/train_metrics.json` because the original training script writes this fixed metrics path even when a separate output is supplied:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe train_linear_v2.py --output runs_linear/retrained_bundle
```

The final trainer fits only training inputs/references, uses validation only for metrics, and returns deterministic linear parameters (lambda=0.01). Independent reconstruction during review produced exactly equal mean, scale and weights; its optimizer converged in 50 iterations. This reconstruction did not overwrite the submitted bundle. The original script does not save optimizer status; the post-review reconstruction does.

The earlier `train_v2.py` trains an ensemble comparison and is not the final model. Historical research is archived separately; its scripts may assume the original desktop directory layout. It is not required for direct final training or inference reproduction.

## 5. Evidence and provenance

- `runs_linear/freeze.json`: unchanged pre-test source/asset hashes. It omits `train_linear_v2.py`; do not claim otherwise.
- `runs_linear/review/integrity.json`: post-test addendum hashing the final trainer, checking original hashes, recording train-only reconstruction, audit interface, official replay, timing and asset sizes.
- `runs_linear/review/selection-provenance.json`: exact OOF counts and proxy-ensemble limitations.
- `runs_linear/validation-score.json`, `test-score.json`: original official scores; `review/test-score.json`: independent fixed-model replay.
- `runs_linear/model_bundle/`: 25,442 bytes on disk; 26,395 bytes conservatively uncompressed (NPZ member contents/NPY headers plus JSON). Do not equate compression with asset eligibility.
- `RELEASE_MANIFEST.json`: complete release file hashes including post-freeze documentation. It serves a different purpose from the original pre-test freeze.

Prior experiments and test exposures are disclosed in the report. The proxy OOF uses a single-seed 30-epoch MLP; it is not exact CV of the tested three-seed, selected-epoch ensemble. The final linear model has not been altered during this submission preparation.

## 6. Regenerate report

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe make_report_linear.py
```

The report remains at most 10 pages. The legacy report-generating scripts are historical and should not be used to overwrite the final report. Rebuild release archives after changing any document.

## 7. AI/reuse and online submission

Substantive assistance came from OpenAI Codex and another assistant identified by the student as DeepSeek. An earlier draft named Claude without verification; the student must confirm the final attribution. All student work must be reviewed and understood. The course supplies data/embeddings/parser/runner/scorer; no external dataset or additional pretrained weights were used. Keep LICENSE/NOTICE. No student ID or online submission is fabricated; public links and actual GitHub issue creation remain pending.
