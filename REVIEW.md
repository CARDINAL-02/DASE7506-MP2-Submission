# Independent review and post-freeze documentation erratum

Read-only independent review checked the code against official rules and saved evidence. No critical correctness or leakage issue was found. All 12 frozen Python source hashes and both asset hashes matched; 27 structural / 2,075 combined features, validation selection order, 50,851-byte asset total and 382/467 official test score were verified. No test scoring was rerun during review.

Two documentation issues were corrected after freezing, without changing frozen sources or inference assets:

1. The initial report said learned all-invalid input returns index 0. Actual frozen behavior returns the learned score argmax; exact ties prefer the earliest candidate. The no-checkpoint majority baseline returns 0. Both are legal indices; no candidate can score correctly if all answers parse to None. `code/report_revision.py` regenerates the report with this wording corrected while keeping `make_report.py` identical to the pre-test source hash.
2. `code/requirements-report.txt` pins the PDF-generation dependencies. They are separate from model inference and training requirements.

Regenerate the corrected report from `code/` using Python 3.12:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe report_revision.py
```

The `.venv` path above assumes the environment created inside `code/` by REPRODUCE.md. In the current desktop workspace use `../../.venv/Scripts/python.exe` and install report dependencies there, or use a separate document environment. Report generation does not train, select or score a model. The report file outputs are `report/REPORT.md` and `report/REPORT.pdf`.

The final release's `RELEASE_MANIFEST.json` identifies the complete delivery files, including documentation additions after freezing. `code/runs/freeze.json` identifies the unchanged exact pre-test Python source and inference assets. These are separate purposes; the original `PACKAGE_MANIFEST.json` remains the course-package manifest, not a manifest of student additions.

## Supplemental v2 review

Independent read-only review checked the eight-candidate search, four restricted ablations, listwise masked loss/gradient, pairwise construction, train-only preprocessing, ensemble normalization, validation promotion and separate freezing. No blocking selection or inference issue was found. All 20 v2 frozen source hashes and both asset hashes matched; 50,839 bytes and the saved 384/467 score were verified without rerunning test scoring.

Review caveats are disclosed in the final report:

- Two listwise optimizers reached the 200-iteration cap. Code excludes non-converged runs even though the preregistration did not explicitly state this filter. Including them would not change the winner; no rerun or extension was made.
- `finalize_v2.py` and `hardware_probe.py` use Windows APIs. Direct frozen inference and training are portable; the Windows observer is not required for peers' score reproduction.
- Process-thread samples are evidence, not continuous proof. `backend_info_parent` describes the monitoring/training parent; it is not the inference-child backend list.
- V2 development started after the v1 score was known. The study is not described as a fresh blind benchmark.

The v2-frozen historical `make_report.py` and `report_revision.py` remain unchanged. New `report_final.py` adds controlled ablations, confidence diagnostics and separately labeled v2 results; it produces the nine-page active report. Reproduction instructions now use that script. Original report and metadata are retained as `REPORT-v1.*` and `reproduction-original.json`; later hardware observations have explicit provenance.

## V3 review and failed target

Read-only review checked safe AST evaluation, feature boundaries, portable HGB/ExtraTrees export, the 36-candidate validation-only budget and exact frozen lineage. A plan-path issue in `train_v3.py` was fixed before freezing to support both desktop and archive layouts; the plan content/hash was unchanged.

All v3 frozen source/asset hashes matched, assets total 27,702 bytes, and the plan hash matched the saved protocol. The converged validation winner had 747/798. Its one official frozen test yielded 380/467 = 81.37044967880085%, so `target_met: false` correctly states that 83.5% was not reached. No test scoring was repeated during review. The active pointer follows validation selection despite the worse historical test comparison; no test-driven rollback or expanded search was performed.

Local full evaluation time is 1.785 seconds; sampled child-thread peaks are three, not a continuous compliance certificate. The uncorrected validation interval [-0.75, 1.75] percentage points includes zero. The final report uses these facts and preserves the nine-page v2 report. `report_v3.py` is a post-freeze document addition; all earlier frozen Python files remain unchanged.

## Final v4 submission review (supersedes historical active pointers above)

The prepared submission now uses `submission.py` with `runs_linear/model_bundle`: validation 748/798, test 391/467 (83.72591006423983%). Historical v1/v2/v3 and ensemble results remain evidence, not current submission pointers. No final model source or weight was changed during this preparation.

`runs_linear/review/integrity.json` records independent fixed-model replay (all saved selections matched), exact train-only parameter reconstruction with successful optimizer convergence, and matching original final freeze source/asset hashes. The original freeze omitted `train_linear_v2.py`; its hash is added only in the explicitly post-test addendum. This is not retroactive preregistration. Historical source snapshots must not be assumed to match today's extended submission loader.

Stored assets total 25,442 bytes; conservative uncompressed NPZ members plus configuration total 26,395 bytes. Fresh sequential prediction/scoring took 1.756 seconds with single-thread library caps and sampled child-thread peak three. Sampling is not continuous proof and the reference Xeon runtime remains unverified.

The grouped training OOF comparison is 1712/2114 versus 1705/2114 hard questions. The ensemble OOF proxy differs from the tested ensemble's MLP seeds/epoch selection, and seven questions are not evidence of strong statistical superiority. Scripts and saved OOF predictions are retained as research evidence. Earlier test/leaderboard exposure and later test chronology are disclosed; no completely blind study is claimed.

The report, reproduction instructions, guide, checkpoint links and release manifest are aligned to final v4. Submission website publication, student identity and repository/release URLs remain pending the account holder.
