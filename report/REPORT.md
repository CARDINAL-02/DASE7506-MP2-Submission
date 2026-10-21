# Selecting Reliable LLM Reasoning with Listwise Verification

DASE7506 MP2 | Individual coursework | Fall 2026

## Abstract

We build an offline CPU selector over four Qwen3-0.6B candidate traces per mathematics question. The prepared final method (v4 linear) combines 54 reference-free structural and lexical features with the supplied 1,024-dimensional trace embeddings and trains a multi-positive listwise logistic model. Official validation is 748/798 (93.7343%); the 355-question answer-diverse subset is 305/355 (85.9155%). The frozen official test result is **391/467 = 83.72591006423983%**, exceeding the requested 83.5% target. Inference uses NumPy only. Original prediction plus independent scoring took 1.323 seconds locally. Assets occupy 25,442 bytes on disk; conservative uncompressed accounting is **26,395 bytes**, including uncompressed NPZ member headers and configuration. This is a prepared local submission, not a claim that online submission has occurred.

## 1. Task and evaluation boundary

Each question has four candidate traces. The selector returns one index per question, in bank order. The fixed scorer parses the chosen candidate's final numeric answer with `common.parse_answer` and accepts fractions, decimals and equivalent forms. It grades the final answer, not the validity of the reasoning. Any correct candidate earns the point, and several candidates can be correct.

The supplied banks contain 4,755 training, 798 validation and 467 test questions. Training references alone define fitting labels; validation is used for development and method selection. Test references are read only by the unchanged official scorer after a method has been frozen. The official parser recognises text after `####` or the last `\boxed{...}`; a candidate with neither marker parses to `None` and can never earn a point, so unparseable candidates are masked before selection.

A label is 1 exactly when the candidate parses to a non-`None` answer equal to the reference's parsed answer. Multiple positives per question are allowed. Training-only preprocessing statistics and weights are saved; question IDs are used only for alignment.

## 2. Method

### 2.1 Features

Each candidate is represented by 54 reference-free features:

- Validity, same-answer count, number of distinct answers, largest answer group and valid fraction.
- Log trace length, relative and within-question standardised length, number of numeric tokens, lines, equations, `####` and `\boxed` markers.
- Log absolute final-answer magnitude, sign, integer and zero flags.
- Mean and maximum cosine similarity to the other candidates, mean/max/sum similarity to same-answer peers, and mean similarity to different-answer peers.
- Word overlap with the question, fraction of question numbers used, occurrences of the final answer, whether the answer appears on the last line, and number density.
- Simple equation checks (`a op b = c`), fraction validity, lexical self-correction and connective flags.

The 54 features are concatenated with the supplied 1,024-dimensional embedding, giving 1,078 inputs. No reference text, answer key or question ID is used at inference.

### 2.2 Listwise objective

For a question with candidate scores `s_j = w · x_j`, we minimise the multi-positive listwise logistic loss

`L = -1/N * sum_i log( sum_{j in C_i} exp(s_j) / sum_k exp(s_k) ) + lambda/2 * ||w||^2`

where `C_i` is the set of correct candidates. This is a differentiable surrogate that maximises total softmax probability assigned to correct candidates. It permits one correct candidate to dominate; it does not require all correct candidates to share probability equally. Earlier v2/v3 experiments used cross-entropy against a uniform distribution over positives, which is a different objective. Groups where all four candidates are correct contribute zero data gradient. `lambda = 0.01` was selected during development. At inference, candidates that parse to `None` are masked before the argmax. Loss, representation and regularization changed across versions, so the observed gain cannot be attributed to the loss alone.

### 2.3 Models compared

We compared fixed-candidate-0, majority vote over parsed answers, a pointwise logistic regression on the initial 27-feature representation, the v2 listwise linear model, a listwise MLP (128 hidden units, three seeds), a pairwise MLP ranker, a validation-weighted ensemble of linear + MLP + embedding-only models, grouped OOF stacking, wider/deeper MLPs, question-SVD/arithmetic features and a full-text TF-IDF model. The main comparison uses the same train/validation split and the official scorer.

## 3. Validation results

| Configuration | Validation | Hard subset |
|---|---|---|
| Fixed candidate 0 | 78.20 | 54.93 |
| Majority vote | 86.22 | 69.01 |
| Initial combined pointwise LR | 92.61 | 83.38 |
| v1 structure only (pointwise) | 90.10 | 77.75 |
| v1 embedding only (pointwise) | 89.60 | 76.62 |
| v4 listwise linear (final) | 93.73 | 85.92 |
| v2 listwise MLP, 128 units | 93.73 | 85.92 |
| Pairwise MLP ranker | 93.11 | 84.51 |
| Later ensemble (linear + MLP + embedding) | 94.36 | 87.32 |

The hard subset contains the 355 validation questions with at least two distinct non-`None` parsed answers. Structure-only and embedding-only rows show that neither evidence source alone matches the combined representation; the listwise v2 model improves over the initial pointwise model by 2.54 percentage points on the hard subset.

The ensemble reaches the highest validation hard accuracy. A grouped five-fold out-of-fold comparison on the 2,114 hard training questions gave 1,712 correct for the linear model (80.9839%) and 1,705 for a proxy ensemble (80.6528%). Standardization is fitted within each training fold, and all four candidates of a question stay together. The proxy uses one MLP seed and 30 epochs; the evaluated final ensemble used three seeds and validation-selected epochs. Consequently this is not exact cross-validation of that final ensemble. The seven-question edge is modest and does not establish statistical superiority. It was additional evidence supporting the simpler linear model, whose validation result remained competitive. Exact counts and proxy limitations are documented in the post-review evidence; the original OOF summary is retained.

Other exploratory results included a 256-unit MLP at 86.48% hard accuracy, a 512-unit MLP at 85.63%, grouped OOF stacking at 87.04%, question-TF-IDF/SVD or answer-arithmetic features at 85.35%, a full-text TF-IDF classifier at 57.18%, and linear bagging at 85.07%. These are development observations on a repeatedly used validation bank, not independent confirmation. Higher complexity did not consistently improve the available evidence. Historical scripts, logs and the OOF score cache are retained in the release research archive; their original desktop path assumptions are distinct from the portable final training/inference commands.

### 3.1 Error patterns

Error rates are higher on questions with three or four distinct answers, although the largest two-answer group still contributes errors. Agreement, length, formatting and local arithmetic checks do not establish whether a trace correctly interprets the problem. No controlled final-answer-only ablation of the final 54-feature model was performed. Earlier models did have restricted information-source controls, reported below; they must not be relabeled as final-model ablations.

Validation error analysis for the linear model: of 355 hard questions, the model is the only correct selector on 75, majority vote is the only correct selector on 15, and both fail on 35. Accuracy is 90.75% on two-answer questions (227), 76.85% on three-answer questions (108) and 80.00% on four-answer questions (20). These imply 21, 25 and 4 errors respectively; they do not by themselves establish the semantic cause of each error.

### 3.2 Historical information controls and uncertainty

Earlier controlled experiments used the same logistic regression C=0.01, train-fitted preprocessing and valid-answer mask: question/final-answer-only features reached 88.10% validation accuracy; surface-only reached 81.95%; their union reached 89.72%; adding supplied embeddings reached 91.73%. The nested embedding addition corrected 19 questions and regressed 3; a descriptive paired bootstrap interval for its gain was [0.88, 3.26] percentage points. These results support additional predictive information beyond that restricted baseline, not proof that reasoning is verified. Feature definitions and data are saved in `runs/supplement/`.

Historical v2 candidate-margin quartiles had accuracies 94.97%, 93.94%, 91.46% and 91.96%, so higher top1-minus-top2 raw logit margin did not monotonically imply higher correctness. Duplicate correct answers can create small candidate margins. On its 99 final-answer disagreements with majority voting, historical v2 was correct on 72 and majority on 17. These are diagnostics of historical v2, not calibrated confidence claims for final v4. The final model has not received a new confidence threshold or rejection policy.

## 4. Final test and evaluation history

| Variant | Selection evidence | Validation | Test |
|---|---|---|---|
| Initial v1 combined pointwise LR | validation | 92.6065 | 81.7987 (382/467) |
| Historical v2 uniform-positive listwise | validation | 93.1078 | 82.2270 (384/467) |
| Historical v3 feature/tree study winner | fixed 36-candidate validation budget | 93.6090 | 81.3704 (380/467) |
| Later multi-positive ensemble | validation hard 87.32 | 94.3609 | 83.0835 (388/467) |
| v4 multi-positive linear (prepared final) | train OOF proxy comparison; validation 85.92 hard | 93.7343 | **83.7259 (391/467)** |

The later ensemble and final linear variant were each frozen before their respective test runs. The ensemble was evaluated first; the grouped training OOF analysis was completed afterwards, followed by freezing and testing the linear model. Prior v1/v2/v3 test results and public peer scores were already known during later development. This is therefore an iterative, test-exposed research history, not one completely blind experiment. Inspection found no test-reference reads in the final fitting code, and the saved linear weights were independently reconstructed from training data exactly. Code and local timestamps cannot prove the absence of all human test-feedback influence; no such claim is made. The prepared final score is exactly the linear model's 391/467 result, never the score of another checkpoint.

## 5. Resources

| Item | Observed value |
|---|---|
| Validation accuracy | 93.7343% (748/798) |
| Validation hard-subset accuracy | 85.9155% (305/355) |
| Official test accuracy | 83.7259% (391/467) |
| Fresh prediction + scoring | 1.323 seconds locally |
| Inference assets, conservative uncompressed | 26,395 bytes / 16,777,216-byte limit |
| Asset files on disk (compressed NPZ + JSON) | 25,442 bytes |
| Linear training time | 10.96 seconds |
| Local CPU | Intel Core i7-10870H @ 2.20 GHz; CPU only |
| Runtime | Python 3.12.14, NumPy 2.5.3; Windows 11 |
| Original recorded thread settings | Library caps 4; no original process-thread monitoring |
| Recommended reproduction settings | All library caps 1; sequential prediction and scoring |

The reference Xeon Platinum 8457C time is not verified locally. Original 1.323-second timing uses recorded four-thread library caps; this alone does not prove an OS-process total of four threads. A separate post-review reproduction uses single-thread library caps and records sampled child-thread peaks and fresh-process timing in `runs_linear/review/integrity.json`. Sampling is not continuous proof. Inference imports NumPy and method sources, not scikit-learn/SciPy. A later hardware observation records 15.780 GiB physical RAM; it is not historical monitoring of the original test.

The 10.96-second training figure describes the final fit, not the entire search. Aggregate historical development/search time was not fully instrumented; available scripts and logs are archived, and no claim is made that the whole project cost only one final training run.

## 6. Reproduction

Prepared final assets are `code/runs_linear/model_bundle/{config.json,weights.npz}`. The original pre-test `freeze.json` is preserved, including its limitation: it hashes the ensemble training script but omits the final `train_linear_v2.py`. The post-review `review/integrity.json` adds that final script's hash, resource accounting, exact train-only reconstruction results and original-freeze checks without pretending to be a pre-test record. The fixed model was independently replayed through the unchanged official scorer and matched all saved selections at 391/467. Reproduction commands use separate output paths; training verification must not overwrite the submitted bundle. NumPy is the only inference dependency.

## 7. Limitations and disclosure

Validation has 798 questions (355 hard), so selection has finite-sample optimism; the test bank has 467 questions, giving a sampling uncertainty of roughly four percentage points for a single accuracy. The features include agreement, length, formatting and lexical signals, so the method may exploit shortcuts rather than verify reasoning. The report does not claim that selected traces have valid intermediate steps. No external dataset or additional pretrained weights are used; questions and references derive from GSM8K under its retained MIT licence, and candidate traces/embeddings were prepared with Qwen3-0.6B as recorded in the course manifest. The student must review and understand this implementation and add their student ID to the submission form; no identity has been fabricated.

AI disclosure: substantive OpenAI Codex assistance and an additional assistant identified by the student as DeepSeek were used for requirements interpretation, feature/loss design, coding, experiments, debugging, analysis and report drafting. The student must confirm this disclosure and understand the work before submitting. Reused materials are the course task, data, embeddings, parser, runner and scorer.
