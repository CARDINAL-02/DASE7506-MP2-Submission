"""Nine-page evidence update. Frozen v1/v2 Python files and assets remain byte-identical."""
from pathlib import Path
import shutil

for ext in ('pdf','md'):
    p=Path('../report')/f'REPORT.{ext}'
    backup=p.with_name(f'REPORT-v1.{ext}')
    if not backup.exists():
        shutil.copy2(p,backup)

source=Path('make_report.py').read_text(encoding='utf-8')
replacements={
    "p('DASE7506 MP2 | Individual coursework | Fall 2026')":
    "p('DASE7506 MP2 | Individual coursework | Fall 2026 | Updated two-stage study')",
    "The frozen method is {winner[\"name\"]}.":"The stage-1 (v1) frozen method is {winner[\"name\"]}.",
    'the one final official test score is':'its stage-1 official test score is',
    'The frozen asset files occupy':'The stage-1 frozen asset files occupy',
    "title('5. Frozen test and resource accounting')":"title('5. Stage-1 test and resource accounting')",
    'Inference reproduction requires the frozen source files and runs/model_bundle, plus the course data and NumPy. Run predict.py with --method submission.py --checkpoint runs/model_bundle --split test, then score_predictions.py on its output. REPRODUCE.md supplies exact installation, training and evaluation commands. runs/test-score.json contains the official score; runs/reproduction.json contains runtime and asset records. Hashes identify the exact matching source and assets.':
    'Active v2 reproduction requires submission.py, features.py and runs/model_bundle_v2, plus course data and NumPy. Run predict.py with --checkpoint runs/model_bundle_v2 --method submission.py --split test, then the official scorer. runs/test-v2-score.json and reproduction-v2.json hold matching results and resources. Historical model_bundle and test-score.json reproduce v1 only. REPRODUCE.md gives exact commands; each stage has a separate pre-test source/asset hash manifest.',
    "    title('1. Data and evaluation boundary')":
    "    p('The active submission is the separately frozen v2 listwise linear ranker: validation 743/798 (93.1078%), test 384/467 (82.2270%). Sections 1-6 preserve stage-1 evidence; Sections 7-9 add controlled information ablations, uncertainty and the preregistered follow-up. V2 was selected solely by validation; the prior v1 test result was already known and is disclosed.')\n    title('1. Data and evaluation boundary')",
    'Learned scoring masks invalid parsed answers before argmax, and also returns 0 for an all-invalid group.':
    'Learned scoring masks invalid parsed answers when any valid candidate exists. For an all-invalid group it returns the learned argmax (earliest on an exact tie), which is legal but cannot earn a point.',
    'No extra search was added after inspecting results.':
    'This was the complete stage-1 grid. A separate fixed-budget follow-up is reported in Section 9, after prior test exposure.',
    'A full-text versus final-answer-only controlled representation experiment was not performed, so no claim is made that the gain is caused specifically by reading reasoning. Cosine summaries and text formatting remain possible shortcuts.':
    'Section 7 adds nested final-answer and surface-feature controls. They quantify predictive information, but do not establish sound reasoning verification; cosine summaries and formatting remain possible shortcuts.',
    'No test-based model change follows.':
    'V1 remains unchanged. Section 9 reports a separately frozen, validation-selected v2; no test-based rollback or further search follows.',
    'OMP/OpenBLAS/MKL/NumExpr each capped at 4; one process at a time':
    'V1: library caps 4, sequential processes. V2: caps 1, sampled OS-thread peaks 3.',
    'Hardware probing of physical RAM was unavailable in the sandbox, so RAM capacity is not claimed.':
    'A later Windows API observation records 16,943,685,632 physical bytes (15.780 GiB RAM) and confirms the CPU model; it is not retroactive monitoring of the v1 test run.',
    'Candidate-level training does not directly optimize the grouped selection objective.':
    'Stage-1 candidate-level training does not directly optimize the grouped objective; v2 tests a listwise objective. Its small validation gain has substantial uncertainty.',
    'The frozen test result is reported without post-test tuning.':
    'Both stages are disclosed: the follow-up began after the v1 test was known, used only training/validation for fitting and selection, and was frozen before its one test evaluation.',
}
for old,new in replacements.items():
    if source.count(old)!=1:
        raise RuntimeError(f'Unexpected frozen report template: {old}')
    source=source.replace(old,new)

extra='''
    page()
    title('7. Controlled information-source ablations')
    supp=load('supplement/results.json')
    paired=load('supplement/paired.json')
    p('These post-stage-1 experiments use train-only fitting and validation-only analysis. Four ablations are excluded from challenger selection. Each uses L2 logistic regression, C=0.01, identical candidate labels, train-fitted standardization, valid-answer masking and earliest-index tie handling. They add no test evaluation.')
    ablations=[e for e in supp['records'] if e['role']=='ablation']
    table(['Information available','Dims','All (%)','Hard (%)'],
      [[e['name'],e['dimensions'],f'{e["accuracy"]:.2f}',f'{e["hard_accuracy"]:.2f}'] for e in ablations],
      [250,50,80,80])
    p('Answer-only (16 features) reads the question and parsed final answers only: validity/consensus, magnitude, sign, integer/zero/fraction flags, question length and numeric statistics, and answer-to-question-number relations. It never uses reasoning-body text, candidate formatting or embeddings. It is a restricted numeric/question baseline, not a fully expressive semantic question-answer model.')
    p('Surface-only (14 features) uses trace length, group-relative length, number/line/equality counts, final-marker flags and six lexical flags. It excludes answer values, agreement, question content and embeddings. Official answer parsing is still used for the common invalid-answer mask. It is therefore a surface scorer plus a shared legality policy, not literally a system with no answer parsing.')
    p('Answer+surface is the nested 30-feature union; adding the 1,024 raw embedding coordinates yields 1,054 features. These two nested comparisons isolate added input groups under the same learner and C. Standardization and the fixed regularizer still interact with dimensionality, so this is not a causal proof that reasoning was verified.')
    p('Adding surface features improves validation from 88.10% to 89.72% (+1.63 points). Adding embeddings on top improves it to 91.73% (+2.01 points): 19 questions improve and 3 regress; the descriptive paired bootstrap interval is [0.88, 3.26] points. Thus trace representations provide additional predictive evidence beyond this restricted final-answer/surface baseline. Surface-only reaches 81.95%, showing considerable shortcut signal, but underperforms majority voting (86.22%).')
    p('The original structure model (90.10%) also includes embedding cosine summaries and a different answer-feature set; combined v1 reaches 92.61%. Those are not strict nested versions of the new controls. No claim is made that all their gains come from reasoning semantics. Saved evidence: supplement/results.json and paired.json; information-boundary unit tests ensure restricted features ignore forbidden content.')
    page()
    title('8. Uncertainty and disagreement on validation')
    u1,u2=load('supplement/uncertainty-v1.json'),load('supplement/uncertainty-v2.json')
    p('For each frozen model, candidate margin is the highest minus second-highest valid raw logit. The 795 questions with at least two valid candidates are split into four observed validation-margin quartiles. The other 3 questions have exactly one valid candidate and are all correct. There are no all-invalid validation groups. Bins are descriptive, not trained calibration or selected rejection thresholds.')
    table(['Margin quartile','N','V1 margin range','V1 acc.','V2 margin range','V2 acc.'],
      [[f'Q{i+1}',q1['n'],f'{q1["lower"]:.3f} - {q1["upper"]:.3f}',f'{q1["accuracy"]:.2f}%',
        f'{q2["lower"]:.3f} - {q2["upper"]:.3f}',f'{q2["accuracy"]:.2f}%']
        for i,(q1,q2) in enumerate(zip(u1['quartiles'],u2['quartiles']))],
      [70,30,110,65,110,65])
    p('Higher margin does not correspond monotonically to higher correctness: v1 Q1 is 96.98% correct while Q4 is 89.45%; v2 is 94.97% versus 91.96%. Multiple correct candidates sharing an answer can have almost equal scores, making a small candidate margin compatible with high answer certainty. Margin therefore does not support a useful confidence/rejection claim here. The two models have separate quartile cutoffs and need not put the same questions in each row.')
    table(['Disagreement definition','V1 N','V1 / vote acc.','V2 N','V2 / vote acc.'],
      [[label,u1[key]['n'],f'{u1[key]["model_accuracy"]:.2f} / {u1[key]["majority_accuracy"]:.2f}',
        u2[key]['n'],f'{u2[key]["model_accuracy"]:.2f} / {u2[key]["majority_accuracy"]:.2f}']
       for label,key in [('Different selected index','index_disagreement'),('Different final answer','answer_disagreement')]],
      [155,45,105,45,100])
    p('Index disagreement overstates substantive disagreement because different candidates may share a final answer. On the 87 v1 answer-disagreement questions, the model is correct on 63 (72.41%) and majority on 12 (13.79%). For v2, 99 answer disagreements give 72 versus 17 correct (72.73% versus 17.17%). These diagnostic subsets are selected by disagreement and do not represent unconditional accuracy.')
    p('All uncertainty and disagreement calculations use only validation references, after frozen model selection. They neither tune the model nor change test predictions. No calibrated confidence probability, coverage guarantee or new test result is claimed. Exact cutoffs, counts and definitions are saved in uncertainty-v1.json and uncertainty-v2.json.')
    page()
    title('9. Fixed-budget follow-up and active v2')
    v2=load('reproduction-v2.json')
    t2=load('test-v2-score.json')
    p('SUPPLEMENT_PLAN.md fixed eight challengers before this follow-up ran: three listwise penalties, three pairwise C values, one training-hard-group-weighted LR and one fixed equal fusion. Train and validation alone are read by the search. No fitting uses validation inputs/references. Promotion requires strictly more whole-validation correct answers than v1; P1 ablations cannot win. V1 test exposure was already known, so this is explicitly a post-test-exposure study, not a fresh blind benchmark.')
    table(['Challenger','All (%)','Hard (%)','Fit (s)','Converged'],
      [[e['name'],f'{e["accuracy"]:.2f}',f'{e["hard_accuracy"]:.2f}',f'{e["fit_seconds"]:.2f}',
        str(e['success']) if 'success' in e else 'not recorded'] for e in supp['records'] if e['role']=='challenger'],[160,65,65,65,105])
    p('V2 is a linear listwise softmax ranker with penalty 0.1. Per question the target is uniform over all correct candidates; invalid candidates are masked. The mean cross-entropy plus penalty times squared weight norm / 2 is minimized with L-BFGS (124 iterations, converged). It reuses v1 combined features and train-only standardization; portable affine inference and the official interface remain unchanged. Pairwise training uses 6,516 positive/negative valid pairs; weighted LR weights training hard groups by 3; fusion logit scales come only from training inputs.')
    p('Two listwise runs reached the fixed 200-iteration cap. Code excludes non-converged runs, an eligibility detail absent from the preregistration; including them would still select penalty 0.1. This discrepancy is disclosed, not repaired by extending the search. Ensemble fitting time is zero because members were already trained; the supplemental total wall time includes feature work, scaling and evaluation.')
    p('Validation improves 739 to 743 correct (+0.50 points); paired v2-only/v1-only counts are 10/6, with descriptive 95% interval [-0.50, 1.50] points. The small gain is not statistically established. V2 was frozen before one test evaluation and kept independently of the test comparison: 384/467 (82.226981%), versus historical v1 382/467. No further search or test-based rollback is performed.')
    p(f'Active assets: {v2["asset_bytes"]:,} bytes, runs/model_bundle_v2. Complete two-process test time: {v2["test_total_seconds"]:.3f} seconds. Each child inherits one-thread BLAS/OpenMP settings; prediction and scoring run sequentially. Windows OS-thread sampling every ~10 ms observed peaks of 3 in each child. Sampling is evidence, not continuous proof; reference-machine eligibility is unverified. A separate v1 validation probe at four BLAS threads observed 7 OS threads, so the original cap settings alone do not prove a four-thread process limit.')
    p('CPU/RAM are backed by a later hardware.json observation; reproduction-original.json preserves original v1 metadata. backend_info_parent in reproduction-v2.json is the monitoring/training parent, not the inference children. Finalize scripts and hardware observer are Windows-only; checkpoint inference and training are otherwise portable. freeze-v2.json identifies pre-test source/assets; documentation-only additions are covered by RELEASE_MANIFEST.json. The root README and REPRODUCE.md point to v2; v1 records remain historical evidence.')
'''
needle='    def footer(canvas,doc):'
assert source.count(needle)==1
source=source.replace(needle,extra+'\n'+needle)
exec(compile(source,'make_report.py (post-freeze evidence update)','exec'),{'__name__':'__main__'})
