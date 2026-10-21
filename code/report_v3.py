"""Add the failed-target v3 study without modifying any frozen source or inference asset."""
from pathlib import Path
import shutil

for ext in ('pdf','md'):
    p=Path('../report')/f'REPORT.{ext}'
    backup=p.with_name(f'REPORT-v2.{ext}')
    if not backup.exists():
        shutil.copy2(p,backup)

template=Path('report_final.py').read_text(encoding='utf-8')
cut=template.rfind('exec(compile(source,')
assert cut>0
namespace={}
exec(compile(template[:cut],'historical report assembly','exec'),namespace)
source=namespace['source']
changes={
    'Updated two-stage study':'Updated three-stage study; 83.5% target not met',
    'The active submission is the separately frozen v2 listwise linear ranker: validation 743/798 (93.1078%), test 384/467 (82.2270%). Sections 1-6 preserve stage-1 evidence; Sections 7-9 add controlled information ablations, uncertainty and the preregistered follow-up. V2 was selected solely by validation; the prior v1 test result was already known and is disclosed.':
    'The latest validation-selected v3 scores 747/798 (93.6090%) on validation but 380/467 (81.3704%) on test. The requested 83.5% test target was NOT met. Historical v2 scored 384/467 (82.2270%). Sections 1-6 preserve v1 evidence; 7-9 describe v2; Section 10 reports the fixed-budget v3 failure. No test-driven rollback or further search was made. Prior test scores and public leaderboard exposure are disclosed.',
    'Active v2 reproduction requires submission.py, features.py and runs/model_bundle_v2, plus course data and NumPy. Run predict.py with --checkpoint runs/model_bundle_v2 --method submission.py --split test, then the official scorer. runs/test-v2-score.json and reproduction-v2.json hold matching results and resources. Historical model_bundle and test-score.json reproduce v1 only. REPRODUCE.md gives exact commands; each stage has a separate pre-test source/asset hash manifest.':
    'Latest v3 reproduction uses v3_method.py with --checkpoint runs/model_bundle_v3, then the unchanged official scorer. Inference needs NumPy and the delivered helper source modules, not sklearn or SciPy. test-v3-score.json and reproduction-v3.json match its 380/467 result. Historical model_bundle_v2 and test-v2-score.json match v2 only; model_bundle and test-score.json match v1 only. REPRODUCE.md and active_submission.json identify the latest method and assets.',
    'Both stages are disclosed: the follow-up began after the v1 test was known, used only training/validation for fitting and selection, and was frozen before its one test evaluation.':
    'All three stages are disclosed. Follow-ups began after earlier test results were known; each fixed budget used train fitting and validation selection and was frozen before its one new test evaluation. V3 did not generalize its small validation improvement or achieve 83.5%.',
    "title('9. Fixed-budget follow-up and active v2')":"title('9. Fixed-budget follow-up and historical v2')",
    'Active assets:':'V2 historical assets:',
    'The root README and REPRODUCE.md point to v2; v1 records remain historical evidence.':
    'The current root README and REPRODUCE.md identify v3; v1/v2 records remain historical evidence.',
}
for old,new in changes.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)

extra='''
    page()
    title('10. V3 optimization: target not achieved')
    grid=load('v3/results.json')
    proto=load('v3/protocol.json')
    rv3=load('reproduction-v3.json')
    tv3=load('test-v3-score.json')
    av3=load('v3/analysis.json')
    p('After prior scores and public peer scores were known, the user requested 83.5%. V3_PLAN.md fixed 36 candidates before training, with promotion only above 743 validation-correct questions. The target was not a test-based stopping rule: no new search, fallback or weight adjustment followed the final evaluation. Models fit training only; validation selects. The arithmetic/coverage design was not derived from test errors.')
    p('Compact features concatenate the old 27 structure features, 16 answer/question features and 44 new local arithmetic/coverage features (87 total). Dense adds 1,024 raw embedding coordinates (1,111 total). A bounded AST evaluator permits only numeric literals and unary/binary +,-,*,/; it never executes candidate code. The equation matcher handles simple explicit numeric equalities, not arbitrary mathematical language. Error counts, relative residuals, word/number coverage and group-relative versions are predictive cues, not proof checks.')
    families=[('HGB compact',lambda n:n.startswith('hgb-compact')),('HGB dense',lambda n:n.startswith('hgb-dense')),
      ('ExtraTrees',lambda n:n.startswith('extra-')),('Fixed tree/v2 fusion',lambda n:n.startswith('fusion-')),
      ('Dense LR',lambda n:n.startswith('linear-')),('Dense listwise',lambda n:n.startswith('listwise-'))]
    rows=[]
    for label,match in families:
        subset=[e for e in grid['records'] if match(e['name'])]
        best=max(subset,key=lambda e:e['correct'])
        rows.append([label,len(subset),best['correct'],f'{best["accuracy"]:.2f}%',f'{best["hard_accuracy"]:.2f}%'])
    table(['Family (best validation)','N','Correct','All','Hard'],rows,[235,35,60,65,65])
    p('The budget contains 6 HGB, 2 ExtraTrees, 24 fixed fusions, 2 LR and 2 listwise models. Hyperparameters and train-only score normalization are in V3_PLAN.md. Portable NumPy outputs and choices were compared with the training-library outputs for all 8 trees and both LR models; listwise scores also matched direct affine computation. Ten behavior/gradient/information-boundary tests passed. A path-only plan lookup fix supports desktop and archive layouts.')
    lo,hi=av3['paired_bootstrap_95_pp']
    p(f'The converged dense listwise model, penalty 0.1 (41 iterations), won at 747/798 versus v2 743/798. V3-only/v2-only correct counts are {av3["v3_only_correct"]}/{av3["v2_only_correct"]}; paired validation gain is 0.50 points, descriptive 95% interval [{lo:.2f}, {hi:.2f}]. The interval includes zero and is not search-corrected. The grid took {grid["seconds"]:.2f} seconds wall time; complete records, warnings and fit times are saved. New features and changed representation/regularization are confounded; no causal arithmetic-feature claim is made.')
    p(f'V3 was frozen before its one final test: {tv3["correct"]}/{tv3["n"]} = {tv3["accuracy"]:.6f}%. Reaching 83.5% requires at least 390/467; v3 is 10 questions short and regresses four questions relative to historical v2. Thus the optimization target failed. This is evidence that a small validation gain can fail to generalize, not evidence that the added arithmetic cues improve test reasoning selection. All earlier checkpoints remain available; none is silently relabeled v3.')
    p(f'V3 assets occupy {rv3["asset_bytes"]:,} bytes. Complete fresh prediction plus scoring took {rv3["test_total_seconds"]:.3f} seconds locally, with single-thread library caps, sequential children and sampled OS-thread peaks of 3. The seven-question audit-format interface check passed. Reference-machine eligibility is not certified. freeze-v3.json identifies unchanged pre-test sources/assets; documentation additions are covered by the release manifest. No further test evaluation was run.')
'''
needle='    def footer(canvas,doc):'
assert source.count(needle)==1
source=source.replace(needle,extra+'\n'+needle)
exec(compile(source,'ten-page report with v3 failure disclosed','exec'),{'__name__':'__main__'})
