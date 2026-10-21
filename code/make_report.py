"""Render a six-page report from saved experiment evidence; never trains or selects."""
import json
from pathlib import Path
from html import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader


def main():
    r = Path('runs')
    load = lambda name: json.loads((r/name).read_text(encoding='utf-8'))
    experiments, selection, analysis, repro = [load(n) for n in
        ('experiments.json','selection.json','analysis.json','reproduction.json')]
    test, val = load('test-score.json'), load('validation-score.json')
    winner = selection['winner']
    out = Path('../report')
    out.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Text', fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallText', fontName='Helvetica',fontSize=8,leading=11,spaceAfter=5))
    styles['Heading1'].textColor = colors.HexColor('#153b56')
    story, md = [], []
    def title(text):
        story.append(Paragraph(escape(text),styles['Heading1']))
        md.append('# '+text+'\n')
    def p(text):
        story.append(Paragraph(escape(text),styles['Text']))
        md.append(text+'\n')
    def table(headers, rows, widths):
        data = [[Paragraph(escape(str(v)),styles['SmallText']) for v in row] for row in [headers,*rows]]
        t = Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff5')),
          ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),
          ('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,colors.grey),
          ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dddddd'))]))
        story.append(t)
        md.append('| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|')
        md.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
        md.append('')
    def page():
        story.append(PageBreak())
    title('Selecting Reliable LLM Reasoning')
    p('DASE7506 MP2 | Individual coursework | Fall 2026')
    p('Research question: can a small learned verifier improve on final-answer majority voting, and does combining supplied trace embeddings with candidate-level consistency and surface features help?')
    p('Hypothesis: agreement is a useful prior but is not sufficient when multiple candidates share the same wrong answer. Frozen trace embeddings and inexpensive structural features may provide complementary evidence. The experiment tests predictive selection, not formal reasoning validity.')
    p(f'The frozen method is {winner["name"]}. It selects one supplied candidate per question and runs entirely offline on CPU. Its official validation accuracy is {val["accuracy"]:.4f}% ({val["correct"]}/{val["n"]}); the one final official test score is {test["accuracy"]:.4f}% ({test["correct"]}/{test["n"]}).')
    p(f'The frozen asset files occupy {repro["asset_bytes"]:,} bytes uncompressed. Fresh prediction plus independent scoring took {repro["test_total_seconds"]:.3f} seconds locally with four-thread library limits. This is local evidence, not certification on the reference Xeon Platinum 8457C.')
    title('1. Data and evaluation boundary')
    p('The supplied, disjoint banks contain 4,755 training, 798 validation and 467 test questions, each with four candidate traces and 1,024-dimensional embeddings. Training references alone define fitting labels; validation chooses the method. A label is 1 exactly when the official parser returns a non-None answer equal to the reference parsed answer. Multiple positives per question are allowed.')
    p('The fixed official parser, scorer and inputs were preserved. Training-only preprocessing statistics and weights are saved. IDs are used only for alignment. No external datasets, pretrained weights, generated solutions, cached answers or test-dependent tuning are used. Official data integrity checking examines reference alignment and hashes without choosing a method.')
    p('Earlier guide/chat material had exposed unverified test baseline statistics. This exposure is disclosed; those numbers were removed and did not define model selection, grid expansion or a stopping rule. All choices below are made from saved validation results. This limits claims of completely blind development.')
    page()
    title('2. Features, models and selection')
    p('Baselines: fixed candidate 0, and majority vote over valid canonical final answers. Equivalent fractions/decimals count together. Ties select the earliest candidate. All-invalid inputs return 0. Learned scoring masks invalid parsed answers before argmax, and also returns 0 for an all-invalid group.')
    p('Embedding mode uses 1,024 supplied coordinates. Structure mode uses 27 engineered features: validity; same-answer count; number of distinct valid answers; largest answer group; valid fraction; log trace length; relative length and standardized within-group length; log counts of numbers, lines and equal signs; final marker flags; log question length; log absolute answer magnitude, sign, integer and zero flags; mean/max cosine similarity to other candidates; mean similarity to same-answer peers; and six lexical flags (therefore, however, wait, check, incorrect, approximately).')
    p('Structure mode still uses embeddings to compute cosine summaries: it is an ablation of direct embedding coordinates, not an embedding-free method. Combined mode concatenates raw and group-centered embeddings with the 27 features (2,075 dimensions). All columns are standardized using training statistics only.')
    p('For linear scoring, adding group-centered embeddings does not add expressiveness over raw embeddings: the subtracted group mean is shared and cancels in within-question ranking. It can change the regularization parameterization. No independent claim is made that centering improves reasoning.')
    p('Each candidate is fitted with binary cross-entropy through scikit-learn logistic regression (L2, lbfgs, at most 600 iterations). Each mode uses C = 0.01, 0.1, 1 and 10. The additional model is a 32-unit ReLU MLP on combined features, alpha=0.01, batch size 256, at most 100 epochs, no early stopping, seeds 17, 29 and 43. No extra search was added after inspecting results.')
    p('The same frozen NumPy inference interface loads float64 arrays, evaluates affine/ReLU layers and ranks final logits. Sigmoid probabilities are unnecessary for ranking; numerical saturation can create false ties. Export decisions were checked against trained-model logit decisions on validation. A synthetic checkpoint regression test protects this behavior.')
    p('Selection was specified before final testing: maximize whole-validation selection accuracy, then hard-subset accuracy, then minimize learned-array bytes, then prefer the earlier run. The hard subset requires at least two distinct non-None parsed answers and contains 355 validation questions. This matches an answer-diversity condition, not an established difficulty distribution.')
    page()
    title('3. Validation experiments and ablations')
    rows = [[e['name'], f'{e["accuracy"]:.2f}',f'{e["hard_accuracy"]:.2f}',f'{e["train_seconds"]:.2f}'] for e in experiments]
    table(['Configuration','All (%)','Hard (%)','Fit (s)'],rows,[265,65,65,65])
    p('All results use the same 798 validation questions; the hard column uses the same 355-question subset. Fit time excludes loading, feature extraction and evaluation. Each mode is independently refitted. For a controlled ablation compare embedding, structure and combined rows at the same C=0.01; selecting each mode independently is a separate tuned-model comparison.')
    best = lambda mode: max((e for e in experiments if e.get('features')==mode and e.get('kind')=='lr'),key=lambda x:x['accuracy'])
    p(f'Best linear accuracies: embedding {best("embedding")["accuracy"]:.2f}%, structure {best("structure")["accuracy"]:.2f}%, combined {best("combined")["accuracy"]:.2f}%. These comparisons assess complementary predictive evidence; they do not isolate semantic reasoning from formatting shortcuts.')
    p(f'MLP validation mean across three seeds: {analysis["mlp_seed_accuracy_mean"]:.2f}%, population SD {analysis["mlp_seed_accuracy_std"]:.2f} percentage points. Seed variability is training variability, not a confidence interval for generalization. Choosing the strongest configuration on this validation set can introduce selection optimism.')
    page()
    title('4. Paired evidence and error analysis')
    lo, hi = analysis['paired_bootstrap_95']
    p(f'Against majority voting, the selected model is uniquely correct on {analysis["winner_only_correct"]} validation questions, while majority is uniquely correct on {analysis["majority_only_correct"]}. Both fail on {analysis["both_wrong"]}. The paired gain is {analysis["paired_delta_pp"]:.2f} percentage points. A question-level paired bootstrap (10,000 resamples, seed 20261021) gives a descriptive 95% interval [{lo:.2f}, {hi:.2f}] percentage points.')
    p('This interval compares paired outcomes on the observed validation bank; it is not corrected for hyperparameter/model selection and is not proof of generalization. Whole-question resampling preserves the four-candidate dependency. A confidence interval for one accuracy alone would not test a difference between methods.')
    for kind in ('improvement','regression','both_wrong'):
        ex = next((e for e in analysis['examples'] if e['kind']==kind),None)
        if ex:
            p(f'Example ({kind}), validation ID {ex["id"]}: {ex["question"]}')
            p(f'Canonical candidate answers: {ex["answers"]}. Model index: {ex["selected"]}; majority index: {ex["majority"]}; correct indices: {ex["correct_indices"]}.')
    p('The examples illustrate successful departures from majority, harmful departures and unresolved errors. Answer agreement can preserve a shared mistake; surface regularity can correlate with correctness without verifying arithmetic. These are outcome categories, not a complete manual taxonomy of mathematical errors. The report does not claim every selected trace has valid intermediate reasoning.')
    p('Feature ablations and errors support a narrow conclusion: combining evidence improves candidate selection on validation. A full-text versus final-answer-only controlled representation experiment was not performed, so no claim is made that the gain is caused specifically by reading reasoning. Cosine summaries and text formatting remain possible shortcuts.')
    page()
    title('5. Frozen test and resource accounting')
    table(['Item','Observed value'],[
      ['Selected configuration',winner['name']],
      ['Official validation',f'{val["accuracy"]:.6f}% ({val["correct"]}/{val["n"]})'],
      ['Official final test',f'{test["accuracy"]:.6f}% ({test["correct"]}/{test["n"]})'],
      ['Fresh prediction + scoring',f'{repro["test_total_seconds"]:.3f} seconds locally'],
      ['Uncompressed inference assets',f'{repro["asset_bytes"]:,} bytes / 16,777,216-byte limit'],
      ['Training script wall time',f'{selection["total_seconds"]:.2f} seconds (final successful run)'],
      ['Sum of final-grid fitting times',f'{sum(e["train_seconds"] for e in experiments):.2f} seconds'],
      ['Local CPU','Intel Core i7-10870H @ 2.20 GHz; CPU only'],
      ['Runtime',f'Python 3.12.14, NumPy {repro["environment"]["numpy"]}; Windows'],
      ['Threads','OMP/OpenBLAS/MKL/NumExpr each capped at 4; one process at a time'],
    ],[210,250])
    p('Before the final test, finalize.py wrote runs/freeze.json with source hashes, selected configuration and full asset hashes. The same asset bundle then produced the official validation and final test files. Test references were accessed for scoring by the unchanged official scorer after freezing. No test-based model change follows.')
    p('The uncompressed NPZ is a ZIP_STORED container; full on-disk bytes plus configuration bytes are counted, including container overhead. The training experiment table records learned array payload bytes plus configuration, while the freeze/reproduction records conservatively count complete files. All training-only checkpoints remain experiment evidence and are not loaded by final inference.')
    p('Local timing includes two fresh subprocess launches, imports, input hash checks, array loading, feature computation, selection and scoring. It excludes training. The reference-machine time remains unverified. Hardware probing of physical RAM was unavailable in the sandbox, so RAM capacity is not claimed. All computation used CPU; no GPU was used.')
    p('Development also included one interrupted grid for debugging saturated neural probabilities and a failed environment-bootstrap attempt. The final successful grid timings are exact saved measurements; earlier development elapsed time was not instrumented and is not included in those figures. Search cost must not be interpreted as total human/development time.')
    page()
    title('6. Reproduction, limitations and disclosure')
    p('Inference reproduction requires the frozen source files and runs/model_bundle, plus the course data and NumPy. Run predict.py with --method submission.py --checkpoint runs/model_bundle --split test, then score_predictions.py on its output. REPRODUCE.md supplies exact installation, training and evaluation commands. runs/test-score.json contains the official score; runs/reproduction.json contains runtime and asset records. Hashes identify the exact matching source and assets.')
    p('The interface was tested using a seven-question audit-format fixture assembled only from training inputs with replacement IDs. No reference file exists in that fixture. This verifies variable bank size and ID independence; it does not measure unseen-distribution accuracy. Unit tests cover fraction/decimal equivalence, invalid-majority handling, ties, all-invalid/empty input and logit saturation.')
    p('Limitations: validation-based selection has finite-sample and search optimism; test has only 467 questions; no independent audit-bank score is available; the public course bank may differ from future questions. Candidate-level training does not directly optimize the grouped selection objective. The method cannot validate a mathematical proof and may exploit lexical or formatting signals. A stronger model was not automatically preferred over a cheaper model: the saved validation criterion determines selection.')
    p('AI disclosure: substantive OpenAI Codex assistance was used to interpret requirements, design and implement features, training and inference, debug numerical behavior, execute experiments, analyze outcomes and draft this report. The student must review, understand and take responsibility for the implementation and submitted claims. No student identity or personal contribution has been fabricated.')
    p('Reuse: the course supplies the task, candidate bank, fixed embeddings and official utilities. The implementation uses NumPy and scikit-learn with pinned dependencies. Questions/references derive from GSM8K under its retained MIT license; candidate traces/embeddings were prepared with Qwen3-0.6B as documented in the course manifest. Preserve data/LICENSE.txt and data/NOTICE.txt.')
    p('Sources: local guide/GUIDE.md; code/README.md; code/common.py; code/data/manifest.json. Experimental evidence: runs/experiments.json, selection.json, analysis.json, freeze.json, reproduction.json and official score JSONs. No new external empirical claim is introduced.')
    p('Submission status: local reproducible materials are prepared. Public repository/checkpoint publication, student-ID entry and actual GitHub issue submission still require the account holder. The deadline in the supplied guide is 31 October 2026, 23:59 Beijing time (UTC+8).')
    p('Conclusion: inexpensive learned verification can improve over answer agreement on the supplied validation bank. The frozen test result is reported without post-test tuning. The evidence supports final-answer selection accuracy, not a claim of sound reasoning verification.')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(48,28,'DASE7506 MP2 | Reproducible candidate selection')
        canvas.drawRightString(A4[0]-48,28,str(doc.page))
    path = out/'REPORT.pdf'
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
    (out/'REPORT.md').write_text('\n'.join(md),encoding='utf-8')
    pages = len(PdfReader(path).pages)
    if pages > 10:
        raise RuntimeError(f'Report exceeds limit: {pages}')
    print(json.dumps({'pdf':str(path.resolve()),'pages':pages}))


if __name__ == '__main__':
    main()
