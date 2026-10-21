"""Build full Release asset, data-light repository archive and versioned checkpoints."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent
package = root/'DASE7506-MP2-main' if (root/'DASE7506-MP2-main').is_dir() else root
out = root/'dist'
out.mkdir(exist_ok=True)
files = [p for p in package.rglob('*') if p.is_file()
         and not any(part in ('__pycache__','.venv','.git','dist') for part in p.relative_to(package).parts)
         and p.suffix not in ('.png','.pyc') and p.name != 'RELEASE_MANIFEST.json']
payload = {p.relative_to(package).as_posix():p.read_bytes() for p in files}
for name in ('GUIDE.md','PLAN.md','SUPPLEMENT_PLAN.md','V3_PLAN.md','RESULTS.md','package_release.py'):
    p=root/name
    if p.exists():
        value=p.read_bytes()
        if p.suffix=='.md':
            # Archive root is the canonical project directory, unlike the desktop wrapper.
            value=value.decode('utf-8').replace('DASE7506-MP2-main/','').encode('utf-8')
        payload[name]=value
active=json.loads((package/'code/runs/active_submission.json').read_text())
research = root/'optimize'
if research.is_dir():
    for p in research.iterdir():
        if p.is_file() and p.suffix in ('.py', '.txt', '.json'):
            payload['research/optimize/'+p.name] = p.read_bytes()
    cache = research/'cache/stack_data.npz'
    if cache.exists():
        payload['research/optimize/cache/stack_data.npz'] = cache.read_bytes()
    payload['research/README.md'] = (
        '# Historical research evidence\n\n'
        'Scripts and logs are exploratory history, not the final inference entry point. '
        'Some assume the original desktop directory layout. Large feature caches and '
        'experimental model bundles are omitted. The retained stack_data.npz contains '
        'saved grouped training OOF predictions. Recompute its comparison from code/ '
        'with python collect_selection_evidence.py; see its documented ensemble-proxy '
        'limitations. Final training uses code/train_linear_v2.py.\n'
    ).encode('utf-8')
def archive(name,items,description):
    hashes={n:hashlib.sha256(v).hexdigest() for n,v in items.items()}
    value=json.dumps({'description':description,'active':active,'files':hashes},indent=2).encode('utf-8')
    with zipfile.ZipFile(out/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,v in items.items():
            z.writestr('DASE7506-MP2/'+n,v)
        z.writestr('DASE7506-MP2/RELEASE_MANIFEST.json',value)
    return value
full_manifest=archive('MP2-complete.zip',payload,'Complete delivery including course data; large Release asset, not Git source')
(package/'RELEASE_MANIFEST.json').write_bytes(full_manifest)
light={n:v for n,v in payload.items() if not (n.startswith('code/data/') and n.endswith(('.jsonl','.npz')))
       and not (n.startswith('code/runs/v3/') and len(n.split('/'))>4)}
archive('MP2-source.zip',light,'Light source/evidence delivery; restore exact course data and verify retained manifest')
def keep_final(n):
    if n in ('README.md','SUBMISSION.md','REVIEW.md','.gitignore','GUIDE.md','RESULTS.md',
             'package_release.py','PACKAGE_MANIFEST.json'):
        return True
    if n in ('report/REPORT.md','report/REPORT.pdf') or n.startswith('research/'):
        return True
    if n.startswith('code/') and len(n.split('/')) == 2:
        return n.endswith(('.py','.txt','.md'))
    if n.startswith('code/runs_linear/'):
        return '/repro_bundle/' not in n
    if n.startswith('code/data/'):
        return n.split('/')[-1] in ('manifest.json','LICENSE.txt','NOTICE.txt')
    if n.startswith('code/runs/audit_inputs/'):
        return True
    # Retain historical numerical evidence cited in the final report, not old weights.
    if n.startswith(('code/runs/','code/runs_v2/')):
        return n.endswith('.json') and '/model_bundle' not in n and '/v3/' not in n
    return False
compact = {n:v for n,v in light.items() if keep_final(n)}
required = ['report/REPORT.pdf','code/submission.py','code/features_v2.py',
            'code/runs_linear/model_bundle/config.json','code/runs_linear/model_bundle/weights.npz',
            'code/runs_linear/review/integrity.json','research/optimize/cache/stack_data.npz']
assert all(n in compact for n in required), 'Final archive lacks required evidence/assets'
archive('MP2-submission.zip',compact,'Recommended final v4 repository delivery; restore course bank data separately')
for name,bundle in [('MP2-checkpoint.zip',Path(active['checkpoint']).name),
                    ('MP2-checkpoint-v2.zip','model_bundle_v2'),('MP2-checkpoint-v1.zip','model_bundle')]:
    base = package/'code'/active['checkpoint'] if name=='MP2-checkpoint.zip' else package/'code/runs'/bundle
    if not base.is_dir():
        continue
    with zipfile.ZipFile(out/name,'w',zipfile.ZIP_STORED) as z:
        for p in base.iterdir():
            z.write(p,bundle+'/'+p.name)
results = {}
for p in out.glob('*.zip'):
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        assert not any('.venv' in n or '__pycache__' in n or n.endswith('.png') for n in z.namelist())
        if 'DASE7506-MP2/RELEASE_MANIFEST.json' in z.namelist():
            manifest=json.loads(z.read('DASE7506-MP2/RELEASE_MANIFEST.json'))
            assert all(hashlib.sha256(z.read('DASE7506-MP2/'+n)).hexdigest()==h for n,h in manifest['files'].items())
        results[p.name] = {'bytes':p.stat().st_size,'files':len(z.namelist()),
                           'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(out/'SHA256.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
