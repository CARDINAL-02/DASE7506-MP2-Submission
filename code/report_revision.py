"""Documentation-only post-freeze erratum; frozen Python sources remain byte-identical."""
from pathlib import Path

source = Path('make_report.py').read_text(encoding='utf-8')
old = 'Learned scoring masks invalid parsed answers before argmax, and also returns 0 for an all-invalid group.'
new = ('Learned scoring masks invalid parsed answers before argmax when any valid answer exists. '
       'For an all-invalid group it returns the highest-scoring candidate (earliest on an exact tie). '
       'This legal-index fallback cannot earn a point under the official parser.')
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, 'make_report.py (documented report erratum)', 'exec'), {'__name__':'__main__'})
