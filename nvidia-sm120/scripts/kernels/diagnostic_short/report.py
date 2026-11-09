# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import json
D = Path(__import__("os").environ.get("HK_WORK", ".")) / 'diagnostic-short'
print('# Bounded GPU diagnostic\n')
for p in sorted(D.glob('control-*.log')) + sorted(D.glob('ncu-*.log')):
    print('## ' + p.stem + '\n')
    print('```\n' + '\n'.join(p.read_text(errors='replace').splitlines()[-18:]) + '\n```\n')
p = D / 'scale_split.json'
if p.exists():
    r = json.loads(p.read_text())
    print('| Scale intervention | NLL | First 8k | After 8k |\n|---|---:|---:|---:|')
    for mode,a in r['arms'].items():
        print(f"| {mode} | {a['nll']:.4f} | {a['first8k']:.4f} | {a['after8k']:.4f} |")
    print('\nExact repeat max absolute logprob difference:', r.get('repeat_max_abs_logprob','not finished'))
    print('\nSingle document, reference BF16 computation. Repeat agreement cannot exclude GPU-sharing defects.\nLayer activation statistics are sparse samples, not full distributions.')
else:
    print('Scale intervention results missing.')
