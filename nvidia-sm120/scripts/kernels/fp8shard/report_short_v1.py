# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only report; list incomplete arms explicitly and compare matched tokens."""
import json
import math
from pathlib import Path

root = Path.home() / 'logprob_eval'
doc = 'prose_chat_16384'
arms = {}
print('# Short scale diagnostic, 27-09-2026\n')
print('Same 16,384 input tokens; lower NLL is better. No NVIDIA/MoE or clipping test.\n')
print('| Arm | Whole NLL | First 8k NLL | After 8k NLL |')
print('|---|---:|---:|---:|')
for arm in ('vllm-served', 'ref-served', 'ref-exact', 'vllm-exact'):
    path = root / f'inv-short-v1-{arm}.json'
    if not path.exists():
        print(f'| {arm} | MISSING | | |')
        continue
    lp = json.loads(path.read_text())[doc]['lp']
    if len(lp) != 16383 or not all(math.isfinite(v) for v in lp):
        raise ValueError(f'{arm}: invalid scores')
    arms[arm] = lp
    nll = lambda xs: -sum(xs) / len(xs)
    print(f'| {arm} | {nll(lp):.4f} | {nll(lp[:7999]):.4f} | {nll(lp[7999:]):.4f} |')
print('\nPaired changes (exact minus shared; positive means worse):')
for runtime in ('ref', 'vllm'):
    a, b = arms.get(runtime + '-served'), arms.get(runtime + '-exact')
    if a is not None and b is not None:
        delta = sum(x - y for x, y in zip(a, b)) / len(a)
        print(f'- {runtime}: NLL delta {delta:.4f}; perplexity ratio {math.exp(delta):.3f}x')
for scale in ('served', 'exact'):
    a, b = arms.get('ref-' + scale), arms.get('vllm-' + scale)
    if a is not None and b is not None:
        print(f'- {scale}: reference/vLLM mean absolute token logprob difference {sum(abs(x-y) for x,y in zip(a,b))/len(a):.4f}')
print('\nOne document only. A runtime discrepancy is a lead, not proof of a vLLM bug; the reference uses BF16 computation and different kernels.')
if len(arms) != 4:
    raise SystemExit(1)
