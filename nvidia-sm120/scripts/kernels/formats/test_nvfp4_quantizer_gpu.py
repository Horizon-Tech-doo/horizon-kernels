# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Load-time CUDA quantizer gate; not a GEMM speed or model-quality test."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch
from safetensors import safe_open

candidate, baseline, weights, output = map(Path, sys.argv[1:5])
def load(path):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if
             isinstance(n, ast.FunctionDef) and n.name == 'quantize_nvfp4_rtn' or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_E2M1_POS' for t in n.targets)]
    ns = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns['quantize_nvfp4_rtn']
new, old = load(candidate), load(baseline)

def equal(a, b):
    return all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b))

def check(w, name, compare_old):
    a = new(w)
    assert equal(a, new(w)), (name, 'nondeterministic quantization')
    packed, scales, alpha = a
    sf = scales.float() * alpha[:, None]
    assert torch.isfinite(sf).all() and (sf >= torch.finfo(torch.float32).tiny).all(), name
    codes = packed.view(torch.uint8)
    codes = torch.stack((codes & 15, codes >> 4), -1).reshape_as(w)
    # A zero input must encode zero, including inside otherwise nonzero blocks.
    assert (codes[w == 0] == 0).all(), name
    if compare_old:
        assert equal(a, old(w)), (name, 'ordinary-weight quantization changed')
    torch.cuda.synchronize()
    return {'case': name, 'shape': list(w.shape), 'finite_normal_denominators': True,
            'zero_codes_exact': True, 'deterministic': True,
            'old_comparison_performed': compare_old,
            'old_bytes_equal': True if compare_old else None}

rows = []
with output.open('x') as f:
    def emit(row):
        rows.append(row)
        line = json.dumps(row)
        print(line, flush=True); f.write(line + '\n'); f.flush()
    for magnitude in (0., 1e-40, 1e-35, 1e-6, .03, 100.):
        w = torch.full((128, 256), magnitude, device='cuda', dtype=torch.bfloat16)
        w[0] = 0
        w[1, ::2] = 0
        w[2] *= -1
        emit(check(w, f'synthetic-{magnitude}', magnitude >= 1e-6))
        del w
    with safe_open(weights, framework='pt', device='cpu') as source:
        names = sorted(k for k in source.keys() if k.endswith('.weight') and len(source.get_slice(k).get_shape()) == 2)
        real = 0
        for name in names:
            w = source.get_tensor(name)
            if w.dtype != torch.bfloat16 or w.shape[1] % 16:
                continue
            w = w.cuda()
            emit(check(w, name, True)); real += 1
            del w
    assert real > 0, 'No real BF16 draft weights tested'
    emit({'complete': True, 'cases': len(rows), 'real_weights': real,
          'candidate_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
          'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest()})
