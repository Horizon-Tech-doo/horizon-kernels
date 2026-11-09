# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Persistent norm exactness, FP64 reference, determinism and graph replay.

Run only under the exclusive GPU lock. Arguments: candidate tasks.py,
portable standalone add_rmsnorm.py, compiler harness (wrapper definition).
The wrapper runs four consecutive row tasks sharing the same reduction scratch.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import json
import torch
import tilelang
import tilelang.language as T


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tk = load('candidate_tasks', sys.argv[1])
standalone = load('standalone_norm', sys.argv[2])
# Use precisely the compiled wrapper without running CPU-only compilation setup.
node = next(n for n in ast.parse(Path(sys.argv[3]).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
ns = dict(tk=tk, tilelang=tilelang, T=T)
exec(compile(ast.Module(body=[node], type_ignores=[]), sys.argv[3], 'exec'), ns)
kernel = ns['kernel']
torch.manual_seed(2909)
count = 0
for h in (256, 512, 768, 4096, 5120, 10240):
    for residual in (False, True):
        for centered in (False, True):
            x = torch.randn((4, h), device='cuda', dtype=torch.bfloat16)
            r = torch.randn_like(x)
            w = torch.randn((h,), device='cuda', dtype=torch.bfloat16)
            y, ro, ref, rr = [torch.empty_like(x) for _ in range(4)]
            k = kernel(h, residual, centered)
            vt, _ = tk.norm_geometry(h)
            refk = standalone.get_kernel(h, 1e-6, residual, vt, centered)
            refk(x, r, w, ref, rr)
            k(x, r, w, y, ro)
            torch.cuda.synchronize()
            assert torch.equal(y, ref), (h, residual, centered, 'standalone mismatch')
            if residual:
                assert torch.equal(ro, rr), (h, 'residual mismatch')
            s = x.double() + r.double() if residual else x.double()
            expected = s * torch.rsqrt(s.square().mean(-1, keepdim=True) + 1e-6)
            expected *= w.double() + (1 if centered else 0)
            err = ((y.double() - expected).norm() / expected.norm().clamp_min(1e-30)).item()
            assert torch.isfinite(y).all() and err <= 0.0025, (h, err)
            for _ in range(3):
                k(x, r, w, y, ro)
                assert torch.equal(y, ref)
            # Retry transient gpushare capture failures once; other gates stay strict.
            for attempt in range(2):
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                        k(x, r, w, y, ro)
                    break
                except RuntimeError:
                    if attempt == 1:
                        raise
                    torch.cuda.synchronize()
            for _ in range(2):
                y.fill_(float('nan')); ro.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(y, ref)
                if residual:
                    assert torch.equal(ro, rr)
            # Weight convention control: zero weights zero standard norm only.
            w.zero_()
            k(x, r, w, y, ro)
            assert torch.isfinite(y).all()
            if not centered:
                assert torch.count_nonzero(y).item() == 0
            else:
                assert torch.count_nonzero(y).item() > 0
            count += 1
            print(json.dumps(dict(h=h, residual=residual, centered=centered,
                                  relative_l2=err, exact=True, graph=True)), flush=True)
print(f'PERSISTENT_NORM_GPU_PASS {count}', flush=True)
