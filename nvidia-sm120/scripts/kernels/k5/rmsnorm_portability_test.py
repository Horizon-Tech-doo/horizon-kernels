# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compile or GPU-test standard/Gemma RMSNorm on several hidden dimensions.

--lower uses no CUDA module loading. --gpu must run under the exclusive GPU job.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
mode = p.add_mutually_exclusive_group(required=True)
mode.add_argument('--lower', action='store_true')
mode.add_argument('--gpu', action='store_true')
a = p.parse_args()
if a.lower:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TVM_FFI_DISABLE_TORCH_C_DLPACK'] = '1'
import torch
if a.lower:
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
spec = importlib.util.spec_from_file_location('portable_norm', a.source)
norm = importlib.util.module_from_spec(spec); spec.loader.exec_module(norm)
a.output.mkdir(parents=True, exist_ok=True)
shapes = (256, 768, 4096, 5120)
if a.lower:
    import tilelang
    from tvm.target import Target
    from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
    from tilelang.contrib.nvcc import get_nvcc_compiler
    target = Target({'kind': 'cuda', 'arch': 'sm_120a'})
    for h in shapes:
        for centered in (False, True):
            for residual in (False, True):
                name = f'norm_h{h}_centered{int(centered)}_residual{int(residual)}'
                jf = norm.add_rmsnorm_kernel
                tir = jf.get_tir(h, 1e-6, residual, norm._threads(h), centered)
                with target:
                    with tilelang.transform.PassContext(config=jf.pass_configs or {}):
                        artifact = tilelang.lower(tir, target=target)
                cu = a.output / (name + '.cu'); cu.write_text(artifact.kernel_source)
                result = subprocess.run([get_nvcc_compiler(), '--cubin', '-O3', '-arch=sm_120a',
                                         '-std=c++20', '-w', '-I'+TILELANG_TEMPLATE_PATH,
                                         '-I'+CUTLASS_INCLUDE_DIR, '-Xptxas', '-v',
                                         '-o', str(cu.with_suffix('.cubin')), str(cu)],
                                        capture_output=True, text=True)
                (a.output / (name + '.log')).write_text(result.stdout + result.stderr)
                result.check_returncode()
                print(json.dumps({'compiled': name, 'threads': norm._threads(h)}), flush=True)
    print('RMSNORM_PORTABILITY_COMPILE_DONE', flush=True)
else:
    torch.manual_seed(29)
    with (a.output / 'gpu.jsonl').open('w') as out:
        for h in shapes:
            for m in (1, 17, 32):
                x = torch.randn(m, h, device='cuda', dtype=torch.bfloat16)
                r = torch.randn_like(x)
                w = torch.randn(h, device='cuda', dtype=torch.bfloat16)
                for centered in (False, True):
                    for residual in (False, True):
                        def run():
                            if residual:
                                return norm.add_rmsnorm(x, r, w, 1e-6, zero_centered_gamma=centered)
                            return (norm.rmsnorm(x, w, 1e-6, zero_centered_gamma=centered),)
                        total = x.float() + r.float() if residual else x.float()
                        gamma = w.float() + 1 if centered else w.float()
                        ref = (total * torch.rsqrt(total.square().mean(-1, keepdim=True)+1e-6) * gamma).bfloat16()
                        expected = run()
                        err = float((expected[0].float()-ref.float()).norm()/ref.float().norm())
                        assert err <= .0025, (h, m, centered, residual, err)
                        if residual:
                            assert torch.equal(expected[1], total.bfloat16())
                        for _ in range(4):
                            assert all(torch.equal(v, e) for v, e in zip(run(), expected))
                        for attempt in range(3):
                            try:
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                                    captured = run()
                                break
                            except RuntimeError:
                                torch.cuda.synchronize()
                                if attempt == 2:
                                    raise
                        for _ in range(2):
                            graph.replay(); torch.cuda.synchronize()
                            assert all(torch.equal(v, e) for v, e in zip(captured, expected))
                        row = dict(H=h, M=m, zero_centered_gamma=centered, residual=residual,
                                   relative_error=err, deterministic=True, graph_replays=2)
                        line = json.dumps(row); print(line, flush=True); out.write(line+'\n'); out.flush()
        # Zero weights distinguish the semantics without a tolerance comparison.
        x = torch.ones(1, 256, device='cuda', dtype=torch.bfloat16)
        w = torch.zeros(256, device='cuda', dtype=torch.bfloat16)
        assert bool((norm.rmsnorm(x, w, 1e-6, zero_centered_gamma=False) == 0).all())
        assert bool((norm.rmsnorm(x, w, 1e-6, zero_centered_gamma=True) != 0).all())
        out.write(json.dumps({'complete': True, 'cases': 48, 'zero_weight_semantics': True})+'\n')
    print('RMSNORM_PORTABILITY_GPU_DONE', flush=True)
