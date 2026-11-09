#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compare single-pass checkpoint states with exact split-prefix states."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def run(output):
    import torch
    from horizon_kernels import gdn_ckpt
    from horizon_kernels.gdn_prefill_state import patch_module
    from vllm.third_party.flash_linear_attention.ops import chunk
    patch_module(chunk, storage_dtype='float16')
    results = []
    for K, V, H, Hg, lengths, cuts in [
        (32, 32, 2, 1, [389], [[64, 320]]),
        (64, 64, 4, 2, [1000, 313, 711], [[832, 896], [256], [640]]),
        (128, 128, 4, 2, [1000, 313, 711], [[832, 896], [], [640]]),
        (256, 64, 2, 2, [389, 771], [[320], [64, 704]]),
        (128, 128, 48, 16, [1000], [[832, 896]]),
        (128, 128, 48, 16, [4313, 7107], [[832, 4224], [7040]]),
    ]:
        torch.manual_seed(K+sum(lengths))
        qsl = [0]
        for n in lengths: qsl.append(qsl[-1]+n)
        T = qsl[-1]
        q = torch.randn(1, T, Hg, K, device='cuda').bfloat16()
        k = torch.randn_like(q)
        v = torch.randn(1, T, H, V, device='cuda').bfloat16()
        g = (-torch.rand(1, T, H, device='cuda')*.02).float()
        beta = torch.rand(1, T, H, device='cuda').bfloat16()
        h0 = (torch.randn(len(lengths), H, V, K, device='cuda')*.01).half()
        tensors = (q,k,v,g,beta)
        def op(values, initial, cu):
            return chunk.chunk_gated_delta_rule(*values, initial_state=initial,
                output_final_state=True, cu_seqlens=torch.tensor(cu,device='cuda',dtype=torch.int32),
                use_qk_l2norm_in_kernel=True)
        reference, reference_final = op(tensors, h0, qsl)
        exports = [(i,j) for i,c in enumerate(cuts) for j in range(len(c))]
        slots = torch.arange(len(exports),device='cuda',dtype=torch.int32)*2+1
        cache = torch.full((len(exports)*2+1,H,V,K),.125,device='cuda',dtype=torch.float16)
        ck = SimpleNamespace(cuts=cuts, exports=exports, prefill_qsl_cpu=qsl,slots=slots)
        layer = SimpleNamespace(kv_cache=[None,cache])
        gdn_ckpt._CUR.update(ck=ck,layer=layer)
        try:
            actual, actual_final = op(tensors, h0, qsl)
            assert gdn_ckpt._CUR['states'] is True
            states = cache[slots]
            assert torch.all(cache[::2] == .125), 'export wrote an unreserved slot'
        finally:
            gdn_ckpt._CUR.clear()
        comparisons = {'output_exact': torch.equal(actual,reference),
                       'final_exact': torch.equal(actual_final,reference_final)}
        for slot,(seq,cut) in enumerate(exports):
            end = qsl[seq]+cuts[seq][cut]
            _, final = op(tuple(t[:,qsl[seq]:end].contiguous() for t in tensors),
                          h0[seq:seq+1], [0,end-qsl[seq]])
            comparisons[f'checkpoint_{slot}_exact'] = torch.equal(states[slot],final[0].half())
        row = dict(K=K,V=V,H=H,Hg=Hg,lengths=lengths,cuts=cuts,comparisons=comparisons)
        results.append(row); print(json.dumps(row),flush=True)
        output.write_text(json.dumps(results,indent=2)+'\n')
    from horizon_kernels.gdn_checkpoint_conv import export_conv
    for dim_first in (False,True):
        source = torch.randn(210,39,device='cuda').bfloat16()[:, :35]
        raw = torch.full((7,35,7) if dim_first else (7,7,35),-.125,device='cuda',dtype=torch.bfloat16)
        cache = raw if dim_first else raw.transpose(1,2)
        expected = cache.clone()
        slots = torch.tensor([4,1,6],device='cuda',dtype=torch.int32)
        rows = torch.tensor([64,131,199],device='cuda',dtype=torch.int64)
        indices = torch.tensor([r-3+j for r in (64,131,199) for j in range(3)],device='cuda')
        expected[slots,:,:3] = source[indices].view(3,3,35).transpose(1,2)
        export_conv(source,cache,slots,rows,3)
        row = dict(conv_dim_first=dim_first,comparisons=dict(conv_and_canaries_exact=torch.equal(cache,expected)))
        results.append(row); print(json.dumps(row),flush=True)
    output.write_text(json.dumps(results,indent=2)+'\n')
    assert all(all(r['comparisons'].values()) for r in results), 'checkpoint export differs'


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output',type=Path)
    run(p.parse_args().output)
