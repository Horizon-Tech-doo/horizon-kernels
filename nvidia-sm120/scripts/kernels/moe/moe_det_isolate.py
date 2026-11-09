# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Separate MoE alignment order from expert GEMM nondeterminism. No timing claims.

Uses one real layer, fixed inputs/weights/routing, and fresh/frozen/canonical
alignment. Snapshots both Marlin projection outputs before buffer reuse.
"""
import hashlib
import json
import sys

sys.argv = [sys.argv[0], "--layers", "3", "--no-timing"]
import moe_fse_test as t
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.experts import marlin_moe as mm


def fingerprint(x):
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def canonical_alignment(ids, block_size, experts):
    """Ascending flat pair indices within each expert, padded exactly like vLLM."""
    rows = ids.cpu().reshape(-1).tolist()
    size, blocks = t.route.align_sizes(len(rows), experts, block_size)
    sorted_ids, expert_ids = [], []
    for expert in range(experts):
        pairs = [index for index, assigned in enumerate(rows) if assigned == expert]
        padded = ((len(pairs) + block_size - 1) // block_size) * block_size
        sorted_ids.extend(pairs + [len(rows)] * (padded - len(pairs)))
        expert_ids.extend([expert] * (padded // block_size))
    used = len(sorted_ids)
    assert used <= size and len(expert_ids) <= blocks
    sorted_ids.extend([len(rows)] * (size - used))
    expert_ids.extend([-1] * (blocks - len(expert_ids)))
    return tuple(torch.tensor(v, device=ids.device, dtype=torch.int32)
                 for v in (sorted_ids, expert_ids, [used]))


def probe(layer, x, weights, ids, alignment, repeats=20):
    bs = t.hmoe._marlin_block_size(x.shape[0], t.TOPK + 1, t.E + 1)
    original = mm.ops.moe_wna16_marlin_gemm
    snapshots = []

    def traced(*args, **kwargs):
        out = original(*args, **kwargs)
        snapshots.append(out.clone())
        return out

    distinct = [set(), set(), set()]
    finite = True
    reuse_before = dict(t.hmoe._ALIGN_STATS)
    mm.ops.moe_wna16_marlin_gemm = traced
    try:
        for _ in range(repeats):
            snapshots.clear()
            t.hmoe._ALIGN[0] = None if alignment is None else (ids, bs, t.E + 1, alignment)
            out = t.moe(layer.m257, x, weights, ids)
            assert len(snapshots) == 2, "expected gate/up and down Marlin calls"
            for seen, value in zip(distinct, [*snapshots, out]):
                seen.add(fingerprint(value))
                finite &= bool(torch.isfinite(value).all())
    finally:
        mm.ops.moe_wna16_marlin_gemm = original
        t.hmoe._ALIGN[0] = None
    reused = t.hmoe._ALIGN_STATS["reused"] - reuse_before["reused"]
    if alignment is not None:
        assert reused == repeats, "frozen alignment must actually reach Marlin"
    return {"distinct_up": len(distinct[0]), "distinct_down": len(distinct[1]),
            "distinct_final": len(distinct[2]), "finite": finite,
            "alignment_reused": reused, "repeats": repeats}


def main():
    with set_current_vllm_config(VllmConfig()):
        layer = t.Layer(3)
        torch.manual_seed(5)
        rows = 0
        for m in (32, 128, 256):
            x = (torch.randn(m, t.H, device="cuda") * 0.5).to(torch.bfloat16)
            weights, ids = t.router_torch(x, F.linear(x, layer.gate264))
            bs = t.hmoe._marlin_block_size(m, t.TOPK + 1, t.E + 1)
            frozen = t.moe_align_block_size(ids, bs, t.E + 1, None, ignore_invalid_experts=True)
            canonical = canonical_alignment(ids, bs, t.E + 1)
            assert int(frozen[2]) == int(canonical[2])
            for name, alignment in (("fresh_stock", None), ("frozen_stock", frozen),
                                    ("canonical", canonical)):
                result = probe(layer, x, weights, ids, alignment)
                print(json.dumps({"check": "moe_determinism_isolation", "M": m,
                                  "alignment": name, **result}), flush=True)
                rows += 1
        assert rows == 9
        print("MOE_DET_ISOLATE_DONE", flush=True)


if __name__ == "__main__":
    main()
