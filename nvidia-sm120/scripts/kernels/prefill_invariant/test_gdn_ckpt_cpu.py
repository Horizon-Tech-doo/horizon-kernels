#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""test_gdn_ckpt_cpu.py (Horizon Tech, 30-09-2026): CPU tests of horizon_kernels.gdn_ckpt's host logic against vLLM
0.30's own checkpoint helpers (get_mamba_prefill_checkpoint_position / is_mamba_prefill_checkpoint_valid), and of the
two-call split assembly with a stand-in chunk rule (a running sum, which is exactly split-invariant).
Run with the vLLM 0.30 venv python and the candidate plugin package on PYTHONPATH. No CUDA."""
import sys

import torch

from horizon_kernels import gdn_ckpt as gc

BS, HBS = 1600, 64
bad = 0


def check(name, cond, info=""):
    global bad
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {info}"))
    bad += not cond


# prefill row order
check("rows_no_spec", gc.prefill_rows(5, 2, None) == [2, 3, 4])
check("rows_spec", gc.prefill_rows(5, 0, [True, False, True, False, False]) == [1, 3, 4])

# cold 1k prompt: a single-block prompt has no checkpoint slot (vLLM: cdiv(seq_len, block) - 2 < 0) -> no checkpoint
off, cols, rows = gc.plan_checkpoints([0], [0, 1024], [1024], BS, HBS, True)
check("cold_1k_no_checkpoint", off == [0] and cols == [-1], (off, cols))
# cold 4k prompt: checkpoint at the EAGLE-shifted last hash boundary 4032 - 64 = 3968, block column 1
off, cols, rows = gc.plan_checkpoints([0], [0, 4096], [4096], BS, HBS, True)
check("cold_4k_checkpoint", off == [3968] and cols == [1] and rows == [3968], (off, cols, rows))
# without EAGLE drop: 4032
off, cols, rows = gc.plan_checkpoints([0], [0, 4096], [4096], BS, HBS, False)
check("cold_4k_no_eagle", off == [4032], off)
# a cached continuation (hit at 3520, 16229-token prompt, one step to the end): checkpoint 16192 - 64 = 16128
off, cols, rows = gc.plan_checkpoints([0], [0, 16229 - 3520], [16229], BS, HBS, True)
check("cached_tail_checkpoint", off == [16128 - 3520] and cols == [-(-16229 // BS) - 2], (off, cols))
# two prefill requests behind one decode row (absolute rows include the decode token)
off, cols, rows = gc.plan_checkpoints([1, 2], [0, 1, 4097, 5121], [900, 4096, 1024], BS, HBS, True)
check("mixed_batch", off == [3968, 0] and rows == [1 + 3968, -1], (off, rows))
# offset must be 64-aligned relative to the chunk start: a chunk starting at 100 (not 64-aligned) gets none
off, cols, rows = gc.plan_checkpoints([0], [0, 4096 - 100], [4096], BS, HBS, True)
check("unaligned_start_none", off == [0], off)

# split assembly: a stand-in chunk rule (running sum per sequence, with an initial state) is exactly split-invariant
def rule(x, cu, h0):
    o, f = torch.empty_like(x), []
    for i in range(len(cu) - 1):
        s = h0[i].clone()
        for t in range(cu[i], cu[i + 1]):
            s = s + x[t]
            o[t] = s
        f.append(s)
    return o, torch.stack(f)

x = torch.arange(20, dtype=torch.float64).unsqueeze(1)
qsl = [0, 12, 20]
offsets = [8, 0]
h0 = torch.tensor([[100.0], [1000.0]], dtype=torch.float64)
ref_o, ref_f = rule(x, qsl, h0)
def chain(x, qsl, cuts, h0):
    """the forward's chained rounds with the stand-in rule -> (outputs, finals, states at cuts in export order)"""
    o, final, at = torch.empty_like(x), [None] * len(cuts), {}
    for r, (idx, cu, seqs) in enumerate(gc.split_rounds(qsl, cuts)):
        hh = h0 if r == 0 else torch.stack([final[i] for i in seqs])
        o_r, f_r = rule(x[idx], cu, hh)
        o[idx] = o_r
        for j, i in enumerate(seqs):
            final[i] = f_r[j]
            if r < len(cuts[i]):
                at[(i, r)] = f_r[j]
    return o, torch.stack(final), at

o, final, at = chain(x, qsl, [[8], []], h0)
check("split_outputs_equal", torch.equal(o, ref_o))
check("split_final_equal", torch.equal(final, ref_f))
check("checkpoint_state", torch.equal(at[(0, 0)], h0[0] + x[:8].sum(0)), at)
rs = gc.split_rounds(qsl, [[8], []])
check("split_shapes", rs[0][0] == list(range(0, 8)) + list(range(12, 20)) and rs[1][0] == list(range(8, 12))
      and rs[1][2] == [0], rs)
# fill mode: several cuts per sequence (fill boundaries + checkpoint), mixed with an uncut and a one-cut sequence
x3 = torch.arange(40, dtype=torch.float64).unsqueeze(1) * 0.37
qsl3, cuts3 = [0, 20, 26, 40], [[4, 9, 15], [], [6]]
h3 = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
ro, rf = rule(x3, qsl3, h3)
o3, f3, at3 = chain(x3, qsl3, cuts3, h3)
check("multi_cut_outputs_equal", torch.equal(o3, ro))
check("multi_cut_final_equal", torch.equal(f3, rf))
# (the reference sums in another order than the running sum: compare with a tolerance; the chain itself is exact above)
check("multi_cut_states", all(torch.allclose(at3[(0, j)], h3[0] + x3[:c].sum(0)) for j, c in enumerate(cuts3[0]))
      and torch.allclose(at3[(2, 0)], h3[2] + x3[26:32].sum(0)), at3)
check("multi_cut_rounds", [r[2] for r in gc.split_rounds(qsl3, cuts3)] == [[0, 1, 2], [0, 2], [0], [0]])
# plan_fills: k11's continuation [8000, 14336), checkpoint column 7; stale scratch blocks at columns 5, 6 (ids 41, 42);
# column 4 (8000 = the step start) is not inside the step
row = [0, 0, 0, 0, 40, 41, 42, 43, 44, 45, 46, 47]
check("fill_k11", gc.plan_fills(row, 8000, 14336, 7, BS) == [(1600, 5), (3200, 6)], gc.plan_fills(row, 8000, 14336, 7, BS))
# fresh 4k (first allocation): column 0 is the null block -> nothing to fill
check("fill_fresh_4k", gc.plan_fills([0, 51, 52, 53, 54, 55], 0, 4096, 1, BS) == [])
# a block id that also sits in another column (a relocated scratch block) is never written
check("fill_dup_skipped", gc.plan_fills([0, 0, 0, 0, 40, 44, 42, 43, 44], 8000, 14336, 7, BS) == [(3200, 6)])
# a hit prefix (columns below the step start) is never written, even if non-null
check("fill_hit_prefix", gc.plan_fills([30, 31, 32, 0, 0, 60, 61], 4800, 9000, 4, BS) == [])

# HORIZON_TAIL_MIN_BLOCK: the tail stop is off only while splitting a prompt of at most one block
import types  # noqa: E402
from vllm.v1.core.sched import scheduler as S  # noqa: E402
seen = []
S.Scheduler._mamba_block_aligned_split = lambda self, req, n, *a, **k: seen.append(self.mamba_partial_cache_hit) or n
gc._log = lambda: types.SimpleNamespace(info=lambda *a, **k: None)
gc._install_tail_min_block()
fake = types.SimpleNamespace(mamba_partial_cache_hit=True, cache_config=types.SimpleNamespace(block_size=BS))
S.Scheduler._mamba_block_aligned_split(fake, types.SimpleNamespace(num_prompt_tokens=1024), 1024)
S.Scheduler._mamba_block_aligned_split(fake, types.SimpleNamespace(num_prompt_tokens=4096), 4096)
check("tail_min_block", seen == [False, True] and fake.mamba_partial_cache_hit is True, (seen, fake.mamba_partial_cache_hit))
# LCP stop: a chunk that would run past the last cacheable block boundary stops there (k11: [8000, 14336) -> 11200)
check("lcp_k11", gc.lcp_clamp(8000, 6336, 14336, BS, True) == 3200)
check("lcp_final_chunk", gc.lcp_clamp(11200, 3136, 14336, BS, True) == 3136)       # from lcp to the end: unchanged
check("lcp_short", gc.lcp_clamp(0, 1024, 1024, BS, True) == 1024)                  # lcp 0: unchanged
check("lcp_4k", gc.lcp_clamp(0, 4096, 4096, BS, True) == 1600)                     # 4k: [0,1600) then [1600,4096)
import os  # noqa: E402
os.environ["HORIZON_GDN_CKPT_STOP"] = "bogus"
try:
    gc.stop_mode()
    check("stop_mode_rejects", False)
except ValueError:
    check("stop_mode_rejects", True)
for mname in ("lcp", "cont", "fill"):
    os.environ["HORIZON_GDN_CKPT_STOP"] = mname
    check(f"stop_mode_{mname}", gc.stop_mode() == mname)
del os.environ["HORIZON_GDN_CKPT_STOP"]
check("stop_mode_default", gc.stop_mode() == "lcp")
# cont: the wrapped split applies the LCP stop only to a continuation chunk (request.num_computed_tokens > 0)
S.Scheduler._mamba_block_aligned_split = lambda self, req, n, a=0, b=0: n
gc._install_lcp_stop()
sch = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=BS), use_eagle_block_drop=True)
mk = lambda nct: types.SimpleNamespace(num_computed_tokens=nct, num_prompt_tokens=14336, num_tokens=14336)
for mname, fresh4k, cont in (("lcp", 1600, 3200), ("cont", 4096, 3200)):
    gc._MODE = mname
    r4 = S.Scheduler._mamba_block_aligned_split(sch, types.SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=4096,
                                                                           num_tokens=4096), 4096)
    rc = S.Scheduler._mamba_block_aligned_split(sch, mk(8000), 6336)
    check(f"lcp_wrapper_{mname}", (r4, rc) == (fresh4k, cont), (r4, rc))
gc._MODE = "lcp"
check("lcp_no_eagle", gc.lcp_clamp(8000, 6336, 14336, BS, False) == 4800)         # lcp 12800 without the drop
print(f"{'OK' if not bad else 'FAILED'} ({bad} failed)")
sys.exit(1 if bad else 0)
