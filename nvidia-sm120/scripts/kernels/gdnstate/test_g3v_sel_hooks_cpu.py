# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""G3v-sel variant (30-09-2026 evening) of test_g3v_hooks_cpu.py: the same real-class checks with the one-launch
expand ON (ctx.sel): a fake gdn_g3v.expand_sel is a Python model of g3v_expand_sel_kernel's selection (MODE 0 / 1 /
2, window gather, ridx write); rows, window slots, modes and the H4 skip are checked against the same replica of
vLLM 0.30's own align decisions. The per-group fallback path is covered by test_g3v_hooks_cpu.py itself.
Real-class hook check of horizon_kernels.gdn_g3 against the installed vLLM 0.30 (30-09-2026). CPU only: run on the
server with CUDA hidden (the vLLM 0.30 modules import without a GPU):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=<dir containing horizon_kernels> $HK_VENV/bin/python this.py
1. the four MambaHybridModelState methods and the Qwen GDN decode method exist with the signatures the hooks mirror
2. install() wraps them; with G3v not integrated the layer hook falls through to vLLM's own method
3. H1/H2 row selection == a Python replica of vLLM 0.30's own decisions (preprocess_mamba_align_fused_kernel:
   crossing = state_idx >= 0 and new != old; postprocess_mamba_fused_kernel: aligned >= running) on random batches,
   incl. -1 rows (PP), non-pure batches (every row), window slot gathering past the block-table end (-> 0)
4. H3 resets exactly the request-state index's meta; H4 fills the request-index buffer (-1 during capture)
The originals are replaced by recorders BEFORE install (they launch Triton kernels); a fake tilelang gdn_g3v records
the expand calls. No kernel runs. Exit 0 = all pass.
"""
import inspect
import os
import sys
import types

import numpy as np
import torch

os.environ["HORIZON_GDN_G3"] = "1"
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState as M  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q  # noqa: E402

EXPECT = {
    "add_request": ["self", "req_index", "new_req_data"],
    "preprocess_state": ["self", "input_batch", "block_tables", "kv_cache_config", "num_computed_tokens"],
    "prepare_attn": ["self", "input_batch", "cudagraph_mode", "block_tables", "slot_mappings", "attn_groups",
                     "kv_cache_config", "for_capture", "ubatch_idx"],
    "postprocess_state": ["self", "idx_mapping", "num_sampled", "num_computed_tokens"],
}
for name, params in EXPECT.items():
    got = list(inspect.signature(getattr(M, name)).parameters)
    check(f"vLLM 0.30 MambaHybridModelState.{name} signature", got == params, str(got))
got = list(inspect.signature(Q.QwenGatedDeltaNetAttention._forward_core_decode_spec_post_conv_fused_norm).parameters)
check("vLLM 0.30 Qwen GDN _forward_core_decode_spec_post_conv_fused_norm signature",
      got == ["self", "mixed_qkv", "b", "a", "output_gate", "core_attn_out", "attn_metadata"], str(got))
for a in ("_align_mode", "_mamba_state_idx_gpu", "_get_mamba_group_info"):
    src = inspect.getsource(M)
    check(f"MambaHybridModelState uses {a}", a in src)

# recorders for the originals (they would launch GPU kernels)
CALLS = []
M.add_request = lambda self, i, d: CALLS.append(("add", i))
M.preprocess_state = lambda self, ib, bt, kc, nc: CALLS.append(("pre",))
M.prepare_attn = lambda self, ib, cm, bt, sm, ag, kc, for_capture=False, ubatch_idx=0: CALLS.append(("prep", for_capture))
M.postprocess_state = lambda self, im, ns, nc=None: CALLS.append(("post",))
ORIG_LAYER = []
Q.QwenGatedDeltaNetAttention._forward_core_decode_spec_post_conv_fused_norm = \
    lambda self, *a, **k: ORIG_LAYER.append(1)

# fake tilelang.contrib.horizon_ops.gdn_g3v recording expand calls
EXP = []
g3 = types.ModuleType("tilelang.contrib.horizon_ops.gdn_g3v")
g3.ring_geometry = lambda HV, H, RF, SW: (3, 4)
g3.expand = lambda saddr, raddr, lmeta, rows, cnt, sidx, ridx, meta, HV, H, SS, RF, ncta=752: EXP.append(
    (rows.clone(), cnt.clone(), sidx.clone(), ridx.clone()))
g3.step = lambda *a, **k: EXP.append(("step",))
SEL = []


def fake_expand_sel(mode, B_, im, sti, nc, aux, bts, saddr, raddr, lmeta, lgrp, meta, ridx, HV, H, SS, RF, SW_,
                    write_ridx, ncta=752):
    n = im.shape[0]
    rows, cols = [], []
    for r in range(n):
        req = int(im[r])
        col = int(sti[req]) if req >= 0 else -1
        cols.append(col)
        if req < 0 or col < 0:
            continue
        ncv = int(nc[req])
        if mode == 1:
            sel = True
        elif mode == 0:
            sel = (ncv + int(aux[r + 1]) - int(aux[r]) + B_ - 1) // B_ - 1 != col
        else:
            sel = (ncv // B_) * B_ >= ncv - max(int(aux[r]), 1) + 1
        if sel:
            rows.append(r)
    wins = [[[int(bt[r, cols[r] + k]) if 0 <= cols[r] and cols[r] + k < bt.shape[1] else 0 for k in range(SW_)]
             for r in range(n)] for bt in bts]
    if write_ridx:
        ridx.fill_(-1)
        ridx[:n] = im.to(torch.int32)
    SEL.append((mode, rows, wins, [int(x) for x in lgrp.tolist()]))


g3.expand_sel = fake_expand_sel
g3.get_expand_sel_kernel = lambda *a, **k: None
import horizon_kernels  # noqa: E402
ops_pkg = types.ModuleType("horizon_kernels.g3v_ops")
ops_pkg.gdn_g3v = g3
sys.modules["horizon_kernels.g3v_ops"] = ops_pkg
sys.modules["horizon_kernels.g3v_ops.gdn_g3v"] = g3

from horizon_kernels import gdn_g3 as P  # noqa: E402

P.install()
check("install wraps the 4 model-state methods and the layer method",
      all(getattr(getattr(M, n), "_horizon_g3", False) for n in EXPECT)
      and getattr(Q.QwenGatedDeltaNetAttention._forward_core_decode_spec_post_conv_fused_norm, "_horizon_g3", False))

# layer hook without an integrated context -> vLLM's method
lay = types.SimpleNamespace(kv_cache=[torch.zeros(1), torch.zeros(2, 48, 128, 128)], prefix="x")
Q.QwenGatedDeltaNetAttention._forward_core_decode_spec_post_conv_fused_norm(lay, None, None, None, None, None,
                                                                             types.SimpleNamespace(
                                                                                 num_spec_decodes=0,
                                                                                 spec_state_indices_tensor=None))
check("layer hook falls through to vLLM when G3v is not integrated", ORIG_LAYER == [1])

# a fake integrated context
B, SW, NRQ, NB, W = 16, 4, 64, 4000, 40
ctx = types.SimpleNamespace(ok=True, groups=[(g, None, None, None) for g in range(3)], B=B, SW=SW, HV=48, H=16,
                            SS=819200, RF=30720, meta=torch.ones(48, NRQ, 48, 4, dtype=torch.int32),
                            rows=torch.zeros(4097, dtype=torch.int32), cnt=torch.zeros(1, dtype=torch.int32),
                            device=torch.device("cpu"), layers=[("a", None)], ptr0=0, idx={},
                            sel=True, sel_warm=True, gids=[0, 1, 2], all_saddr=torch.zeros(6, dtype=torch.int64),
                            all_raddr=torch.zeros(6, dtype=torch.int64),
                            all_lmeta=torch.arange(6, dtype=torch.int32),
                            all_lgrp=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32))
P._CTX[torch.device("cpu")] = ctx
P._model_ctx = lambda ms, kc: ctx
ms = object.__new__(M)
ms._align_mode = True
rng = np.random.default_rng(0)


def batch(n, pure):
    im = torch.tensor(rng.permutation(NRQ)[:n], dtype=torch.int64)
    if n > 2:
        im[1] = -1                                                   # a PP-filtered row
    ql = np.full(n, 4 if rng.random() < 0.5 else 2)
    d = ql - 1
    if not pure:
        d[0] = 0
        ql[0] = 37
    qsl = torch.zeros(n + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(np.cumsum(ql), dtype=torch.int32)
    ib = types.SimpleNamespace(num_reqs=n, idx_mapping=im, query_start_loc=qsl, num_scheduled_tokens=ql,
                               num_draft_tokens_per_req=d, is_prefilling_np=np.zeros(n, bool))
    return ib


def replica_pre(ib, state_idx, num_computed, pure):
    n = ib.num_reqs
    rows = []
    for r in range(n):
        req = int(ib.idx_mapping[r])
        if req < 0:
            continue
        old = int(state_idx[req])
        if old < 0:
            continue
        if not pure:
            rows.append(r)
            continue
        after = int(num_computed[req]) + int(ib.query_start_loc[r + 1] - ib.query_start_loc[r])
        new = (after + B - 1) // B - 1
        if new != old:
            rows.append(r)
    return rows


bad_pre = bad_post = bad_sidx = bad_mode = bad_ridx = 0
for trial in range(200):
    n = int(rng.integers(1, 40))
    pure = rng.random() < 0.7
    ib = batch(n, pure)
    ms._mamba_state_idx_gpu = torch.tensor(rng.integers(-1, W - 1, NRQ), dtype=torch.int32)
    nc = torch.tensor(rng.integers(1, B * (W - 2), NRQ), dtype=torch.int32)
    bts = tuple(torch.tensor(rng.integers(1, NB, (n + 3, W)), dtype=torch.int32) for _ in range(3))
    SEL.clear(); EXP.clear()
    M.preprocess_state(ms, ib, bts, None, nc)
    want = replica_pre(ib, ms._mamba_state_idx_gpu, nc, P._pure_spec(ib))
    mode, got, wins, lgrp = SEL[0]
    bad_mode += (mode != (0 if P._pure_spec(ib) else 1)) + (len(SEL) != 1) + (len(EXP) != 0)
    bad_pre += sorted(got) != want
    for g in range(3):
        for r in got:
            old = int(ms._mamba_state_idx_gpu[int(ib.idx_mapping[r])])
            exp = [int(bts[g][r, old + k]) if old + k < W else 0 for k in range(SW)]
            bad_sidx += wins[g][r] != exp
    # H4 right after H1 (same batch): no fill / copy of its own, the buffer holds H1's ridx write
    rb = P._ridx(torch.device("cpu"))
    rb.fill_(123)
    SEL.clear()
    snap_calls = len(CALLS)
    M.prepare_attn(ms, ib, None, None, None, None, None, for_capture=False)
    bad_ridx += bool((rb != 123).any())                  # untouched by H4 (H1's launch wrote it)
    bad_ridx += CALLS[snap_calls:] != [("prep", False)]
    ns = torch.tensor(rng.integers(0, 5, n), dtype=torch.int32)
    nc2 = nc.clone()
    SEL.clear(); EXP.clear()
    M.postprocess_state(ms, ib.idx_mapping, ns, nc2)
    want2 = []
    for r in range(n):
        req = int(ib.idx_mapping[r])
        if req < 0 or int(ms._mamba_state_idx_gpu[req]) < 0:
            continue
        acc = max(int(ns[r]), 1)
        newc = int(nc2[req])
        if (newc // B) * B >= newc - acc + 1:
            want2.append(r)
    bad_mode += (SEL[0][0] != 2) + (len(EXP) != 0)
    bad_post += sorted(SEL[0][1]) != want2
check("G3v-sel: one launch per hook, MODE 0 (pure) / 1 (mixed) / 2 (post), no per-group expand", bad_mode == 0,
      f"{bad_mode} mismatches")
check("G3v-sel H1 rows == vLLM's align pre-copy decision (every row for non-pure batches), 200 random batches",
      bad_pre == 0, f"{bad_pre} mismatches")
check("G3v-sel H1 window slots == block_table[row, state_idx + k] (0 past the table end), 3 groups", bad_sidx == 0,
      f"{bad_sidx} mismatches")
check("G3v-sel H4 after H1 on the same batch: no fill / copy (H1's launch wrote the buffer)", bad_ridx == 0,
      f"{bad_ridx} mismatches")
check("G3v-sel H2 rows == vLLM's align post-copy decision, 200 random batches", bad_post == 0, f"{bad_post} mismatches")
# the ridx the kernel model writes == what the old H4 wrote
ib = batch(9, True)
SEL.clear()
M.preprocess_state(ms, ib, bts[:0] or tuple(torch.ones(12, W, dtype=torch.int32) for _ in range(3)), None, nc)
rb = P._ridx(torch.device("cpu"))
check("G3v-sel H1 writes the request-index buffer as H4 did (rows < n = idx_mapping, rest -1)",
      rb[:9].tolist() == ib.idx_mapping.tolist() and bool((rb[9:] == -1).all()))
# fallback: block tables that don't fit the kernel (int64) -> per-group path
SEL.clear(); EXP.clear()
M.preprocess_state(ms, ib, tuple(torch.ones(12, W, dtype=torch.int64) for _ in range(3)), None, nc)
check("G3v-sel falls back to the per-group expand when the block tables don't fit", SEL == [] and len(EXP) == 3)
ctx.sel = False
SEL.clear(); EXP.clear()
M.preprocess_state(ms, ib, tuple(torch.ones(12, W, dtype=torch.int32) for _ in range(3)), None, nc)
check("HORIZON_GDN_G3_SEL=0 (ctx.sel False) -> per-group expand", SEL == [] and len(EXP) == 3)
ctx.sel = True

EXP.clear(); SEL.clear()
M.postprocess_state(ms, ib.idx_mapping, 0, None)
check("H2 skipped for non-speculative steps (num_sampled int)", EXP == [] and SEL == [])
before = ctx.meta.clone()
M.add_request(ms, 7, None)
check("H3 resets exactly meta[:, 7]", bool((ctx.meta[:, 7] == 0).all()) and bool(
    (ctx.meta[:, :7] == before[:, :7]).all() and (ctx.meta[:, 8:] == before[:, 8:]).all()))
ib = batch(5, True)
ms._hz_ridx_batch = None                       # H4 without a preceding H1 launch (dummy runs): its own fill
M.prepare_attn(ms, ib, None, None, None, None, None, for_capture=False)
rb = P._ridx(torch.device("cpu"))
check("H4 fills the request-index buffer", rb[:5].tolist() == ib.idx_mapping.tolist() and bool((rb[5:] == -1).all()))
M.prepare_attn(ms, ib, None, None, None, None, None, for_capture=True)
check("H4 during capture: all -1", bool((rb == -1).all()))
check("originals still called in order", [c[0] for c in CALLS][-3:] == ["add", "prep", "prep"])
check("originals called once per hook (pre / post)", sum(c[0] == "pre" for c in CALLS) == 203 and sum(c[0] == "post" for c in CALLS) == 201, f"pre {sum(c[0] == 'pre' for c in CALLS)} post {sum(c[0] == 'post' for c in CALLS)}")
print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
