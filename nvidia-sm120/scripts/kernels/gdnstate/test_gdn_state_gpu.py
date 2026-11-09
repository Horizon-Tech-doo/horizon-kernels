# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GPU gate for the GDN state kernels (horizon_ops.gdn_state, branch horizon/gdn-state), 29-09-2026. MEASURE ONLY.

usage: python test_gdn_state_gpu.py OUT.jsonl --pkg-dir PKG [quick]
  PKG contains the package hzgs/ = copies of horizon_ops/{__init__,gdn_decode,gdn_state}.py. vLLM must be stopped
  (clean timings); run under the GPU lock with the baseline env's LD_LIBRARY_PATH (forward-compat libcuda).
Sections (records in OUT.jsonl; verify_gdn_state_results.py checks completeness and the correctness gates):
  window       G1S vs vLLM's CUDA kernel on identical inputs, float32 and bfloat16 state: out + every state slot
               bit-identical; nr 1/2/8/32 x QL 2/4 x rows per CTA 32/64/128, variable token counts, zero slots, sigmoid
  window_fp16  float16 state vs our float32 kernel on the widened state + torch RNE / saturating rounding: bit-identical
  replay_chain G3 (float32 checkpoint) chained vs vLLM's kernel chained, QL switching 2 <-> 4 between steps (the draft
               schedule), random acceptance, padding requests, forced flushes; outputs bit-identical every step, the
               checkpoint after materialize bit-identical to vLLM's slot state_idx[r, n_acc - 1]
  drift        report only: bf16 / fp16 window and G3 (bf16 / fp16 checkpoint) vs float32 over 2000 chained steps with
               real gating parameters (relative error of the outputs)
  graph        determinism (3 runs) + CUDA-graph capture (thread_local) replay == eager
  bench        per-layer us at nr 1/8/32 x QL 2/4: vLLM f32 / bf16, G1S f32 / bf16 / f16, G3 f32 / bf16 / f16
               (+ G3 f32 with HALF 32); CUDA-graph replay over > 512 MB of rotated state buffers, 3 interleaved rounds
"""
import argparse
import json
import math
import os
import sys
import time

import torch

# globals bound in main() (or by the CPU harness dry-run, test_gdn_state_harness_cpu.py, with reference fakes)
ops = GS = None
QUICK = False
OUT = None
dev = None
HV, H, DK, DV = 48, 16, 128, 128
CQ = 2 * H * DK + HV * DV            # 10240
QKVZ = CQ + HV * DV                   # 16384: production row stride of mixed_qkv and z (views of one tensor)
SLOT = 819200                         # production state slot stride in elements (padded mamba page)
W = 4
MODEL = "$HK_DATA/models/Qwen3.8-27B-NVFP4"
T0 = time.time()


def sync():
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize()


def log(rec):
    rec["t"] = round(time.time() - T0, 1)
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def real_params(layer=0, dt_dtype=torch.bfloat16, nw_dtype=torch.bfloat16):
    from safetensors import safe_open
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    p = f"model.language_model.layers.{layer}.linear_attn."
    out = {}
    with safe_open(f"{MODEL}/{idx[p + 'A_log']}", "pt") as f:
        out["A_log"] = f.get_tensor(p + "A_log").float().to(dev).contiguous()
        out["dt_bias"] = f.get_tensor(p + "dt_bias").to(dt_dtype).to(dev).contiguous()
        out["norm_w"] = f.get_tensor(p + "norm.weight").to(nw_dtype).to(dev).contiguous()
    return out


def make_state(ns, dtype, gen, amp=0.05):
    x = torch.randn(ns * SLOT, device=dev, generator=gen) * amp
    flat = x if dtype == torch.float32 else store_round(x, dtype)
    return torch.as_strided(flat, (ns, HV, DV, DK), (SLOT, DV * DK, DK, 1)), flat


def view_like(flat, st):
    return torch.as_strided(flat, st.shape, st.stride())


def make_inputs(ntoks, gen, amp=1.0):
    L = max(1, sum(ntoks))
    qkvz = torch.nn.functional.silu(torch.randn(L, QKVZ, device=dev, generator=gen) * amp).to(torch.bfloat16)
    mixed = qkvz[:, :CQ]
    z = qkvz[:, CQ:].view(L, HV, DV)
    ba = (torch.randn(L, 2 * HV, device=dev, generator=gen) * 2).to(torch.bfloat16)
    b, a = ba[:, :HV], ba[:, HV:]
    cu = torch.zeros(len(ntoks) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(ntoks), 0)
    return mixed, a, b, z, cu.to(dev)


def vllm_call(mixed, a, b, P, sidx, cu, nacc, state, z, out, act="silu"):
    ops.fused_gdn_decode_post_conv_mtp(mixed_qkv=mixed, a=a, b=b, A_log=P["A_log"], dt_bias=P["dt_bias"],
                                       state_indices=sidx, cu_seqlens=cu, num_accepted_tokens=nacc, state=state,
                                       output_gate=z, norm_weight=P["norm_w"], out=out, scale=DK ** -0.5,
                                       norm_eps=1e-6, output_gate_activation=act)


def g1s_call(mixed, a, b, P, sidx, cu, nacc, state, z, out, act="silu", vr=None):
    GS.fused_gdn_decode_post_conv_mtp(mixed, a, b, P["A_log"], P["dt_bias"], sidx, cu, nacc, state, z, P["norm_w"],
                                      out, DK ** -0.5, 1e-6, act, vr=vr)


def g3_call(mixed, a, b, P, cidx, ridx, cu, nacc, ff, state, ring, z, out, act="silu", vr=None):
    GS.replay_step(mixed, a, b, P["A_log"], P["dt_bias"], cidx, ridx, cu, nacc, ff, state, ring, z, P["norm_w"], out,
                   DK ** -0.5, 1e-6, act, vr=vr)


def bits_diff(x, y):
    if x.dtype in (torch.bfloat16, torch.float16):
        xi, yi = x.view(torch.int16), y.view(torch.int16)
    else:
        xi, yi = x.view(torch.int32), y.view(torch.int32)
    both_zero = (x == 0) & (y == 0)
    return int(((xi != yi) & ~both_zero).sum())


def store_round(x, dtype):
    if dtype == torch.float16:
        return x.clamp(-65504.0, 65504.0).to(torch.float16)
    return x.to(dtype)


# ------------------------------------------------------------------------------------------------------ A window
def window_case(name, sdt, ntoks, accs, P, gen, vr=None, act="silu", zero_src=(), ql=None):
    nr = len(ntoks)
    ns = nr * W + 1
    st_ref, flat_ref = make_state(ns, sdt, gen)
    flat_our = flat_ref.clone()
    st_our = view_like(flat_our, st_ref)
    sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W)
    for r in zero_src:
        sidx[r, :] = 0
    sidx = sidx.to(dev).contiguous()
    nacc = torch.tensor(accs, dtype=torch.int32, device=dev)
    mixed, a, b, z, cu = make_inputs(ntoks, gen)
    L = mixed.shape[0]
    o_ref = torch.full((L, HV, DV), 7.0, device=dev, dtype=torch.bfloat16)
    o_our = o_ref.clone()
    vllm_call(mixed, a, b, P, sidx, cu, nacc, st_ref, z, o_ref, act)
    g1s_call(mixed, a, b, P, sidx, cu, nacc, st_our, z, o_our, act, vr)
    sync()
    rec = {"test": "window", "case": name, "sdt": str(sdt).split(".")[-1], "nr": nr, "ql": ql, "vr": vr, "act": act,
           "out_diff": bits_diff(o_ref, o_our), "state_diff": bits_diff(flat_ref, flat_our)}
    log(rec)
    return rec["out_diff"] + rec["state_diff"]


def g1_report(P):
    """report only: horizon_ops.gdn_decode (G1 as on horizon/ops-bodies since 56935fbe, explicit-PTX helpers) vs
    vLLM, one case; documents whether G1 itself is still bit-identical (the 27-09 23:06 ops-gate said no)"""
    try:
        from hzgs import gdn_decode as GD
    except Exception as e:  # noqa: BLE001
        log({"test": "g1_report", "error": repr(e)[:300]}); return
    gen = torch.Generator(device=dev); gen.manual_seed(4321)
    nr, ql = 8, 4
    st_ref, flat_ref = make_state(nr * W + 1, torch.float32, gen)
    flat_our = flat_ref.clone(); st_our = view_like(flat_our, st_ref)
    sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
    nacc = torch.tensor([(i % ql) + 1 for i in range(nr)], dtype=torch.int32, device=dev)
    mixed, a, b, z, cu = make_inputs([ql] * nr, gen)
    o_ref = torch.zeros(mixed.shape[0], HV, DV, device=dev, dtype=torch.bfloat16); o_our = o_ref.clone()
    vllm_call(mixed, a, b, P, sidx, cu, nacc, st_ref, z, o_ref)
    GD.fused_gdn_decode_post_conv_mtp(mixed, a, b, P["A_log"], P["dt_bias"], sidx, cu, nacc, st_our, z, P["norm_w"],
                                      o_our, DK ** -0.5, 1e-6, "silu", vr=32)
    sync()
    log({"test": "g1_report", "report_only": True, "out_diff": bits_diff(o_ref, o_our),
         "state_diff": bits_diff(flat_ref, flat_our)})


def section_window(P):
    gen = torch.Generator(device=dev); gen.manual_seed(1234)
    bad = 0
    for sdt in (torch.float32, torch.bfloat16):
        tag = str(sdt).split(".")[-1]
        for nr in (1, 2, 8, 32):
            for ql in (2, 4):
                for vr in ((32,) if QUICK else (32, 64, 128)):
                    accs = [(i % ql) + 1 for i in range(nr)]
                    bad += window_case(f"{tag}_nr{nr}_ql{ql}_vr{vr}", sdt, [ql] * nr, accs, P, gen, vr=vr, ql=ql)
        bad += window_case(f"{tag}_var", sdt, [1, 4, 2, 3, 9, 0, 4, 1], [1, 2, 3, 4, 4, 1, 0, 5], P, gen, vr=32)
        bad += window_case(f"{tag}_zero_src", sdt, [4, 4, 4, 4], [1, 2, 3, 4], P, gen, vr=32, zero_src=(0, 2), ql=4)
        bad += window_case(f"{tag}_sigmoid", sdt, [4, 4], [2, 3], P, gen, vr=32, act="sigmoid", ql=4)
        bad += window_case(f"{tag}_default_vr", sdt, [2] * 8, [2] * 8, P, gen, vr=None, ql=2)
    return bad


# ------------------------------------------------------------------------------------------------- B window fp16
def fp16_case(name, nr, ql, P, gen, amp_state=0.05, amp_in=1.0):
    ns = nr * W + 1
    st16, flat16 = make_state(ns, torch.float16, gen, amp_state)
    flat32 = flat16.float()
    st32 = view_like(flat32, st16)
    sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
    nacc = torch.tensor([(i % ql) + 1 for i in range(nr)], dtype=torch.int32, device=dev)
    mixed, a, b, z, cu = make_inputs([ql] * nr, gen, amp_in)
    L = mixed.shape[0]
    o16 = torch.full((L, HV, DV), 7.0, device=dev, dtype=torch.bfloat16)
    o32 = o16.clone()
    g1s_call(mixed, a, b, P, sidx, cu, nacc, st16, z, o16, vr=32)
    g1s_call(mixed, a, b, P, sidx, cu, nacc, st32, z, o32, vr=32)
    sync()
    expect = store_round(flat32, torch.float16)
    rec = {"test": "window_fp16", "case": name, "nr": nr, "ql": ql, "out_diff": bits_diff(o16, o32),
           "state_diff": bits_diff(expect, flat16), "state_absmax": float(flat32.abs().max()),
           "saturated": int((flat32.abs() > 65504).sum())}
    log(rec)
    return rec["out_diff"] + rec["state_diff"]


def section_fp16(P):
    gen = torch.Generator(device=dev); gen.manual_seed(77)
    bad = 0
    for nr, ql in ((1, 2), (1, 4), (8, 2), (8, 4), (32, 4)):
        bad += fp16_case(f"f16_nr{nr}_ql{ql}", nr, ql, P, gen)
    bad += fp16_case("f16_saturation", 2, 4, P, gen, amp_state=3.0e4, amp_in=40.0)
    return bad


# ------------------------------------------------------------------------------------------------ C replay chain
def replay_chain(P, nr, steps, half, seed, flush_p=0.03, mat_every=25):
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    rnd = torch.Generator().manual_seed(seed)
    ns = nr * W + 1
    st_v, flat_v = make_state(ns, torch.float32, gen)
    sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
    st_g = torch.as_strided(torch.zeros((nr + 1) * SLOT, device=dev), (nr + 1, HV, DV, DK), (SLOT, DV * DK, DK, 1))
    cidx = torch.arange(1, nr + 1, dtype=torch.int32, device=dev)
    ridx = torch.arange(nr, dtype=torch.int32, device=dev)
    st_g[1:] = st_v[sidx[:, 0].long()]
    ring = GS.ring_buffers(nr, HV, H, half, dev)
    flat_w = flat_v.clone()                      # our G1S window chain (fp32), same protocol as vLLM
    st_w = view_like(flat_w, st_v)
    nacc = torch.ones(nr, dtype=torch.int32)
    out_bad, ck_bad, flushes, mats = 0, 0, 0, 0
    out_bad_w, ck_bad_w, win_bad = 0, 0, 0
    half_prev = ring[3][:, 0, 0].clone() // 65536
    for s in range(steps):
        ql = 4 if torch.rand(1, generator=rnd).item() < 0.5 else 2      # the draft schedule switches 3 <-> 1 drafts
        ntoks = [ql] * nr
        if s % 13 == 7 and nr > 1:
            ntoks[-1] = 0                                              # CUDA-graph padding request
        mixed, a, b, z, cu = make_inputs(ntoks, gen)
        L = mixed.shape[0]
        ov = torch.full((L, HV, DV), 7.0, device=dev, dtype=torch.bfloat16)
        og = ov.clone()
        na = nacc.to(dev)
        ff = (torch.rand(nr, generator=rnd) < flush_p).to(torch.int32).to(dev)
        ow = ov.clone()
        vllm_call(mixed, a, b, P, sidx, cu, na, st_v, z, ov)
        g1s_call(mixed, a, b, P, sidx, cu, na, st_w, z, ow, vr=32)
        g3_call(mixed, a, b, P, cidx, ridx, cu, na, ff, st_g, ring, z, og)
        out_bad += bits_diff(ov, og)
        out_bad_w += bits_diff(ow, og)
        win_bad += bits_diff(ov, ow)
        h_now = ring[3][:, 0, 0] // 65536
        flushes += int((h_now != half_prev).sum())
        half_prev = h_now.clone()
        new = nacc.clone()
        for r in range(nr):
            if ntoks[r] > 0:
                new[r] = int(torch.randint(1, ntoks[r] + 1, (1,), generator=rnd))
        nacc = new
        if s % mat_every == mat_every - 1:
            GS.materialize(cidx, ridx, nacc.to(dev), st_g, ring)
            mats += 1
            vs = sidx[torch.arange(nr, device=dev), nacc.to(dev).long() - 1].long()
            ck_bad += bits_diff(st_v[vs].contiguous(), st_g[1:].contiguous())
            ck_bad_w += bits_diff(st_w[vs].contiguous(), st_g[1:].contiguous())
    sync()
    rec = {"test": "replay_chain", "sdt": "float32", "nr": nr, "half": half, "steps": steps, "out_diff": out_bad,
           "ckpt_diff": ck_bad, "flushes": flushes, "materializes": mats,
           "diag_out_diff_vs_g1s": out_bad_w, "diag_ckpt_diff_vs_g1s": ck_bad_w, "diag_g1s_vs_vllm_out": win_bad}
    log(rec)
    return out_bad + ck_bad


def section_replay(P):
    bad = 0
    plan = [(2, 300, 16, 11), (8, 150, 16, 12), (8, 150, 32, 13), (32, 60, 16, 14)]
    if QUICK:
        plan = [(2, 100, 16, 11), (8, 60, 16, 12), (32, 30, 16, 14)]
    for nr, steps, half, seed in plan:
        bad += replay_chain(P, nr, steps, half, seed)
    return bad


# -------------------------------------------------------------------------------------------------------- D drift
def section_drift(P, steps=2000, nr=2):
    """report only: 16-bit window and G3 checkpoints vs the float32 window chain, same inputs"""
    arms = {"window_f32": ("w", torch.float32), "window_bf16": ("w", torch.bfloat16),
            "window_f16": ("w", torch.float16), "g3_bf16": ("g", torch.bfloat16), "g3_f16": ("g", torch.float16)}
    gen0 = torch.Generator(device=dev); gen0.manual_seed(5)
    init = torch.randn(nr, HV, DV, DK, device=dev, generator=gen0) * 0.05
    st = {}
    for name, (kind, dt) in arms.items():
        if kind == "w":
            s = torch.as_strided(torch.zeros((nr * W + 1) * SLOT, device=dev, dtype=dt), (nr * W + 1, HV, DV, DK),
                                 (SLOT, DV * DK, DK, 1))
            sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
            s[sidx[:, 0].long()] = store_round(init, dt)
            st[name] = (s, sidx, None)
        else:
            s = torch.as_strided(torch.zeros((nr + 1) * SLOT, device=dev, dtype=dt), (nr + 1, HV, DV, DK),
                                 (SLOT, DV * DK, DK, 1))
            s[1:] = store_round(init, dt)
            st[name] = (s, None, GS.ring_buffers(nr, HV, H, 16, dev))
    cidx = torch.arange(1, nr + 1, dtype=torch.int32, device=dev)
    ridx = torch.arange(nr, dtype=torch.int32, device=dev)
    ff = torch.zeros(nr, dtype=torch.int32, device=dev)
    rnd = torch.Generator().manual_seed(3)
    gen = torch.Generator(device=dev); gen.manual_seed(6)
    nacc = torch.ones(nr, dtype=torch.int32, device=dev)
    errs = {k: [] for k in arms if k != "window_f32"}
    for s_ in range(steps):
        ql = 4 if torch.rand(1, generator=rnd).item() < 0.5 else 2
        mixed, a, b, z, cu = make_inputs([ql] * nr, gen)
        outs = {}
        for name, (kind, dt) in arms.items():
            o = torch.empty(mixed.shape[0], HV, DV, device=dev, dtype=torch.bfloat16)
            s, sidx, ring = st[name]
            if kind == "w":
                g1s_call(mixed, a, b, P, sidx, cu, nacc, s, z, o, vr=32)
            else:
                g3_call(mixed, a, b, P, cidx, ridx, cu, nacc, ff, s, ring, z, o, vr=32)
            outs[name] = o.float()
        ref = outs["window_f32"]
        if s_ % 10 == 0:
            for k in errs:
                e = (outs[k] - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)
                errs[k].append(e.flatten())
        nacc = torch.randint(1, ql + 1, (nr,), generator=rnd, dtype=torch.int32).to(dev)
    rec = {"test": "drift", "steps": steps, "nr": nr, "report_only": True}
    for k, v in errs.items():
        e = torch.cat(v)
        rec[k] = {"median": float(e.median()), "p99": float(torch.quantile(e[:1_000_000], 0.99))}
    log(rec)


# -------------------------------------------------------------------------------------------------------- E graph
def _graph(fn):
    g = torch.cuda.CUDAGraph()
    for attempt in range(5):
        try:
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                fn()
            return g
        except Exception as e:  # noqa: BLE001
            print("capture retry", attempt, e, flush=True)
            sync()
            g = torch.cuda.CUDAGraph()
    raise RuntimeError("capture failed 5 times")


def section_graph(P, nr=8, ql=4):
    ok_all = True
    gen = torch.Generator(device=dev); gen.manual_seed(5)
    mixed, a, b, z, cu = make_inputs([ql] * nr, gen)
    L = mixed.shape[0]
    nacc = torch.tensor([(i % ql) + 1 for i in range(nr)], dtype=torch.int32, device=dev)
    for sdt in (torch.float32, torch.bfloat16, torch.float16):
        st, flat = make_state(nr * W + 1, sdt, gen)
        base = flat.clone()
        sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
        o = torch.empty(L, HV, DV, device=dev, dtype=torch.bfloat16)
        res = []
        for _ in range(3):
            flat.copy_(base)
            g1s_call(mixed, a, b, P, sidx, cu, nacc, st, z, o, vr=32)
            sync()
            res.append((o.clone(), flat.clone()))
        det = all(bits_diff(res[0][0], x[0]) == 0 and bits_diff(res[0][1], x[1]) == 0 for x in res[1:])
        flat.copy_(base)
        g = _graph(lambda: g1s_call(mixed, a, b, P, sidx, cu, nacc, st, z, o, vr=32))
        flat.copy_(base); o.zero_()
        g.replay(); sync()
        eq = bits_diff(o, res[0][0]) == 0 and bits_diff(flat, res[0][1]) == 0
        flat.copy_(base); o.zero_()
        g.replay(); sync()
        eq = eq and bits_diff(o, res[0][0]) == 0 and bits_diff(flat, res[0][1]) == 0
        log({"test": "graph", "kernel": f"window_{str(sdt).split('.')[-1]}", "deterministic": det,
             "graph_replay_equal": eq})
        ok_all = ok_all and det and eq
    for sdt in (torch.float32, torch.bfloat16):
        st, flat = make_state(nr + 1, sdt, gen)
        base = flat.clone()
        cidx = torch.arange(1, nr + 1, dtype=torch.int32, device=dev)
        ridx = torch.arange(nr, dtype=torch.int32, device=dev)
        ring = GS.ring_buffers(nr, HV, H, 16, dev)
        # a ring with content: 5 plain steps first (records + meta), then snapshot everything
        ff = torch.zeros(nr, dtype=torch.int32, device=dev)
        o = torch.empty(L, HV, DV, device=dev, dtype=torch.bfloat16)
        for _ in range(5):
            g3_call(mixed, a, b, P, cidx, ridx, cu, nacc, ff, st, ring, z, o, vr=32)
        snap = [x.clone() for x in ring] + [flat.clone()]

        def restore():
            for x, y in zip(list(ring) + [flat], snap):
                x.copy_(y)

        res = []
        for _ in range(3):
            restore()
            g3_call(mixed, a, b, P, cidx, ridx, cu, nacc, ff, st, ring, z, o, vr=32)
            sync()
            res.append([o.clone(), flat.clone()] + [x.clone() for x in ring])
        det = all(all(bits_diff(p, q) == 0 for p, q in zip(res[0], x)) for x in res[1:])
        restore()
        g = _graph(lambda: g3_call(mixed, a, b, P, cidx, ridx, cu, nacc, ff, st, ring, z, o, vr=32))
        eq = True
        for _ in range(2):
            restore(); o.zero_()
            g.replay(); sync()
            now = [o, flat] + list(ring)
            eq = eq and all(bits_diff(p, q) == 0 for p, q in zip(res[0], now))
        log({"test": "graph", "kernel": f"replay_{str(sdt).split('.')[-1]}", "deterministic": det,
             "graph_replay_equal": eq})
        ok_all = ok_all and det and eq
    return ok_all


# -------------------------------------------------------------------------------------------------------- F bench
def section_bench(P, rounds=3, iters=20):
    gen = torch.Generator(device=dev); gen.manual_seed(7)
    for nr in (1, 8, 32):
        for ql in (2, 4):
            mixed, a, b, z, cu = make_inputs([ql] * nr, gen)
            L = mixed.shape[0]
            o = torch.empty(L, HV, DV, device=dev, dtype=torch.bfloat16)
            nacc = torch.tensor([(i % ql) + 1 for i in range(nr)], dtype=torch.int32, device=dev)
            sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
            cidx = torch.arange(1, nr + 1, dtype=torch.int32, device=dev)
            ridx = torch.arange(nr, dtype=torch.int32, device=dev)
            ff = torch.zeros(nr, dtype=torch.int32, device=dev)
            arms = {}

            def layers_for(per_layer):
                n = max(4, math.ceil(600e6 / per_layer))
                return min(n, max(2, int(8e9 // per_layer)))

            for name, dt in (("vllm_f32", torch.float32), ("vllm_bf16", torch.bfloat16), ("g1s_f32", torch.float32),
                             ("g1s_bf16", torch.bfloat16), ("g1s_f16", torch.float16)):
                per = (nr * W + 1) * SLOT * dt.itemsize
                nl = layers_for(per)
                sts = [make_state(nr * W + 1, dt, gen)[0] for _ in range(nl)]
                fn = vllm_call if name.startswith("vllm") else g1s_call
                arms[name] = (nl, (lambda fn=fn, sts=sts: [fn(mixed, a, b, P, sidx, cu, nacc, s, z, o) for s in sts]),
                              sts)
            for name, dt, half in (("g3_f32", torch.float32, 16), ("g3_bf16", torch.bfloat16, 16),
                                   ("g3_f16", torch.float16, 16), ("g3_f32_h32", torch.float32, 32)):
                ring_bytes = nr * 2 * half * (HV * DV + H * DK + HV) * 4
                per = (nr + 1) * SLOT * dt.itemsize + ring_bytes
                nl = layers_for(per)
                sts = [(make_state(nr + 1, dt, gen)[0], GS.ring_buffers(nr, HV, H, half, dev)) for _ in range(nl)]
                arms[name] = (nl, (lambda sts=sts: [g3_call(mixed, a, b, P, cidx, ridx, cu, nacc, ff, s, rg, z, o)
                                                    for s, rg in sts]), sts)
            graphs = {}
            for name, (nl, fn, _) in arms.items():
                for _ in range(3):
                    fn()                    # warm: compile, and bring the G3 rings to a steady state
                sync()
                graphs[name] = (nl, _graph(fn))
            best = {n: float("inf") for n in graphs}
            order = list(graphs)
            for rnd_ in range(rounds):
                seq = order[rnd_ % len(order):] + order[:rnd_ % len(order)]
                for name in seq:
                    nl, g = graphs[name]
                    g.replay(); sync()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    ts = []
                    for _ in range(iters):
                        e0.record(); g.replay(); e1.record(); e1.synchronize()
                        ts.append(e0.elapsed_time(e1) * 1000 / nl)
                    ts.sort()
                    best[name] = min(best[name], ts[len(ts) // 2])
            S32 = HV * DV * DK * 4
            floor = {"window_f32_MB": nr * (1 + ql) * S32 / 1e6, "window_16bit_MB": nr * (1 + ql) * S32 / 2e6}
            log({"test": "bench", "nr": nr, "ql": ql, "us": {k: round(v, 2) for k, v in best.items()},
                 "layers": {k: v[0] for k, v in graphs.items()}, "state_traffic": floor,
                 "floor_us_window_f32_at_1630GBs": round(floor["window_f32_MB"] / 1.63e3 * 1e3, 2)})
            del arms, graphs
            torch.cuda.empty_cache()


def main():
    global ops, GS, QUICK, OUT, dev
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("mode", nargs="?", default="full")
    ap.add_argument("--pkg-dir", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.pkg_dir)
    from vllm import _custom_ops as _ops
    from hzgs import gdn_state as _gs
    ops, GS = _ops, _gs
    QUICK = args.mode == "quick"
    OUT = args.out
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    P = real_params(0)
    log({"test": "start", "mode": args.mode, "pkg": args.pkg_dir, "torch": torch.__version__,
         "device": torch.cuda.get_device_name()})
    bad_w = section_window(P)
    try:
        g1_report(P)
    except Exception as e:  # noqa: BLE001
        log({"test": "g1_report", "error": repr(e)[:300]})
    bad_f = section_fp16(P)
    bad_r = section_replay(P)
    ok_g = section_graph(P)
    try:
        section_drift(real_params(24), steps=500 if QUICK else 2000)
    except Exception as e:  # noqa: BLE001  (report-only section must not block the gate)
        log({"test": "drift", "error": repr(e)[:500]})
    section_bench(P)
    log({"test": "end", "complete": True, "window_bad": bad_w, "fp16_bad": bad_f, "replay_bad": bad_r,
         "graph_ok": ok_g, "secs": round(time.time() - T0, 1)})


if __name__ == "__main__":
    main()
