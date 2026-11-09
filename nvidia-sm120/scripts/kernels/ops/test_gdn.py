# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# G1 gate for horizon_ops.gdn_decode vs vLLM 0.29's fused_gdn_decode_post_conv_mtp (CUDA kernel, --use_fast_math).
#  1. bit-identity of out and of every written state slot, many shapes / acceptance patterns / edge cases
#  2. 1000-step chained drift test (same inputs to both, states compared bitwise every step)
#  3. determinism (ours twice), CUDA-graph capture (thread_local) + replay == eager
#  4. speed: CUDA-graph replay of LAYERS calls on separate state buffers (> L2), 3 interleaved rounds, fastest
# usage: python test_gdn.py OUT.jsonl [quick]
import json, math, os, sys, time
import torch
from vllm import _custom_ops as ops
from tilelang.contrib.horizon_ops import gdn_decode as gd

OUT = sys.argv[1]
QUICK = len(sys.argv) > 2 and sys.argv[2] == "quick"
dev = torch.device("cuda")
HV, H, DK, DV = 48, 16, 128, 128
CQ = 2 * H * DK + HV * DV
PAD = 1024                         # extra floats per slot (vLLM pads the mamba page)
MODEL = "$HK_DATA/models/Qwen3.8-27B-NVFP4"


def log(rec):
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


def make_state(ns, gen):
    stride = HV * DV * DK + PAD
    flat = torch.randn(ns * stride, device=dev, generator=gen) * 0.05
    return torch.as_strided(flat, (ns, HV, DV, DK), (stride, DV * DK, DK, 1)), flat


def make_inputs(ntoks, gen, qscale=1.0):
    L = sum(ntoks)
    mixed = (torch.nn.functional.silu(torch.randn(L, CQ, device=dev, generator=gen) * qscale)).to(torch.bfloat16)
    ba = torch.randn(L, 2 * HV, device=dev, generator=gen).to(torch.bfloat16) * 2
    b, a = ba[:, :HV], ba[:, HV:]
    qkvz = torch.randn(L, CQ + HV * DV, device=dev, generator=gen).to(torch.bfloat16)
    z = qkvz[:, CQ:].view(L, HV, DV)
    cu = torch.zeros(len(ntoks) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(ntoks), 0)
    return mixed, a, b, z, cu.to(dev)


def call(fn, mixed, a, b, P, sidx, cu, nacc, state, z, out, act="silu", vr=None):
    kw = {} if fn is ops.fused_gdn_decode_post_conv_mtp else {"vr": vr}
    fn(mixed_qkv=mixed, a=a, b=b, A_log=P["A_log"], dt_bias=P["dt_bias"], state_indices=sidx, cu_seqlens=cu,
       num_accepted_tokens=nacc, state=state, output_gate=z, norm_weight=P["norm_w"], out=out,
       scale=DK ** -0.5, norm_eps=1e-6, output_gate_activation=act, **kw)


def bits_equal(x, y):
    """bitwise equality treating +0 / -0 as equal (count of differing elements)"""
    if x.dtype == torch.bfloat16:
        xi, yi = x.view(torch.int16), y.view(torch.int16)
        both_zero = (x == 0) & (y == 0)
    else:
        xi, yi = x.view(torch.int32), y.view(torch.int32)
        both_zero = (x == 0) & (y == 0)
    return int(((xi != yi) & ~both_zero).sum())


def one_case(name, ntoks, acc_list, P, gen, act="silu", vr=None, zero_src=()):
    nr = len(ntoks)
    W = 4
    ns = nr * W + 1
    st_ref, flat_ref = make_state(ns, gen)
    flat_our = flat_ref.clone()
    stride = st_ref.stride(0)
    st_our = torch.as_strided(flat_our, st_ref.shape, st_ref.stride())
    sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W)
    for r in zero_src:
        sidx[r, :] = 0 if r % 2 == 0 else sidx[r, :]
    sidx = sidx.to(dev).contiguous()
    nacc = torch.tensor(acc_list, dtype=torch.int32, device=dev)
    mixed, a, b, z, cu = make_inputs(ntoks, gen)
    L = mixed.shape[0]
    o_ref = torch.full((L, HV, DV), 7.0, device=dev, dtype=torch.bfloat16)
    o_our = o_ref.clone()
    call(ops.fused_gdn_decode_post_conv_mtp, mixed, a, b, P, sidx, cu, nacc, st_ref, z, o_ref, act)
    call(gd.fused_gdn_decode_post_conv_mtp, mixed, a, b, P, sidx, cu, nacc, st_our, z, o_our, act, vr)
    torch.cuda.synchronize()
    d_out = bits_equal(o_ref, o_our)
    d_st = bits_equal(flat_ref, flat_our)
    return {"case": name, "nr": nr, "ntoks": ntoks[:8], "act": act, "vr": vr, "out_diff": d_out, "state_diff": d_st,
            "out_n": o_ref.numel(), "max_abs_out": float((o_ref.float() - o_our.float()).abs().max())}


def correctness(P, tag):
    gen = torch.Generator(device=dev); gen.manual_seed(1234)
    res = []
    for nr in (1, 2, 3, 8, 16, 32):
        for vr in (32, 64, 128):
            accs = [(i % 4) + 1 for i in range(nr)]
            res.append(one_case(f"{tag}_nr{nr}_t4", [4] * nr, accs, P, gen, vr=vr))
    # variable token counts (1..4 and up to 8), acceptance 1..W, a request with 0 tokens, zero / out-of-range src
    res.append(one_case(f"{tag}_var", [1, 4, 2, 3, 9, 0, 4, 1], [1, 2, 3, 4, 4, 1, 0, 5], P, gen, vr=32))
    res.append(one_case(f"{tag}_var64", [1, 4, 2, 3, 9, 0, 4, 1], [1, 2, 3, 4, 4, 1, 0, 5], P, gen, vr=64))
    res.append(one_case(f"{tag}_zero_src", [4, 4, 4, 4], [1, 2, 3, 4], P, gen, vr=32, zero_src=(0, 2)))
    res.append(one_case(f"{tag}_sigmoid", [4, 4], [2, 3], P, gen, act="sigmoid", vr=32))
    res.append(one_case(f"{tag}_default_vr", [4] * 8, [4] * 8, P, gen, vr=None))
    for r in res:
        log(r)
    return sum(r["out_diff"] + r["state_diff"] for r in res)


def drift(P, steps, nr=2, vr=None):
    """chained decode steps: both kernels carry their own state; inputs identical; compared bitwise every step"""
    gen = torch.Generator(device=dev); gen.manual_seed(99)
    W = 4
    ns = nr * W + 1
    st_ref, flat_ref = make_state(ns, gen)
    flat_our = flat_ref.clone()
    st_our = torch.as_strided(flat_our, st_ref.shape, st_ref.stride())
    sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
    first_bad = None
    maxdiff = 0.0
    t0 = time.time()
    for s in range(steps):
        acc = torch.randint(1, W + 1, (nr,), generator=torch.Generator().manual_seed(s), dtype=torch.int32).to(dev)
        mixed, a, b, z, cu = make_inputs([4] * nr, gen)
        o_ref = torch.empty(4 * nr, HV, DV, device=dev, dtype=torch.bfloat16)
        o_our = torch.empty_like(o_ref)
        call(ops.fused_gdn_decode_post_conv_mtp, mixed, a, b, P, sidx, cu, acc, st_ref, z, o_ref)
        call(gd.fused_gdn_decode_post_conv_mtp, mixed, a, b, P, sidx, cu, acc, st_our, z, o_our, vr=vr)
        if s % 50 == 0 or s == steps - 1:
            d = bits_equal(flat_ref, flat_our) + bits_equal(o_ref, o_our)
            if d and first_bad is None:
                first_bad = s
            rel = float((flat_ref - flat_our).abs().max() / flat_ref.abs().max().clamp_min(1e-30))
            maxdiff = max(maxdiff, rel)
    torch.cuda.synchronize()
    rec = {"test": "drift", "steps": steps, "nr": nr, "vr": vr, "first_bad_step": first_bad, "max_rel_state": maxdiff,
           "state_absmax": float(flat_ref.abs().max()), "secs": round(time.time() - t0, 1)}
    log(rec)
    return first_bad is None


def determinism_and_graph(P, nr=8, vr=None):
    gen = torch.Generator(device=dev); gen.manual_seed(5)
    W = 4
    ns = nr * W + 1
    st, flat = make_state(ns, gen)
    base = flat.clone()
    sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
    acc = torch.tensor([(i % 4) + 1 for i in range(nr)], dtype=torch.int32, device=dev)
    mixed, a, b, z, cu = make_inputs([4] * nr, gen)
    outs = []
    for rep in range(3):
        flat.copy_(base)
        o = torch.empty(4 * nr, HV, DV, device=dev, dtype=torch.bfloat16)
        call(gd.fused_gdn_decode_post_conv_mtp, mixed, a, b, P, sidx, cu, acc, st, z, o, vr=vr)
        torch.cuda.synchronize()
        outs.append((o.clone(), flat.clone()))
    det = all(bits_equal(outs[0][0], x[0]) == 0 and bits_equal(outs[0][1], x[1]) == 0 for x in outs[1:])
    # graph
    flat.copy_(base)
    og = torch.empty(4 * nr, HV, DV, device=dev, dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call(gd.fused_gdn_decode_post_conv_mtp, mixed, a, b, P, sidx, cu, acc, st, z, og, vr=vr)  # warm
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    for attempt in range(5):
        try:
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                call(gd.fused_gdn_decode_post_conv_mtp, mixed, a, b, P, sidx, cu, acc, st, z, og, vr=vr)
            break
        except Exception as e:  # noqa: BLE001
            print("capture retry", attempt, e, flush=True)
            g = torch.cuda.CUDAGraph()
    flat.copy_(base)
    og.zero_()
    g.replay()
    torch.cuda.synchronize()
    gr = bits_equal(og, outs[0][0]) == 0 and bits_equal(flat, outs[0][1]) == 0
    log({"test": "determinism_graph", "nr": nr, "vr": vr, "deterministic": det, "graph_replay_equal": gr})
    return det and gr


def bench(P, nrs=(1, 2, 4, 8, 16, 32), vrs=(32, 64, 128), rounds=3, iters=20):
    gen = torch.Generator(device=dev); gen.manual_seed(7)
    W = 4
    for nr in nrs:
        ns = nr * W + 1
        per_layer = ns * (HV * DV * DK + PAD) * 4
        layers = max(4, math.ceil(512e6 / per_layer))
        if layers * per_layer > 6e9:
            layers = max(2, int(6e9 // per_layer))
        states = [make_state(ns, gen)[0] for _ in range(layers)]
        sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
        acc = torch.tensor([(i % 4) + 1 for i in range(nr)], dtype=torch.int32, device=dev)
        mixed, a, b, z, cu = make_inputs([4] * nr, gen)
        o = torch.empty(4 * nr, HV, DV, device=dev, dtype=torch.bfloat16)
        arms = [("vllm", ops.fused_gdn_decode_post_conv_mtp, None)] + [(f"ours_vr{v}", gd.fused_gdn_decode_post_conv_mtp, v) for v in vrs]
        graphs = {}
        for name, fn, vr in arms:
            for st in states[:1]:
                call(fn, mixed, a, b, P, sidx, cu, acc, st, z, o, vr=vr)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            for attempt in range(5):
                try:
                    with torch.cuda.graph(g, capture_error_mode="thread_local"):
                        for st in states:
                            call(fn, mixed, a, b, P, sidx, cu, acc, st, z, o, vr=vr)
                    break
                except Exception as e:  # noqa: BLE001
                    print("capture retry", attempt, e, flush=True)
                    g = torch.cuda.CUDAGraph()
            graphs[name] = g
        best = {n: 1e30 for n in graphs}
        for rnd in range(rounds):
            for name, g in graphs.items():
                g.replay(); torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ts = []
                for _ in range(iters):
                    e0.record(); g.replay(); e1.record(); e1.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1000 / layers)
                ts.sort()
                best[name] = min(best[name], ts[len(ts) // 2])
        state_mb = nr * (1 + 4) * HV * DV * DK * 4 / 1e6
        rec = {"test": "bench", "nr": nr, "tokens": 4 * nr, "layers": layers, "state_rw_mb_per_call": round(state_mb, 2),
               "floor_us_at_1630GBs": round(state_mb / 1.63e3 * 1e3, 2)}
        rec.update({k: round(v, 2) for k, v in best.items()})
        log(rec)
        del states
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    total = 0
    for dt in (torch.bfloat16, torch.float32):
        P = real_params(0, dt, dt)
        total += correctness(P, f"dt_{str(dt).split('.')[-1]}")
    # MoE model shape (Qwen3.6-35B-A3B: 32 value heads, 16 key heads), random parameters
    HV, CQ = 32, 2 * H * DK + 32 * DV
    Pm = {"A_log": torch.randn(HV, device=dev) * 0.5, "dt_bias": torch.randn(HV, device=dev).to(torch.bfloat16),
          "norm_w": torch.randn(DV, device=dev).to(torch.bfloat16)}
    total += correctness(Pm, "moe_hv32")
    HV, CQ = 48, 2 * H * DK + 48 * DV
    P = real_params(0)
    P40 = real_params(40)
    ok_drift = drift(P, 200 if QUICK else 1000, nr=2) and drift(P40, 100 if QUICK else 500, nr=8)
    ok_graph = determinism_and_graph(P, 8) and determinism_and_graph(P, 1, 32)
    log({"summary": "gdn_g1", "bit_diffs_total": total, "drift_ok": ok_drift, "graph_det_ok": ok_graph})
    bench(P, nrs=(1, 8) if QUICK else (1, 2, 4, 8, 16, 32))
    print("TEST_GDN_DONE", flush=True)
