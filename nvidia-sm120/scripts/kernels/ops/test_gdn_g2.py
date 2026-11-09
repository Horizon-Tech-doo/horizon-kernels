# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# G2 standalone gate: verify (no state write) + commit (after "sampling") vs G1 (per-token states), bit-identical,
# chained over many steps with random acceptance; plus the speed of verify + commit vs G1 vs vLLM per layer.
# usage: python test_gdn_g2.py OUT.jsonl [pkg]      (pkg: horizon_ops or horizon_ops_wip)
import importlib, json, math, sys, time
import torch
from vllm import _custom_ops as ops

OUT = sys.argv[1]
PKG = sys.argv[2] if len(sys.argv) > 2 else "horizon_ops"
gd = importlib.import_module(f"tilelang.contrib.{PKG}.gdn_decode")
gr = importlib.import_module(f"tilelang.contrib.{PKG}.gdn_recover")
sys.path.insert(0, "$HK_WORK/ops")
import test_gdn as tg   # noqa: E402  (helpers: real_params, make_state, make_inputs, bits_equal)

dev = torch.device("cuda")
HV, H, DK, DV, T_MAX = 48, 16, 128, 128, 8


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def buffers(nl, rm):
    return (torch.zeros(nl, rm, HV, T_MAX, DV, device=dev), torch.zeros(nl, rm, H, T_MAX, DK, device=dev),
            torch.zeros(nl, rm, HV, T_MAX, device=dev))


def chained(P, steps, nr, vr=None, nl=2):
    """nl layers x nr requests. G1 arm: vLLM-style slots (4 per request, read slot acc-1). G2 arm: 1 checkpoint slot
    per request, verify + commit. Same inputs. Compared every step: outputs, and G2's checkpoint vs G1's state in the
    slot of the last accepted token."""
    gen = torch.Generator(device=dev); gen.manual_seed(11)
    W = 4
    g1_states, g2_states = [], []
    for _ in range(nl):
        st, flat = tg.make_state(nr * W + 1, gen)
        g1_states.append(st)
        st2 = torch.zeros(nr + 1, HV, DV, DK, device=dev)
        g2_states.append(torch.as_strided(torch.zeros((nr + 1) * st.stride(0), device=dev), (nr + 1, HV, DV, DK),
                                          st.stride()))
    # both start from the same state: G1 slot for "acc = 1" = column 0
    sidx1 = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
    sidx2 = (torch.arange(nr, dtype=torch.int32) + 1).view(nr, 1).to(dev)
    for li in range(nl):
        for r in range(nr):
            g2_states[li][r + 1].copy_(g1_states[li][int(sidx1[r, 0])])
    corr, kn, dbuf = buffers(nl, nr)
    addrs = torch.tensor([s.data_ptr() for s in g2_states], dtype=torch.int64, device=dev)
    acc = torch.ones(nr, dtype=torch.int32, device=dev)
    ones = torch.ones(nr, dtype=torch.int32, device=dev)
    bad = None
    t0 = time.time()
    for s in range(steps):
        mixed, a, b, z, cu = tg.make_inputs([4] * nr, gen)
        for li in range(nl):
            o1 = torch.empty(4 * nr, HV, DV, device=dev, dtype=torch.bfloat16)
            o2 = torch.empty_like(o1)
            gd.fused_gdn_decode_post_conv_mtp(mixed, a, b, P["A_log"], P["dt_bias"], sidx1, cu, acc, g1_states[li], z,
                                              P["norm_w"], o1, DK ** -0.5, 1e-6, "silu", vr=vr)
            gr.verify(mixed, a, b, P["A_log"], P["dt_bias"], sidx2, cu, ones, g2_states[li], z, P["norm_w"], o2,
                      corr[li], kn[li], dbuf[li], DK ** -0.5, 1e-6, "silu", vr=vr)
            if tg.bits_equal(o1, o2) and bad is None:
                bad = ("out", s, li)
        # "sampling": new acceptance 1..4 per request
        acc = torch.randint(1, W + 1, (nr,), generator=torch.Generator().manual_seed(1000 + s),
                            dtype=torch.int32).to(dev)
        src = sidx2[:, 0].contiguous()
        zero = torch.zeros(nr, dtype=torch.int32, device=dev)
        gr.commit(addrs, src, src, zero, zero, acc, corr, kn, dbuf, HV, H, g2_states[0].stride(0))
        if s % 25 == 0 or s == steps - 1:
            for li in range(nl):
                for r in range(nr):
                    x = g1_states[li][int(sidx1[r, int(acc[r]) - 1])]
                    y = g2_states[li][r + 1]
                    if tg.bits_equal(x, y) and bad is None:
                        bad = ("state", s, li, r)
    torch.cuda.synchronize()
    log({"test": "g2_chained", "steps": steps, "nr": nr, "layers": nl, "vr": vr, "first_bad": bad,
         "secs": round(time.time() - t0, 1)})
    return bad is None


def bench(P, nrs=(1, 8, 32), iters=20, rounds=3):
    gen = torch.Generator(device=dev); gen.manual_seed(3)
    W = 4
    for nr in nrs:
        per_layer_g1 = (nr * W + 1) * (HV * DV * DK + tg.PAD) * 4
        layers = max(4, math.ceil(512e6 / per_layer_g1))
        if layers * per_layer_g1 > 6e9:
            layers = max(2, int(6e9 // per_layer_g1))
        s1 = [tg.make_state(nr * W + 1, gen)[0] for _ in range(layers)]
        s2 = [tg.make_state(nr + 1, gen)[0] for _ in range(layers)]
        sidx1 = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
        sidx2 = (torch.arange(nr, dtype=torch.int32) + 1).view(nr, 1).to(dev)
        acc = torch.tensor([(i % 4) + 1 for i in range(nr)], dtype=torch.int32, device=dev)
        ones = torch.ones(nr, dtype=torch.int32, device=dev)
        corr, kn, dbuf = buffers(layers, nr)
        addrs = torch.tensor([s.data_ptr() for s in s2], dtype=torch.int64, device=dev)
        zero = torch.zeros(nr, dtype=torch.int32, device=dev)
        src = sidx2[:, 0].contiguous()
        mixed, a, b, z, cu = tg.make_inputs([4] * nr, gen)
        o = torch.empty(4 * nr, HV, DV, device=dev, dtype=torch.bfloat16)

        def g_vllm():
            for st in s1:
                ops.fused_gdn_decode_post_conv_mtp(mixed, a, b, P["A_log"], P["dt_bias"], sidx1, cu, acc, st, z,
                                                   P["norm_w"], o, DK ** -0.5, 1e-6, "silu")

        def g_g1():
            for st in s1:
                gd.fused_gdn_decode_post_conv_mtp(mixed, a, b, P["A_log"], P["dt_bias"], sidx1, cu, acc, st, z,
                                                  P["norm_w"], o, DK ** -0.5, 1e-6, "silu")

        def g_verify():
            for li, st in enumerate(s2):
                gr.verify(mixed, a, b, P["A_log"], P["dt_bias"], sidx2, cu, ones, st, z, P["norm_w"], o, corr[li],
                          kn[li], dbuf[li], DK ** -0.5, 1e-6, "silu")

        def g_commit():
            gr.commit(addrs, src, src, zero, zero, acc, corr, kn, dbuf, HV, H, s2[0].stride(0))

        graphs = {}
        for name, fn in (("vllm", g_vllm), ("g1", g_g1), ("g2_verify", g_verify), ("g2_commit", g_commit)):
            fn(); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            for attempt in range(5):
                try:
                    with torch.cuda.graph(g, capture_error_mode="thread_local"):
                        fn()
                    break
                except Exception as e:  # noqa: BLE001
                    print("capture retry", attempt, e, flush=True); g = torch.cuda.CUDAGraph()
            graphs[name] = g
        best = {n: 1e30 for n in graphs}
        for _ in range(rounds):
            for name, g in graphs.items():
                g.replay(); torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ts = []
                for _ in range(iters):
                    e0.record(); g.replay(); e1.record(); e1.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1000 / layers)
                ts.sort(); best[name] = min(best[name], ts[len(ts) // 2])
        rec = {"test": "g2_bench", "nr": nr, "layers": layers, "us_per_layer": {k: round(v, 2) for k, v in best.items()},
               "g2_total": round(best["g2_verify"] + best["g2_commit"], 2)}
        log(rec)
        del s1, s2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    P = tg.real_params(0)
    ok = chained(P, 300, 2, None, 2) and chained(P, 100, 8, None, 2) and chained(P, 50, 3, 128, 1)
    log({"summary": "gdn_g2", "bit_identical_to_g1": ok})
    bench(P)
    print("TEST_GDN_G2_DONE", flush=True)
