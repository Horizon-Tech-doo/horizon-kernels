# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# MoE step 4 race (27-09-2026): our grouped W4A16 NVFP4 MoE (tilelang.contrib.horizon_moe.grouped_w4a16: up + SiLU*up
# epilogue, down + slot-weight epilogue, fixed-order slot sum) vs vLLM's Marlin-MoE on the same 257 experts (routed +
# shared as #256, i.e. what HORIZON_MOE_FSE runs), real layer weights of nvidia/Qwen3.6-35B-A3B-NVFP4.
#   moe_grouped_race.py [--layers 3,7,11,15] [--m 1,2,4,8,16,32,64,128] [--reps 50] [--quick]
# Checks: accuracy vs the fp32 dequantised reference (same routing) for both, determinism (eager repeat, graph replay).
# Timing (CUDA-graph replay, layers rotated): the whole MoE op after routing: Marlin = fused_marlin_moe (align + 2 GEMMs
# + SiLU + sum); ours = up + down + moe_sum on the route kernel's 16-row alignment (the alignment itself is part of the
# route launch in both our step-3 path and here, so it is excluded; Marlin's own align is included in its time, as it
# is in vLLM). Config sweep per M for ours; KILL CRITERION (docs/kernel-master-plan §5.3): >= 1.1x Marlin at M = 1-4.
import argparse, itertools, json, os, sys, time

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="3,7,11,15")
ap.add_argument("--m", default="1,2,4,8,16,32,64,128")
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default=os.path.expanduser("$HK_WORK/moe-k/moe_grouped_race.jsonl"))
A = ap.parse_args()
if A.quick:
    A.layers, A.m, A.reps = "3", "1,4,32", 5
sys.argv = [sys.argv[0], "--layers", A.layers, "--no-timing"]
import moe_fse_test as t  # noqa: E402  (layer loading, Marlin tensors, fp32 reference, capture helper)
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
import vllm._custom_ops as ops  # noqa: E402
from tilelang.contrib.horizon_moe import grouped_w4a16 as gw, route  # noqa: E402

E1, H, I, TOPK1 = t.E + 1, t.H, t.I, t.TOPK + 1
OUT = []


def emit(d):
    d = {"tag": "moe-grouped-race", **d}
    print(json.dumps(d), flush=True)
    OUT.append(d)


def ours_weights(P):
    ex = P.ex
    w13 = torch.stack([torch.cat([g[0], u[0]], 0) for g, u, _ in ex]).to("cuda")
    s13 = torch.stack([torch.cat([g[1], u[1]], 0) for g, u, _ in ex]).to("cuda")
    a13 = torch.stack([g[2] for g, _, _ in ex]).to("cuda")
    w2 = torch.stack([d[0] for _, _, d in ex]).to("cuda")
    s2 = torch.stack([d[1] for _, _, d in ex]).to("cuda")
    a2 = torch.stack([d[2] for _, _, d in ex]).to("cuda")
    W13 = gw.prepare_experts(w13, s13, a13, interleave=True)
    W2 = gw.prepare_experts(w2, s2, a2)
    del w13, s13, w2, s2
    return W13, W2


def routing(P, x, bs):
    lg = F.linear(x, P.gate264)
    if x.shape[0] <= route.ALIGN_MAX_M:
        w, ids, al = route.route_align(lg, t.E, t.TOPK, True, bs)
    else:
        w, ids = route.route_shared(lg, t.E, t.TOPK, True)
        al = t.moe_align_block_size(ids, bs, E1, None, ignore_invalid_experts=True)
    return w, ids, al


def ours(P, x, w, al, cu=None, cd=None):
    out = torch.empty_like(x)
    return gw.moe_forward(x, w, al, P.W13, P.W2, cfg_up=cu, cfg_down=cd,
                          moe_sum=lambda y, o: ops.moe_sum(y, o))


def marlin(P, x, w, ids):
    return t.moe(P.m257, x, w, ids)


def main():
    with set_current_vllm_config(VllmConfig()):
        layers = [t.Layer(int(L)) for L in A.layers.split(",")]
        for P in layers:
            t0 = time.time()
            P.W13, P.W2 = ours_weights(P)
            print(f"layer {P.L}: grouped weights prepared in {time.time() - t0:.1f}s", flush=True)
        P = layers[0]
        route.warm(t.E, t.LD, t.TOPK, True, "cuda", block_sizes=(8, 16))
        Ms = [int(v) for v in A.m.split(",")]
        torch.manual_seed(7)
        # ---- accuracy + determinism
        ok_all = True
        for M in Ms:
            x = (torch.randn(M, H, device="cuda") * 0.5).to(torch.bfloat16)
            w, ids, al = routing(P, x, 16)
            yo = ours(P, x, w, al)
            ym = marlin(P, x, w, ids)
            row = {"check": "accuracy", "M": M}
            if M <= 32:
                ref = t.reference(P, x, w[:, :t.TOPK], ids[:, :t.TOPK])
                P._deq.clear()
                for name, y in (("ours", yo), ("marlin", ym)):
                    err = y.float() - ref
                    row[name] = {"rel_l2": float(err.norm() / ref.norm()), "max_abs": float(err.abs().max())}
                row["pass"] = row["ours"]["rel_l2"] <= row["marlin"]["rel_l2"] * 1.10 + 1e-4
                ok_all &= row["pass"]
            d = (yo.float() - ym.float())
            row["ours_vs_marlin_rel_l2"] = float(d.norm() / ym.float().norm())
            y2 = ours(P, x, w, al)
            g, yg = t.capture(lambda: ours(P, x, w, al))
            g.replay(); torch.cuda.synchronize()
            row["deterministic"] = t.bits_equal(yo, y2) and t.bits_equal(yo, yg)
            ok_all &= row["deterministic"]
            emit(row)
        emit({"check": "summary", "accuracy_and_determinism": ok_all})
        # ---- timing
        for M in Ms:
            x = (torch.randn(M, H, device="cuda") * 0.5).to(torch.bfloat16)
            rt = [routing(Q, x, 16) for Q in layers]
            rm = [(w, ids) for (w, ids, _) in rt]

            def bench(fn):
                for i, Q in enumerate(layers):
                    fn(i, Q)
                gs = [t.capture(lambda i=i, Q=Q: fn(i, Q))[0] for i, Q in enumerate(layers)]
                for gg in gs:
                    gg.replay()
                torch.cuda.synchronize()
                best = 1e9
                for _ in range(3):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(A.reps):
                        for gg in gs:
                            gg.replay()
                    e1.record(); torch.cuda.synchronize()
                    best = min(best, e0.elapsed_time(e1) * 1e3 / (A.reps * len(gs)))
                return round(best, 2)

            tm = bench(lambda i, Q: marlin(Q, x, rm[i][0], rm[i][1]))
            res = []
            k_up, k_dn = 2048 // 64, 512 // 64
            for nw_u, sk_u, st_u in itertools.product((4, 8), (1, 2, 4, 8), (3, 4)):
                if k_up % sk_u:
                    continue
                for sk_d in (1, 2, 4):
                    cu, cd = (nw_u, 64, sk_u, st_u), (4, 64, sk_d, st_u)
                    try:
                        us = bench(lambda i, Q: ours(Q, x, rt[i][0], rt[i][2], cu, cd))
                    except Exception as e:  # noqa: BLE001
                        emit({"check": "cfg_error", "M": M, "cfg_up": cu, "cfg_down": cd, "err": str(e)[:160]})
                        continue
                    res.append((us, cu, cd))
            res.sort()
            best = res[0] if res else (None, None, None)
            emit({"check": "timing", "M": M, "marlin_fse_us": tm, "ours_best_us": best[0], "cfg_up": best[1],
                  "cfg_down": best[2], "speedup": round(tm / best[0], 3) if best[0] else None,
                  "top3": [(u, c1, c2) for u, c1, c2 in res[:3]], "default_cfg_us": next(
                      (u for u, c1, c2 in res if c1 == gw.config("up", 1024, 2048, M) and c2 == gw.config("down", 2048, 512, M)), None)})
        sp = {r["M"]: r["speedup"] for r in OUT if r.get("check") == "timing"}
        small = [sp[m] for m in (1, 2, 4) if m in sp and sp[m]]
        emit({"check": "kill_criterion", "speedup_m1_4": small, "go": bool(small) and min(small) >= 1.1})
    with open(A.out, "a") as f:
        for d in OUT:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
