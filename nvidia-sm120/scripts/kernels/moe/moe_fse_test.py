# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# MoE steps 1-3 op-level gate + microbenchmark (27-09-2026), nvidia/Qwen3.6-35B-A3B-NVFP4 real layer weights.
# No vLLM server: vLLM 0.29's own kernels (fused_topk, fused_marlin_moe, dense Marlin NVFP4) and ours.
#
# One MoE block (router gate, top-8 routing, 256 routed experts, shared expert with sigmoid gate) computed four ways:
#   stock   exactly as vLLM runs it today: F.linear gate (256) + F.linear shared gate (1) + topk_softmax +
#           Marlin-MoE (256 experts) + shared expert on dense Marlin (gate_up, SiLU*mul, down) + sigmoid * + add
#   fse     step 1 (HORIZON_MOE_FSE=1): one F.linear over the 264-row fused gate (256 + shared + 7 zero rows), torch
#           router (contiguous copy + topk_softmax + sigmoid + cat), Marlin-MoE over 257 experts (9 slots)
#   fse_rt  step 3 (HORIZON_MOE_ROUTE=1): fse with tilelang.contrib.horizon_moe.route (one launch)
#   fse_rtg step 2+3 (HORIZON_MOE_GATE=1): fse_rt with the gate on K5a skinny for M <= skinny.SUPPORTS_M
#
# Checks (printed as JSON lines, also appended to --out):
#   route_bitwise  route kernel vs the torch router: ids equal and weights bit-equal (random, real-gate and tied logits)
#   accuracy       each arm vs an fp32 reference (weights dequantised exactly, fp32 activations, the arm's own
#                  routing): relative L2 error and max |err|; gate: fse* no worse than stock
#   determinism    two eager runs bit-equal; CUDA-graph replay bit-equal to eager (thread_local capture + retry)
#   timing         per-block us per arm (CUDA-graph replay, --layers real layers rotated to defeat L2)
#
#   moe_fse_test.py [--layers 3,7,11,15] [--m 1,2,4,8,16,32,64,128] [--reps 50] [--quick]
import argparse, json, os, time, types

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="3,7,11,15")
ap.add_argument("--m", default="1,2,4,8,16,32,64,128")
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--out", default=os.path.expanduser("$HK_WORK/moe-k/moe_fse_test.jsonl"))
ap.add_argument("--tag", default="moe-fse-27-09-2026")
ap.add_argument("--quick", action="store_true", help="1 layer, M 1,4,32, few reps")
ap.add_argument("--no-timing", action="store_true")
A = ap.parse_args()
if A.quick:
    A.layers, A.m, A.reps = "3", "1,4,32", 5

from safetensors import safe_open  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe  # noqa: E402
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa: E402
    apply_fp4_marlin_linear, prepare_fp4_layer_for_marlin, prepare_nvfp4_moe_layer_for_marlin)
from vllm.scalar_type import scalar_types  # noqa: E402
from tilelang.contrib.horizon_fused import skinny_bf16 as sk  # noqa: E402
from tilelang.contrib.horizon_moe import route  # noqa: E402
from horizon_kernels import moe as hmoe  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size  # noqa: E402
hmoe._install_align_reuse()   # the plugin's hook: Marlin reuses the route kernel's alignment for the same ids tensor

CK = "$HK_DATA/models/nvidia-Qwen3.6-35B-A3B-NVFP4"
E, H, I, TOPK, LD = 256, 2048, 512, 8, 264
DEV = "cuda"
OUT = []


def emit(d):
    d = {"tag": A.tag, **d}
    print(json.dumps(d), flush=True)
    OUT.append(d)


# ------------------------------------------------------------------------------------------------ weights
_IDX = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
_FILES = {}


def get(name):
    fn = _IDX[name]
    f = _FILES.get(fn)
    if f is None:
        f = _FILES[fn] = safe_open(f"{CK}/{fn}", "pt", device="cpu")
    return f.get_tensor(name)


def proj(prefix):
    return get(prefix + ".weight"), get(prefix + ".weight_scale"), get(prefix + ".weight_scale_2").float().reshape(())


def load_layer(L):
    p = f"model.language_model.layers.{L}.mlp"
    ex = []
    for e in list(range(E)) + ["shared"]:
        b = f"{p}.shared_expert" if e == "shared" else f"{p}.experts.{e}"
        ex.append((proj(b + ".gate_proj"), proj(b + ".up_proj"), proj(b + ".down_proj")))
    assert all(torch.equal(g[2], u[2]) for g, u, _ in ex), "gate/up weight_scale_2 not tied"
    gate = get(f"{p}.gate.weight").to(DEV, torch.bfloat16)
    sgate = get(f"{p}.shared_expert_gate.weight").to(DEV, torch.bfloat16)
    return ex, gate, sgate


def marlin_moe_tensors(ex):
    n = len(ex)
    w13 = torch.stack([torch.cat([g[0], u[0]], 0) for g, u, _ in ex]).to(DEV)
    w13s = torch.stack([torch.cat([g[1], u[1]], 0) for g, u, _ in ex]).to(DEV)
    g13 = torch.stack([g[2] for g, _, _ in ex]).to(DEV)
    w2 = torch.stack([d[0] for _, _, d in ex]).to(DEV)
    w2s = torch.stack([d[1] for _, _, d in ex]).to(DEV)
    g2 = torch.stack([d[2] for _, _, d in ex]).to(DEV)
    layer = types.SimpleNamespace(num_experts=n, hidden_size=H, intermediate_size_per_partition=I,
                                  params_dtype=torch.bfloat16)
    w13m, w13sm, gg1, w2m, w2sm, gg2 = prepare_nvfp4_moe_layer_for_marlin(layer, w13, w13s, g13, w2, w2s, g2,
                                                                          is_act_and_mul=True)
    return dict(w1=w13m, w2=w2m, w1_scale=w13sm, w2_scale=w2sm, global_scale1=gg1, global_scale2=gg2,
                workspace=layer.workspace)


def dense_marlin(w, s, s2, n, k):
    m = torch.nn.Module()
    m.weight = torch.nn.Parameter(w.to(DEV), requires_grad=False)
    m.weight_scale = torch.nn.Parameter(s.to(DEV), requires_grad=False)
    m.weight_global_scale = torch.nn.Parameter(s2.reshape(1).to(DEV), requires_grad=False)
    m.output_size_per_partition, m.input_size_per_partition, m.params_dtype = n, k, torch.bfloat16
    prepare_fp4_layer_for_marlin(m)
    return m


def shared_dense(ex):
    (gw, gs, g2), (uw, us, u2), (dw, ds, d2) = ex[E]
    gu = dense_marlin(torch.cat([gw, uw]), torch.cat([gs, us]), torch.maximum(g2, u2), 2 * I, H)
    dn = dense_marlin(dw, ds, d2, H, I)
    return gu, dn


_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def deq(w, s, s2):
    """exact fp32 dequantisation: e2m1 (low nibble = even k) x e4m3 block scale (16) x weight_scale_2"""
    w = w.to(DEV); lut = _LUT.to(DEV)
    v = torch.stack([lut[(w & 15).long()], lut[(w >> 4).long()]], -1).reshape(w.shape[0], -1)
    return v * s.to(DEV).float().repeat_interleave(16, 1) * s2.to(DEV)


class Layer:
    def __init__(self, L):
        t0 = time.time()
        self.L = L
        self.ex, self.gate, self.sgate = load_layer(L)
        self.m256 = marlin_moe_tensors(self.ex[:E])
        self.m257 = marlin_moe_tensors(self.ex)
        self.sgu, self.sdn = shared_dense(self.ex)
        g = torch.zeros(LD, H, device=DEV, dtype=torch.bfloat16)
        g[:E] = self.gate; g[E] = self.sgate[0]
        self.gate264 = g.contiguous()
        self._deq = {}
        print(f"layer {L} ready in {time.time() - t0:.1f}s", flush=True)

    def deq_expert(self, e):
        d = self._deq.get(e)
        if d is None:
            d = self._deq[e] = tuple(deq(*p) for p in self.ex[e])
        return d


def moe(c, x, tw, ti):
    return fused_marlin_moe(hidden_states=x, w1=c["w1"], w2=c["w2"], bias1=None, bias2=None,
                            w1_scale=c["w1_scale"], w2_scale=c["w2_scale"], topk_weights=tw, topk_ids=ti,
                            quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=-1,
                            activation=MoEActivation.SILU, global_scale1=c["global_scale1"],
                            global_scale2=c["global_scale2"], workspace=c["workspace"])


def dlin(m, x):
    return apply_fp4_marlin_linear(x, m.weight, m.weight_scale, m.weight_global_scale, m.workspace,
                                   m.output_size_per_partition, m.input_size_per_partition)


# ------------------------------------------------------------------------------------------------ arms
def router_torch(x, logits):
    w, ids, _ = fused_topk(x, logits[:, :E].contiguous(), TOPK, True, None, "softmax")
    sw = torch.sigmoid(logits[:, E:E + 1]).float()
    return torch.cat([w, sw], 1), torch.cat([ids, torch.full_like(ids[:, :1], E)], 1)


def arm_stock(P, x, want_route=False):
    logits = F.linear(x, P.gate)
    tw, ti, _ = fused_topk(x, logits, TOPK, True, None, "softmax")
    routed = moe(P.m256, x, tw, ti)
    gu = dlin(P.sgu, x)
    h = F.silu(gu[:, :I]) * gu[:, I:]
    s = dlin(P.sdn, h)
    y = routed + torch.sigmoid(F.linear(x, P.sgate)) * s
    return (y, tw, ti) if want_route else y


def arm_fse(P, x, kernel=False, skinny=False, align=False, want_route=False, skinny_max=None):
    if skinny and x.shape[0] <= (skinny_max or sk.SUPPORTS_M):
        logits = sk.linear(x, P.gate264)
    else:
        logits = F.linear(x, P.gate264)
    if align and x.shape[0] <= route.ALIGN_MAX_M:
        bs = hmoe._marlin_block_size(x.shape[0], TOPK + 1, E + 1)
        tw, ti, al = route.route_align(logits, E, TOPK, True, bs)
        hmoe._ALIGN[0] = (ti, bs, E + 1, al)
    elif kernel:
        tw, ti = route.route_shared(logits, E, TOPK, True)
    else:
        tw, ti = router_torch(x, logits)
    y = moe(P.m257, x, tw, ti)
    return (y, tw, ti) if want_route else y


ARMS = {"stock": arm_stock, "fse": arm_fse, "fse_rt": lambda P, x, **k: arm_fse(P, x, kernel=True, **k),
        "fse_rtg": lambda P, x, **k: arm_fse(P, x, kernel=True, skinny=True, **k),
        "fse_rta": lambda P, x, **k: arm_fse(P, x, kernel=True, align=True, **k),
        "fse_rtag": lambda P, x, **k: arm_fse(P, x, kernel=True, skinny=True, align=True, **k),
        "fse_rtag256": lambda P, x, **k: arm_fse(P, x, kernel=True, skinny=True, align=True, skinny_max=256, **k)}


def reference(P, x, tw, ti):
    """fp32 truth for the given routed slots (tw, ti: (M, TOPK)) plus the shared expert gated by the fp32 sigmoid of
    the fp32 gate logit (the same truth for every arm: the arms' own sigmoid rounding counts as their error)."""
    x32 = x.float()
    y = torch.zeros(x.shape[0], H, device=DEV)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            Wg, Wu, Wd = P.deq_expert(int(ti[t, j]))
            h = F.silu(x32[t] @ Wg.T) * (x32[t] @ Wu.T)
            y[t] += float(tw[t, j]) * (h @ Wd.T)
    Wg, Wu, Wd = P.deq_expert(E)
    sg = torch.sigmoid(x32 @ P.sgate.float().T)
    y += sg * ((F.silu(x32 @ Wg.T) * (x32 @ Wu.T)) @ Wd.T)
    return y


# ------------------------------------------------------------------------------------------------ checks
def bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def check_route(P):
    torch.manual_seed(1)
    fails, n = 0, 0
    cases = []
    for M in (1, 2, 3, 4, 5, 8, 17, 32, 33, 64, 128, 255, 256, 1024, 4096):
        x = (torch.randn(M, H, device=DEV) * 0.5).to(torch.bfloat16)
        cases.append(("real_gate", M, F.linear(x, P.gate264)))
        cases.append(("randn", M, (torch.randn(M, LD, device=DEV) * 3).to(torch.bfloat16)))
        tied = (torch.randint(-3, 4, (M, LD), device=DEV).float() * 0.5).to(torch.bfloat16)   # many exact ties
        cases.append(("tied", M, tied))
        cases.append(("big", M, (torch.randn(M, LD, device=DEV) * 40).to(torch.bfloat16)))
    for kind, M, lg in cases:
        w0, i0 = router_torch(lg[:, :0].new_empty(M, H), lg)
        w1, i1 = route.route_shared(lg, E, TOPK, True)
        ok = torch.equal(i0, i1) and bits_equal(w0, w1)
        n += 1
        if not ok:
            fails += 1
            bad = (i0 != i1).any(1) | (w0.view(torch.int32) != w1.view(torch.int32)).any(1)
            emit({"check": "route_bitwise_case_fail", "kind": kind, "M": M, "rows_bad": int(bad.sum()),
                  "ids_diff": int((i0 != i1).sum()), "max_w_diff": float((w0 - w1).abs().max())})
    emit({"check": "route_bitwise", "cases": n, "failed": fails, "pass": fails == 0})
    return fails == 0


def check_align(P):
    """route_align's alignment vs vLLM's moe_align_block_size on the same ids: identical num_tokens_post_padded and
    expert_ids, identical sorted_token_ids per expert segment up to order (both use atomics, so an expert with more
    than one block may split its pairs differently across its blocks); Marlin-MoE output bit-identical."""
    torch.manual_seed(3)
    fails, n = 0, 0
    structure_failures, output_failures = 0, 0
    for M in (1, 2, 3, 4, 7, 8, 16, 31, 32, 64, 100, 128):
        for kind in ("real_gate", "tied"):
            x = (torch.randn(M, H, device=DEV) * 0.5).to(torch.bfloat16)
            lg = F.linear(x, P.gate264) if kind == "real_gate" else \
                (torch.randint(-3, 4, (M, LD), device=DEV).float() * 0.5).to(torch.bfloat16)
            bs = hmoe._marlin_block_size(M, TOPK + 1, E + 1)
            w, ids, (so, eo, no) = route.route_align(lg, E, TOPK, True, bs)
            w2, ids2 = route.route_shared(lg, E, TOPK, True)
            ss, es, ns = moe_align_block_size(ids, bs, E + 1, None, ignore_invalid_experts=True)
            ok = torch.equal(ids, ids2) and bits_equal(w, w2) and int(no) == int(ns) and so.shape == ss.shape \
                and eo.shape == es.shape
            if ok:
                nb = int(ns) // bs
                ok = torch.equal(eo[:nb], es[:nb]) and bool((eo[nb:] == -1).all())
                for e in eo[:nb].unique().tolist():   # per expert segment (its blocks are contiguous): same pair set
                    blk = (eo[:nb] == e).nonzero().flatten()
                    a, b = int(blk[0]) * bs, (int(blk[-1]) + 1) * bs
                    ok &= torch.equal(so[a:b].sort().values, ss[a:b].sort().values)
                ok &= bool((so[int(no):] == M * (TOPK + 1)).all())
            structure_ok = bool(ok)
            output_equal = None
            if structure_ok:
                hmoe._ALIGN[0] = (ids, bs, E + 1, (so, eo, no))
                y1 = moe(P.m257, x, w, ids)          # reuses our alignment
                y0 = moe(P.m257, x, w, ids.clone())  # different tensor object -> vLLM's own align
                output_equal = bits_equal(y0, y1)
                ok = output_equal
            structure_failures += not structure_ok
            output_failures += output_equal is False
            n += 1
            if not ok:
                fails += 1
                emit({"check": "align_case_fail", "kind": kind, "M": M, "bs": bs, "ntp": [int(no), int(ns)],
                      "structure_pass": structure_ok, "expert_output_bitwise": output_equal})
    emit({"check": "align_contract", "cases": n, "failed": fails, "pass": fails == 0,
          "structure_failures": structure_failures, "expert_output_failures": output_failures,
          "reuse_stats": dict(hmoe._ALIGN_STATS)})
    return fails == 0


def check_accuracy(P, Ms):
    torch.manual_seed(2)
    ok_all = True
    for M in Ms:
        x = (torch.randn(M, H, device=DEV) * 0.5).to(torch.bfloat16)
        res = {"check": "accuracy", "layer": P.L, "M": M}
        ids = {}
        for name, fn in ARMS.items():
            y, tw, ti = fn(P, x, want_route=True)
            ids[name] = ti[:, :TOPK]
            ref = reference(P, x, tw[:, :TOPK], ti[:, :TOPK])
            err = (y.float() - ref)
            res[name] = {"rel_l2": float(err.norm() / ref.norm()), "max_abs": float(err.abs().max()),
                         "ref_rms": float(ref.pow(2).mean().sqrt())}
        res["routing_equal_to_stock"] = {k: float((ids[k].sort(1).values == ids["stock"].sort(1).values)
                                                  .all(1).float().mean()) for k in ids if k != "stock"}
        P._deq.clear()
        base = res["stock"]["rel_l2"]
        res["pass"] = all(res[k]["rel_l2"] <= base * 1.10 + 1e-4 for k in ARMS if k != "stock")
        ok_all &= res["pass"]
        emit(res)
    return ok_all


def capture(fn, tries=5):
    for t in range(tries):
        try:
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                out = fn()
            torch.cuda.synchronize()
            return g, out
        except Exception as e:  # noqa: BLE001  (gpushare can break a capture at random: retry)
            print(f"capture retry {t}: {str(e)[:120]}", flush=True)
            torch.cuda.synchronize()
            time.sleep(1)
    raise RuntimeError("CUDA-graph capture failed")


GATED = {"fse_rtag256"}   # FSE + route/align kernel + skinny gate for every decode M (<= 256)


def check_determinism(P, Ms):
    ok_all = True
    for M in Ms:
        x = (torch.randn(M, H, device=DEV) * 0.5).to(torch.bfloat16)
        stock_ok = None
        for name in ARMS:
            fn = ARMS[name]
            a, b = fn(P, x), fn(P, x)
            g, out = capture(lambda: fn(P, x))
            g.replay(); torch.cuda.synchronize()
            ok = bits_equal(a, b) and bits_equal(a, out)
            if name == "stock":
                stock_ok = ok
                emit({"check": "determinism", "layer": P.L, "M": M, "arm": name, "eager_repeat_equal": bits_equal(a, b),
                      "graph_equal_eager": bits_equal(a, out), "pass": ok, "reference_only": True})
                continue
            if name in GATED:   # the candidate configuration; the other arms are reported (cuBLAS gate: see probe)
                ok_all &= ok
            emit({"check": "determinism", "layer": P.L, "M": M, "arm": name, "eager_repeat_equal": bits_equal(a, b),
                  "graph_equal_eager": bits_equal(a, out), "pass": ok})
    return ok_all


def timing(layers, Ms):
    for M in Ms:
        x = (torch.randn(M, H, device=DEV) * 0.5).to(torch.bfloat16)
        row = {"check": "timing", "M": M, "layers_rotated": len(layers)}
        for name, fn in ARMS.items():
            for P in layers:           # eager warm-up
                fn(P, x)
            graphs = [capture(lambda P=P: fn(P, x))[0] for P in layers]
            for g in graphs:
                g.replay()
            torch.cuda.synchronize()
            best = 1e9
            for rnd in range(3):
                ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ev0.record()
                for r in range(A.reps):
                    for g in graphs:
                        g.replay()
                ev1.record(); torch.cuda.synchronize()
                best = min(best, ev0.elapsed_time(ev1) * 1e3 / (A.reps * len(graphs)))
            row[name + "_us"] = round(best, 2)
            del graphs
        row["fse_gain_us"] = round(row["stock_us"] - row["fse_us"], 2)
        best = min(row[k + "_us"] for k in ARMS if k != "stock")
        row["best_arm"] = [k for k in ARMS if k != "stock" and row[k + "_us"] == best][0]
        row["best_gain_us"] = round(row["stock_us"] - best, 2)
        row["x40_layers_ms_saved"] = round(40 * row["best_gain_us"] / 1e3, 3)
        emit(row)


def main():
    with set_current_vllm_config(VllmConfig()):
        Ms = [int(v) for v in A.m.split(",")]
        layers = [Layer(int(L)) for L in A.layers.split(",")]
        P = layers[0]
        # compile/load our kernels once, outside capture
        route.warm(E, LD, TOPK, True, DEV, block_sizes=(8, 16))
        for cfg in sk.all_configs(LD, H):
            for m in (1, cfg[0], sk.SUPPORTS_M):
                sk.linear(torch.zeros(m, H, device=DEV, dtype=torch.bfloat16), P.gate264, cfg=cfg)
        torch.cuda.synchronize()
        r1a = check_route(P)
        r1b = check_align(P)
        r1 = r1a and r1b
        if not r1b:   # a wrong alignment can make Marlin read out of bounds: keep it away from the other checks
            for k in ("fse_rta", "fse_rtag"):
                ARMS.pop(k, None)
        r2 = check_accuracy(P, [m for m in Ms if m <= 32])
        r3 = check_determinism(P, [m for m in Ms if m in (1, 4, 32, 128)] or Ms[:2])
        # Preserve the legacy combined field for existing gate consumers.
        # New fields distinguish routing arithmetic from alignment/output failures.
        emit({"check": "summary", "route_bitwise": r1, "routing_weights_ids_bitwise": r1a,
              "alignment_and_expert_output": r1b, "accuracy": r2, "determinism": r3})
        if not A.no_timing:
            timing(layers, Ms)
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    with open(A.out, "a") as f:
        for d in OUT:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
