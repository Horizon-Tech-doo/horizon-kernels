# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# prefill_rs (K3 on #3257: weights as register operand A from regfrag's Wg) - correctness on the rebased build.
#   1. quantizer: quantize_rs codes/scales == K3's quantize codes/scales, up to the documented in-group permutation
#   2. GEMM bit-exactness vs the current K3 (sm120_nvfp4.linear) and vs prefill_rf: 5 shapes x M, passes 2 and 1,
#      conv=1 (one copy: Wg) and conv=0 (fragment-ordered copy, measurement arm); zero / negative block scales;
#      real checkpoint layers (per-shard and collapsed global scales)
#   3. CUDA graph capture (thread_local) + replay, new input through the same graph
#   gpu_lock.sh rs_check python rs_check.py [--quick]
import json, os, sys, time
import torch
from tilelang.contrib import sm120_nvfp4 as nv
from tilelang.contrib.sm120_nvfp4 import decode_regfrag as rfm
from tilelang.contrib.sm120_nvfp4 import prefill_rs as prs

QUICK = "--quick" in sys.argv
SHAPES = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o": (5120, 6144), "gate_up": (34816, 5120), "down": (5120, 17408)}
MS = (128, 384, 512, 1000, 1024, 4096, 8192) if not QUICK else (128, 1024)
fails = []


def same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def out(row, ok):
    print(json.dumps(row), flush=True)
    if not ok:
        fails.append(row)


# 1. quantizer
def unpack(p):
    u = p.view(torch.uint8)
    c = torch.empty(u.shape[0], u.shape[1] * 2, dtype=torch.uint8, device=u.device)
    c[:, 0::2] = u & 0xF; c[:, 1::2] = u >> 4
    return c


perm = torch.tensor([16 * (p // 16) + prs.perm_rel((p % 16) // 2, p % 2, (p // 16) % 2) for p in range(32)])
for K in (5120, 6144, 17408):
    for M in (128, 256):
        x = (torch.randn(M, K, device="cuda") * torch.rand(M, 1, device="cuda") * 3).to(torch.bfloat16)
        for passes in (2, 1):
            A0, S0, r0 = nv.quantize(x, passes)
            A1, S1, r1 = prs.quantize_rs(x, passes)
            idx = (torch.arange(K, device="cuda") // 32) * 32 + perm.cuda()[torch.arange(K, device="cuda") % 32]
            c0 = unpack(A0.view(M, passes, K // 2).reshape(M * passes, K // 2))
            c1 = unpack(A1.view(M, passes, K // 2).reshape(M * passes, K // 2))
            ok_codes = torch.equal(c1, c0[:, idx])
            # scales: K3 chunk-kmajor (passes, M, K/64) -> row-major; rs tiles (passes, M/128 * K/256 * 128, 4)
            s0 = S0.reshape(passes, M // 128, K // 64, 32, 4).permute(0, 1, 4, 3, 2).reshape(passes, M, K // 64)
            s1 = S1.view(passes, M // 128, K // 256, 128, 4).permute(0, 1, 3, 2, 4).reshape(passes, M, K // 64)
            ok = ok_codes and torch.equal(s0, s1) and torch.equal(r0, r1)
            out({"case": "quantizer", "K": K, "M": M, "passes": passes, "codes_permuted_equal": ok_codes,
                 "scales_equal": bool(torch.equal(s0, s1)), "pass": ok}, ok)


# 2. GEMM
def rand_layer(N, K, seed, zero_frac=0.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    wp = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    ws = (torch.rand(N, K // 16, device="cuda", generator=g) * 2 + 0.5).to(torch.float8_e4m3fn)
    if zero_frac:
        z = torch.rand(N, K // 16, device="cuda", generator=g) < zero_frac
        ws = ws.view(torch.uint8).masked_fill(z, 0).view(torch.float8_e4m3fn)
    return wp, ws, torch.tensor([448.0 * 6], device="cuda")


def check(tag, wp, ws, gs, sizes=None, ms=MS, passes_list=(2, 1), ws_k3=None, wp_k3=None, conv0=True):
    N, K = wp.shape[0], wp.shape[1] * 2
    w0 = nv.prepare_weight(wp if wp_k3 is None else wp_k3, ws if ws_k3 is None else ws_k3, gs, shard_sizes=sizes)
    prep, w1 = prs.prepare_layer_rs(wp, ws, gs, shard_sizes=sizes)
    assert w1.w is prep.Wg and torch.equal(w1.alpha, w0.alpha)
    wf = None
    if conv0:
        wf = prs.NVFP4WeightRS(prs.fragment_order_weights(wp, ws), w1.sw, w1.alpha, N, K, 0)
    g = torch.Generator(device="cuda").manual_seed(N + K)
    for M in ms:
        x = (torch.randn(M, K, device="cuda", generator=g) * torch.rand(M, 1, device="cuda", generator=g) * 3).to(torch.bfloat16)
        for passes in passes_list:
            y0 = nv.linear(x, w0, passes=passes)
            y1 = prs.linear_rs(x, w1, passes=passes)
            row = {"case": tag, "N": N, "K": K, "M": M, "passes": passes, "conv1_bit_identical": same(y0, y1)}
            if not row["conv1_bit_identical"]:
                row["conv1_n_diff"] = int((y0.view(torch.int16) != y1.view(torch.int16)).sum())
            if wf is not None:
                y2 = prs.linear_rs(x, wf, passes=passes)
                row["conv0_bit_identical"] = same(y0, y2)
            ok = row["conv1_bit_identical"] and row.get("conv0_bit_identical", True)
            out(row, ok)
    del w0, w1, prep, wf
    torch.cuda.empty_cache()


t0 = time.time()
for name, (N, K) in SHAPES.items():
    check("random:" + name, *rand_layer(N, K, 1))
if not QUICK:
    for name in ("o", "down"):
        N, K = SHAPES[name]
        check("zero_scales:" + name, *rand_layer(N, K, 2, zero_frac=0.01), ms=(128, 1024))
    N, K = SHAPES["o"]
    wp, ws, gs = rand_layer(N, K, 3)
    neg = torch.rand(N, K // 16, device="cuda") < 0.3
    wsn = (ws.view(torch.uint8) | (neg.to(torch.uint8) << 7)).view(torch.float8_e4m3fn)
    flip = neg.repeat_interleave(8, 1).to(torch.uint8) * 0x88
    check("neg_scales:o", wp, wsn, gs, ms=(128, 1024), ws_k3=ws, wp_k3=wp ^ flip)
    del wp, ws, wsn, neg, flip
    from safetensors import safe_open
    D = "$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors"
    P = "model.language_model.layers."
    REAL = {"qkvz": ["0.linear_attn.in_proj_qkv", "0.linear_attn.in_proj_z"], "o": ["0.linear_attn.out_proj"],
            "gate_up": ["0.mlp.gate_proj", "0.mlp.up_proj"], "down": ["0.mlp.down_proj"],
            "qkv": ["3.self_attn.q_proj", "3.self_attn.k_proj", "3.self_attn.v_proj"], "o_attn": ["3.self_attn.o_proj"]}
    with safe_open(D, "pt", device="cpu") as h:
        for name, parts in REAL.items():
            wps = [h.get_tensor(P + p + ".weight_packed") for p in parts]
            wss = [h.get_tensor(P + p + ".weight_scale") for p in parts]
            gss = [h.get_tensor(P + p + ".weight_global_scale").float().reshape(-1) for p in parts]
            wp = torch.cat(wps).cuda(); ws = torch.cat(wss).cuda(); gs = torch.cat(gss).cuda()
            sizes = [t.shape[0] for t in wps]
            check("real:" + name + ":per_shard", wp, ws, gs, sizes=sizes, passes_list=(2,), conv0=False)
            check("real:" + name + ":collapsed", wp, ws, gs.max().reshape(1), ms=(512, 4096), passes_list=(2,), conv0=False)
            del wp, ws, gs, wps, wss

# 3. CUDA graphs
for name, (N, K) in SHAPES.items():
    wp, ws, gs = rand_layer(N, K, 4)
    w0 = nv.prepare_weight(wp, ws, gs)
    prep, w1 = prs.prepare_layer_rs(wp, ws, gs)
    for M in ((1024, 4096) if not QUICK else (1024,)):
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        ref = nv.linear(x, w0, passes=2)
        prs.linear_rs(x, w1); torch.cuda.synchronize()
        for attempt in range(5):
            try:
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph, capture_error_mode="thread_local"):
                    yg = prs.linear_rs(x, w1)
                break
            except Exception as ex:  # noqa: BLE001
                print(json.dumps({"capture_retry": f"{type(ex).__name__}: {str(ex)[:100]}"}), flush=True)
                torch.cuda.synchronize(); time.sleep(2)
        outs = []
        for r in range(3):
            yg.zero_(); gph.replay(); torch.cuda.synchronize(); outs.append(same(yg, ref))
        x.copy_(torch.randn(M, K, device="cuda").to(torch.bfloat16))
        gph.replay(); ref2 = nv.linear(x, w0, passes=2); torch.cuda.synchronize()
        row = {"case": "graph:" + name, "M": M, "replays_identical": outs, "new_input_identical": same(yg, ref2)}
        out(row, all(outs) and row["new_input_identical"])
        del gph
    del w0, w1, prep, wp, ws
    torch.cuda.empty_cache()
print(json.dumps({"RESULT": "PASS" if not fails else "FAIL", "n_fail": len(fails), "secs": round(time.time() - t0)}))
