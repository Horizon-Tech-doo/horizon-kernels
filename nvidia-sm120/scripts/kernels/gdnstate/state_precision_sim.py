# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# GDN recurrent-state precision study (CPU only, numpy; 29-09-2026).
#
# Question: what happens to the GDN (gated delta rule) layer outputs of Qwen3.8-27B if the recurrent state that is
# carried between decode steps is stored in BF16 (or FP16) instead of FP32? The arithmetic inside a step stays FP32
# (as in vLLM's CUDA kernel: the state is loaded, widened to fp32, updated for the step's tokens, rounded on store).
#
# Recurrence (vLLM 0.29 fused_gdn_decode_post_conv_mtp, per value head, state h: 128 value rows x 128 key columns):
#   q = l2norm(q) * 128^-0.5, k = l2norm(k), g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
#   per token:  h *= g;  delta = (v - h k) * beta;  h += delta k^T;  o = h q
# Storage policies compared (rounding of h only where the state leaves the registers):
#   fp32          reference (what production does)
#   bf16_rne      round-to-nearest-even after every decode STEP (the source state of the next step): vLLM's own
#                 --mamba-ssm-cache-dtype bfloat16 path (its kernel stores __floats2bfloat162_rn)
#   bf16_sr       stochastic rounding after every step (unbiased; what our kernel can do)
#   fp16_rne      IEEE half, round-to-nearest after every step
#   bf16_rne_r16  bf16 checkpoint rounded only every 16 tokens (G3 replay ring: the per-token updates are kept exactly
#                 and folded into the checkpoint every 16 accepted tokens)
#   bf16_rne_8k   bf16 rounding every 8192 tokens (what a prefill with 8k chunks does; decode excluded)
# Decode steps: tokens are consumed in steps of 1..T accepted tokens (T = 1 + drafts, acceptance drawn per position).
#
# Inputs are synthetic but use the real per-head gating parameters (A_log, dt_bias) and the real scale of a / b
# (row norms of in_proj_a / in_proj_b times the input RMSNorm weight) of every GDN layer (extract_gdn_params.py).
# q / k are drawn from a finite set of 512 bf16 prototypes + noise (keys recur like tokens do), v ~ N(0, 1) per
# channel with a per-channel scale. This is NOT real activation data: the GPU gate with real text decides.
#
# Output: JSON lines, relative error of the pre-norm output o (per token and head, ||o - o_ref|| / ||o_ref|| over the
# 128 values) summarised per position window, and the same for the gated-RMSNorm direction; plus the relative error
# that storing o itself in bf16 already introduces (vLLM writes o as bf16), as the yardstick.
import argparse
import json
import math
import sys
import time

import numpy as np

DK = DV = 128


def bf16_rne(x):
    b = x.astype(np.float32).view(np.uint32)
    b = (b + (((b >> 16) & 1) + 0x7FFF)) & 0xFFFF0000
    return b.view(np.float32)


def bf16_sr(x, rng):
    b = x.astype(np.float32).view(np.uint32)
    r = rng.integers(0, 1 << 16, size=b.shape, dtype=np.uint32)
    b = (b + r) & 0xFFFF0000            # add uniform [0, 2^16) below the kept bits, truncate: unbiased
    return b.view(np.float32)


def fp16_rne(x):
    return x.astype(np.float16).astype(np.float32)


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def l2n(x):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)


def make_inputs(rng, N, HV, H, a_norm, b_norm):
    G = HV // H
    P = 512
    proto_q = rng.standard_normal((P, H, DK)).astype(np.float32)
    proto_k = rng.standard_normal((P, H, DK)).astype(np.float32)
    # token stream with Zipf-like reuse of prototypes
    ids = np.minimum(rng.zipf(1.3, size=N) - 1, P - 1)
    q = bf16_rne(proto_q[ids] + 0.5 * rng.standard_normal((N, H, DK)).astype(np.float32))
    k = bf16_rne(proto_k[ids] + 0.5 * rng.standard_normal((N, H, DK)).astype(np.float32))
    vscale = np.exp(0.5 * rng.standard_normal((HV, DV))).astype(np.float32)
    v = bf16_rne(rng.standard_normal((N, HV, DV)).astype(np.float32) * vscale)
    a = bf16_rne(rng.standard_normal((N, HV)).astype(np.float32) * a_norm[None, :])
    b = bf16_rne(rng.standard_normal((N, HV)).astype(np.float32) * b_norm[None, :])
    return q, k, v, a, b


def run(q, k, v, a, b, A_log, dt_bias, steps, policy, rng_sr, dtype=np.float32):
    """steps: list of token counts per decode step (sum = N). Returns pre-norm outputs (N, HV, DV)."""
    N, H, _ = q.shape
    HV = v.shape[1]
    G = HV // H
    qn = (l2n(q.astype(dtype)) * dtype(DK ** -0.5)).astype(dtype)
    kn = l2n(k.astype(dtype)).astype(dtype)
    g = np.exp(-np.exp(A_log.astype(dtype))[None, :] * softplus(a.astype(dtype) + dt_bias.astype(dtype)[None, :]))
    g = g.astype(dtype)
    beta = (1.0 / (1.0 + np.exp(-b.astype(dtype)))).astype(dtype)
    h = np.zeros((HV, DV, DK), dtype)
    out = np.empty((N, HV, DV), np.float32)
    t = 0
    since = 0
    for n in steps:
        for _ in range(n):
            kk = np.repeat(kn[t], G, axis=0)         # (HV, DK)
            qq = np.repeat(qn[t], G, axis=0)
            h *= g[t][:, None, None]
            hk = np.einsum("hvk,hk->hv", h, kk)
            delta = (v[t].astype(dtype) - hk) * beta[t][:, None]
            h += delta[:, :, None] * kk[:, None, :]
            out[t] = np.einsum("hvk,hk->hv", h, qq)
            t += 1
        since += n
        if policy == "bf16_rne":
            h = bf16_rne(h)
        elif policy == "bf16_sr":
            h = bf16_sr(h, rng_sr)
        elif policy == "fp16_rne":
            h = fp16_rne(h)
        elif policy == "bf16_rne_r16":
            if since >= 16:
                h = bf16_rne(h)
                since = 0
        elif policy == "bf16_rne_8k":
            if since >= 8192:
                h = bf16_rne(h)
                since = 0
        elif policy in ("fp32", "fp64"):
            pass
        else:
            raise ValueError(policy)
    return out, h


def decode_steps(rng, N, T, p_accept):
    """1 + accepted drafts per step: position i (1..T-1) accepted with prob p_accept[i-1] if all before were."""
    steps = []
    left = N
    while left > 0:
        n = 1
        for i in range(T - 1):
            if rng.random() < p_accept[i]:
                n += 1
            else:
                break
        n = min(n, left)
        steps.append(n)
        left -= n
    return steps


def rel_err(x, ref):
    num = np.sqrt(((x - ref) ** 2).sum(-1))
    den = np.sqrt((ref ** 2).sum(-1)) + 1e-30
    return num / den


def rms_dir(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)


def summarize(err, windows):
    rows = {}
    for lo, hi in windows:
        if lo >= err.shape[0]:
            break
        e = err[lo:hi].ravel()
        rows[f"{lo}-{hi}"] = {"median": float(np.median(e)), "p99": float(np.quantile(e, 0.99)),
                              "max": float(e.max())}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--layers", default="0,1,24,45,62")
    ap.add_argument("--N", type=int, default=32768)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--policies", default="fp64,bf16_rne,bf16_sr,fp16_rne,bf16_rne_r16,bf16_rne_8k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heads", type=int, default=48, help="value heads simulated per layer (multiple of 3)")
    args = ap.parse_args()
    d = np.load(args.params)
    layers = [int(x) for x in args.layers.split(",")]
    windows = [(0, 1024), (1024, 4096), (4096, 16384), (16384, 32768), (32768, 65536), (65536, 131072),
               (131072, 262144)]
    p_acc = [0.95, 0.88, 0.77, 0.69, 0.58][: args.T - 1] if args.T > 1 else []
    for L in layers:
        HV = args.heads
        H = HV // 3
        A_log = d[f"A_log_{L}"][:HV]
        dt_bias = d[f"dt_bias_{L}"][:HV]
        rng = np.random.default_rng(args.seed + 1000 * L)
        q, k, v, a, b = make_inputs(rng, args.N, HV, H, d[f"a_norm_{L}"][:HV], d[f"b_norm_{L}"][:HV])
        steps = decode_steps(rng, args.N, args.T, p_acc)
        t0 = time.time()
        ref, href = run(q, k, v, a, b, A_log, dt_bias, steps, "fp32", None)
        yard = rel_err(bf16_rne(ref), ref)       # the bf16 rounding of o that vLLM already applies
        g0 = np.exp(-np.exp(A_log) * softplus(dt_bias))
        base = {"layer": L, "N": args.N, "T": args.T, "steps": len(steps), "mean_tokens_per_step":
                args.N / len(steps), "slow_heads_1mg_lt_2e-3": int((1 - g0 < 2e-3).sum()), "heads": HV}
        print(json.dumps({**base, "policy": "yardstick_bf16_output", "err_o": summarize(yard, windows)}),
              flush=True)
        for pol in args.policies.split(","):
            if pol == "fp64":
                x, hx = run(q, k, v, a, b, A_log, dt_bias, steps, "fp64", None, dtype=np.float64)
            else:
                x, hx = run(q, k, v, a, b, A_log, dt_bias, steps, pol, np.random.default_rng(7))
            e = rel_err(x, ref)
            ed = rel_err(rms_dir(x), rms_dir(ref))
            es = float(np.sqrt(((hx - href) ** 2).sum()) / (np.sqrt((href ** 2).sum()) + 1e-30))
            # per-head worst window, to expose slow-decay heads
            per_head_late = [float(np.median(e[-2048:, hh])) for hh in range(HV)]
            worst = int(np.argmax(per_head_late))
            print(json.dumps({**base, "policy": pol, "err_o": summarize(e, windows),
                              "err_dir": summarize(ed, windows), "final_state_rel_err": es,
                              "worst_head_late": {"head": worst, "median_err_last2k": per_head_late[worst],
                                                  "one_minus_g0": float(1 - g0[worst])},
                              "max_abs_state": float(np.abs(hx).max()),
                              "secs": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
