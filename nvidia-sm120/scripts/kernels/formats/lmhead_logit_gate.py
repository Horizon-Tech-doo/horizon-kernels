# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# S1 (formats track, 27-09-2026): offline logit gate for a quantized lm_head on OUR checkpoint.
# Real hidden states (the lm_head's input, post final norm) go through the BF16 lm_head (reference) and through
# candidate quantized heads; reports top-1 agreement, the reference logit gap at every disagreement (nats), KL and
# weight error. No vLLM, no production files touched; runs on CPU (default) or GPU.
#   python lmhead_logit_gate.py H.pt [H2.pt ...]  (env: DEVICE=cpu|cuda, THREADS=32, SAVE_DIR=..., VARIANTS=...)
# Candidates:
#   nv_nvfp4    NVIDIA's NVFP4 lm_head tensors (nvidia/Qwen3.8-27B-NVFP4, ModelOpt, calibrated from the same BF16 model)
#   rtn_nvfp4   our NVFP4, round-to-nearest, block scale amax/6 (llm-compressor style)
#   mse_nvfp4   our NVFP4, per-block scale search (min block MSE over 16 scale factors)
#   hmse_nvfp4  our NVFP4, per-block scale search weighted by diag E[h^2] from the calibration half of the states
#   fp8_row     FP8 e4m3 per output row (amax/448)
# The gate (kernel-master-plan-27-09-2026 (not in this release) S1): >= 99% top-1 agreement with the BF16 reference and every
# disagreement a near-tie (< 0.1 nat).
import json
import os
import sys
import time

import torch
from safetensors import safe_open

OURS = "$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors"
NV = "$HK_DATA/models/nvidia-Qwen3.8-27B-NVFP4/model-00003-of-00003.safetensors"
DEV = os.environ.get("DEVICE", "cpu")
torch.set_num_threads(int(os.environ.get("THREADS", "32")))
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
ROWS = 8192   # rows per quantization chunk (memory)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_h(paths):
    hs = []
    for p in paths:
        d = torch.load(p, map_location="cpu")
        if isinstance(d, dict):
            d = d.get("h", next(iter(d.values())))
        hs.append(d.reshape(-1, d.shape[-1]).to(torch.bfloat16))
    h = torch.cat(hs)
    h = torch.unique(h.float(), dim=0).to(torch.bfloat16)   # the K1a capture repeats rows
    cap = int(os.environ.get("MAXROWS", "0"))
    if cap and h.shape[0] > cap:                             # random subset (memory: logits are T x 248320 fp32)
        h = h[torch.randperm(h.shape[0], generator=torch.Generator().manual_seed(1))[:cap]]
    return h


def e2m1_round(x):
    """Round |x| (already divided by the block scale, clipped to 6) to the nearest e2m1 magnitude; ties to even code."""
    a = x.abs().clamp(max=6.0)
    mids = (POS[1:] + POS[:-1]) / 2                       # 0.25 0.75 1.25 1.75 2.5 3.5 5
    idx = torch.bucketize(a, mids.to(a.device), right=False)
    # bucketize(right=False): a == mid goes to the lower bucket; e2m1 ties-to-even: 0.25->0, 0.75->1.0(code2),
    # 1.25->1.0, 1.75->2.0, 2.5->2.0, 3.5->4.0, 5->4.0. Codes 0..7; even code wins on a tie.
    tie = (a.unsqueeze(-1) == mids.to(a.device)).any(-1)
    up = tie & (idx % 2 == 1)                             # lower code odd -> round up to the even code
    idx = idx + up.long()
    return POS.to(a.device)[idx] * torch.sign(x)


def quant_nvfp4(W, mode, hdiag=None):
    """W (N, K) bf16 -> dequantised fp32 (N, K) of an NVFP4 quantisation (block 16, e4m3 scales, fp32 global)."""
    N, K = W.shape
    amax_t = W.float().abs().max()
    gs = amax_t / (6.0 * 448.0)                           # global multiplier: block scales fit e4m3
    out = torch.empty(N, K, dtype=torch.float32)
    facs = torch.linspace(1.0, 0.70, 16) if mode != "rtn" else torch.tensor([1.0])
    for r0 in range(0, N, ROWS):
        w = W[r0:r0 + ROWS].to(DEV).float().reshape(-1, K // 16, 16)
        bmax = w.abs().amax(-1, keepdim=True)
        best, best_err = None, None
        for f in facs:
            s = (bmax / 6.0 * f / gs).clamp(min=2 ** -9, max=448.0).to(torch.float8_e4m3fn).float() * gs
            s = torch.where(s == 0, torch.ones_like(s), s)
            q = e2m1_round(w / s) * s
            if len(facs) == 1:
                best = q
                break
            d = (q - w) ** 2
            if hdiag is not None:
                d = d * hdiag.to(DEV).reshape(1, K // 16, 16)
            err = d.sum(-1, keepdim=True)
            if best is None:
                best, best_err = q, err
            else:
                m = err < best_err
                best = torch.where(m, q, best)
                best_err = torch.where(m, err, best_err)
        out[r0:r0 + ROWS] = best.reshape(-1, K).cpu()
    return out


def dequant_nv():
    with safe_open(NV, "pt", device="cpu") as f:
        wp = f.get_tensor("lm_head.weight")               # uint8 (N, K/2), low nibble = even k
        ws = f.get_tensor("lm_head.weight_scale")         # fp8 e4m3 (N, K/16)
        a = f.get_tensor("lm_head.weight_scale_2").float().reshape(-1)[0]
    N, K2 = wp.shape
    out = torch.empty(N, K2 * 2, dtype=torch.float32)
    for r0 in range(0, N, ROWS):
        p = wp[r0:r0 + ROWS].to(DEV)
        lo, hi = (p & 0xF).long(), (p >> 4).long()
        v = torch.stack([E2M1.to(DEV)[lo], E2M1.to(DEV)[hi]], -1).reshape(p.shape[0], -1)
        s = ws[r0:r0 + ROWS].to(DEV).float().repeat_interleave(16, dim=1)
        out[r0:r0 + ROWS] = (v * s * a).cpu()
    return out, float(a)


def quant_fp8_row(W):
    out = torch.empty(W.shape, dtype=torch.float32)
    for r0 in range(0, W.shape[0], ROWS):
        w = W[r0:r0 + ROWS].to(DEV).float()
        s = w.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
        out[r0:r0 + ROWS] = ((w / s).to(torch.float8_e4m3fn).float() * s).cpu()
    return out


def logits(h, W, chunk=32768):
    hd = h.to(DEV).float()
    return torch.cat([(hd @ W[i:i + chunk].to(DEV).float().T).cpu() for i in range(0, W.shape[0], chunk)], 1)


def compare(ref, cand, name, wref, wcand):
    """ref / cand: fp32 logits (T, V). Also the bf16-rounded versions (what the served kernels emit)."""
    r1, c1 = ref.argmax(1), cand.argmax(1)
    cb1 = cand.to(torch.bfloat16).argmax(1)
    rb1 = ref.to(torch.bfloat16).argmax(1)
    dis = (r1 != c1).nonzero().flatten()
    gaps = (ref[dis, r1[dis]] - ref[dis, c1[dis]]) if dis.numel() else torch.zeros(0)
    lr, lc = torch.log_softmax(ref, 1), torch.log_softmax(cand, 1)
    kl = (lr.exp() * (lr - lc)).sum(1)
    dtop = (lr.gather(1, r1[:, None]) - lc.gather(1, r1[:, None])).abs().flatten()
    rel = ((wcand - wref).norm() / wref.norm()).item() if wcand is not None else None
    res = {"variant": name, "T": ref.shape[0],
           "top1_vs_fp32": round((r1 == c1).float().mean().item() * 100, 3),
           "top1_bf16out_vs_bf16ref": round((rb1 == cb1).float().mean().item() * 100, 3),
           "n_disagree": int(dis.numel()),
           "gap_max_nat": round(gaps.max().item(), 4) if dis.numel() else 0.0,
           "gap_median_nat": round(gaps.median().item(), 4) if dis.numel() else 0.0,
           "n_gap_ge_0.1": int((gaps >= 0.1).sum()),
           "kl_mean": float(f"{kl.mean().item():.3e}"), "kl_p99": float(f"{kl.quantile(0.99).item():.3e}"),
           "dlogp_top1_mean": round(dtop.mean().item(), 5), "dlogp_top1_max": round(dtop.max().item(), 4),
           "w_rel_err": round(rel, 5) if rel is not None else None}
    res["gate"] = "PASS" if (res["top1_vs_fp32"] >= 99.0 and res["n_gap_ge_0.1"] == 0) else "FAIL"
    return res


def main():
    h = load_h(sys.argv[1:])
    log("hidden states", tuple(h.shape), "norm median", h.float().norm(dim=1).median().item())
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(h.shape[0], generator=g)
    calib, ev = h[perm[: h.shape[0] // 2]], h[perm[h.shape[0] // 2:]]
    hdiag = (calib.float() ** 2).mean(0) + 1e-6
    with safe_open(OURS, "pt", device="cpu") as f:
        W = f.get_tensor("lm_head.weight")
    log("BF16 lm_head", tuple(W.shape), W.dtype)
    Wf = W.float()
    ref = logits(ev, W)
    # floor: vLLM's own BF16 logits (fp32 accumulate, bf16 output) against the fp32 reference
    floor = compare(ref, ref.to(torch.bfloat16).float(), "bf16_logits_floor", None, None)
    print(json.dumps(floor), flush=True)
    results = [floor]
    want = os.environ.get("VARIANTS", "nv_nvfp4,rtn_nvfp4,mse_nvfp4,hmse_nvfp4,fp8_row").split(",")
    save_dir = os.environ.get("SAVE_DIR")
    for name in want:
        t0 = time.time()
        if name == "nv_nvfp4":
            Wq, a = dequant_nv()
        elif name == "rtn_nvfp4":
            Wq = quant_nvfp4(W, "rtn")
        elif name == "mse_nvfp4":
            Wq = quant_nvfp4(W, "mse")
        elif name == "hmse_nvfp4":
            Wq = quant_nvfp4(W, "hmse", hdiag)
        elif name == "fp8_row":
            Wq = quant_fp8_row(W)
        else:
            continue
        log(name, "quantised in %.0f s" % (time.time() - t0))
        res = compare(ref, logits(ev, Wq), name, Wf, Wq)
        res["seconds"] = round(time.time() - t0)
        print(json.dumps(res), flush=True)
        results.append(res)
        del Wq
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "lmhead_logit_gate_%s.json" % time.strftime("%H%M")), "w") as fo:
            json.dump({"h_files": sys.argv[1:], "T_eval": ev.shape[0], "results": results}, fo, indent=1)


if __name__ == "__main__":
    main()
