# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Offline (CPU) proof and sizing for the exact-argmax lm_head screen (lm-head-screening (not in this release)). No GPU, no vLLM.
#
# Real inputs: the served model's BF16 lm_head (248,320 x 5,120) and real final hidden states captured on a
# benchmark arm (S3 L0, 27-09-2026, HORIZON_DUMP_LMHEAD_H: ~65k rows). For every sampled row and every token it computes the
# reference fp32 logit (CPU summation order) and its BF16 rounding, the screen logits of each candidate copy, and the
# rigorous bound of horizon_kernels/lmhead_screen.py. It checks, per variant:
#   bound_ok      |screen - reference fp32| <= B for every (row, token)       (must hold; the proof's premise)
#   contains_k    the exact BF16 top-k set (all ties at the k-th value) is inside the selection, every row
#   argmax_exact  first-index argmax of the sparse logits == that of the full BF16 logits, every row (k = 1)
# and sizes the recheck: selected tokens per row (percentiles, max, tail fractions) for k = 1 and k = 20, with the
# exact threshold and with the kernels' per-tile threshold (tile 128).
# Variants: fp8_row (W8A16: FP8 e4m3 weights per row, BF16 activations; the decode_w8a16 kernel), fp8_w8a8 (FP8
# activations too: FP8 tensor-core throughput at 65-128 rows; bound adds ||x - x^|| ||w^||), nvfp4_rtn (W4A16, the
# regfrag_xl kernel; 0.72 GB per read).
#   env THREADS (32), ROWS (2048), OUT (json path), H_GLOB, VARIANTS
import glob
import json
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.set_num_threads(int(os.environ.get("THREADS", "32")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo server/
from safetensors import safe_open  # noqa: E402

from horizon_kernels import lmhead_screen as ls  # noqa: E402

MODEL = "$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors"
H_GLOB = os.environ.get("H_GLOB", "$HK_WORK/formats/s1/mainh_*.pt")
NROWS = int(os.environ.get("ROWS", "2048"))
OUT = os.environ.get("OUT", "cpu_screen_study.json")
VARIANTS = os.environ.get("VARIANTS", "fp8_row,fp8_w8a8,nvfp4_rtn").split(",")
KS = [int(k) for k in os.environ.get("KS", "1,20").split(",")]
CH = 8192
POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def e2m1_round(x):
    a = x.abs().clamp(max=6.0)
    mids = (POS[1:] + POS[:-1]) / 2
    idx = torch.bucketize(a, mids, right=False)
    tie = (a.unsqueeze(-1) == mids).any(-1)
    idx = idx + (tie & (idx % 2 == 1)).long()
    return POS[idx] * torch.sign(x)


def nvfp4_rtn(w, gs):
    """(n, K) float -> dequantised NVFP4 (block 16, e4m3 block scale amax/6, fp32 global multiplier gs)."""
    n, K = w.shape
    wb = w.reshape(n, K // 16, 16)
    s = (wb.abs().amax(-1, keepdim=True) / 6.0 / gs).clamp(min=2 ** -9, max=448.0).to(torch.float8_e4m3fn).float() * gs
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (e2m1_round(wb / s) * s).reshape(n, K)


def load_rows():
    files = sorted(glob.glob(H_GLOB))
    hs = [torch.load(f, map_location="cpu").reshape(-1, ls.K_HIDDEN).to(torch.bfloat16) for f in files]
    h = torch.cat(hs)
    h = h[h.float().abs().amax(-1) > 0]
    h = torch.unique(h.float(), dim=0).to(torch.bfloat16)
    g = torch.Generator().manual_seed(1)
    h = h[torch.randperm(h.shape[0], generator=g)[:NROWS]]
    return h, len(files)


def pct(t, qs=(50, 90, 99, 99.9)):
    t = t.double().sort().values
    return {f"p{q}": float(t[min(t.numel() - 1, int(math.ceil(q / 100 * t.numel())) - 1)]) for q in qs}


def main():
    t0 = time.time()
    x, nfiles = load_rows()
    M, K = x.shape
    log(f"rows {M} from {nfiles} files")
    with safe_open(MODEL, "pt", device="cpu") as f:
        W = f.get_tensor("lm_head.weight")
    V = W.shape[0]
    log(f"lm_head {tuple(W.shape)} {W.dtype}")
    xf = x.float()
    xn = ls.row_norms(x)                                        # (M,)
    xq, xs = ls.quant_fp8_rows(x)                               # W8A8 activations (per row)
    xhat = ls.dequant_fp8_rows(xq, xs)
    xerr = (xf.double() - xhat.double()).norm(dim=1).float() * ls.XNORM_SLACK
    xhn = xhat.norm(dim=1) * ls.XNORM_SLACK
    gs_nv = W.float().abs().max() / (6.0 * 448.0)

    ref = torch.empty(M, V, dtype=torch.float32)
    scr = {v: torch.empty(M, V, dtype=torch.bfloat16) for v in VARIANTS}
    P = {v: torch.empty(V, dtype=torch.float32) for v in VARIANTS}
    Pw = torch.empty(V, dtype=torch.float32)                    # ||w^|| (W8A8 activation-error term)
    Wn = torch.empty(V, dtype=torch.float32)
    for c0 in range(0, V, CH):
        w = W[c0:c0 + CH]
        wf = w.float()
        ref[:, c0:c0 + CH] = xf @ wf.t()
        Wn[c0:c0 + CH] = wf.norm(dim=1)
        for v in VARIANTS:
            if v in ("fp8_row", "fp8_w8a8"):
                q, sc = ls.quant_fp8_rows(w)
                wh = ls.dequant_fp8_rows(q, sc)
                if v == "fp8_row":
                    scr[v][:, c0:c0 + CH] = ((xf @ q.float().t()) * sc[None, :]).to(torch.bfloat16)
                else:
                    scr[v][:, c0:c0 + CH] = ((xq.float() @ q.float().t()) * xs[:, None] * sc[None, :]).to(torch.bfloat16)
                    Pw[c0:c0 + CH] = wh.norm(dim=1)
            else:
                wh = nvfp4_rtn(wf, gs_nv)
                scr[v][:, c0:c0 + CH] = (xf @ wh.t()).to(torch.bfloat16)
            P[v][c0:c0 + CH] = ls.token_terms(w, wh)
        if (c0 // CH) % 8 == 0:
            log(f"vocab {c0 + CH}/{V}")
    refb = ref.to(torch.bfloat16)
    log("reference and screens done")

    res = {"rows": M, "files": nfiles, "V": V, "K": K, "u_eff": ls.U_EFF, "gamma": ls.gamma(K),
           "xnorm": pct(xn), "variants": {}}
    top1 = ls.first_argmax(refb)
    exact_sets = {k: ls.topk_set_mask(refb, k) for k in KS}
    res["topk_set_size"] = {k: pct(m.sum(-1)) for k, m in exact_sets.items()}
    for v in VARIANTS:
        s = scr[v].float()
        if v == "fp8_w8a8":
            # |x.w^ - x^.w^| <= ||x - x^|| ||w^||, and the screen accumulates x^.w^ (norms of x^, w^)
            g = ls.gamma(K)
            B = (xn[:, None] * P[v][None, :] + xerr[:, None] * Pw[None, :] + g * xhn[:, None] * Pw[None, :]
                 + ls.REL * s.abs() + ls.ABS)
        else:
            B = xn[:, None] * P[v][None, :] + ls.REL * s.abs() + ls.ABS
        err = (s - ref).abs()
        ratio = err / B
        r = {"bound_ok": bool((err <= B).all()), "max_err_over_bound": float(ratio.max()),
             "err_over_bound": pct(ratio.flatten()[::97], (50, 99, 99.99)),
             "screen_top1_vs_bf16": float((ls.first_argmax(scr[v]) == top1).float().mean()),
             "P_median": float(P[v].median()), "bound_median_row_top": None, "k": {}}
        lower, upper = s - B, s + B
        r["bound_at_top_token"] = pct(B.gather(1, top1[:, None]).flatten())
        for k in KS:
            for tile in (None, 16, 32, 128):
                T = ls.threshold(lower, k) if tile is None else ls.tile_threshold(lower, k, tile)
                sel = ls.up(upper) >= ls.dn(T)[:, None]
                cnt = sel.sum(-1)
                contains = bool((exact_sets[k] <= sel).all())
                e = {"contains_topk_set": contains, "rows_missing": int((exact_sets[k] & ~sel).any(-1).sum()),
                     "count": pct(cnt), "count_max": int(cnt.max()), "count_mean": float(cnt.float().mean()),
                     "frac_over": {str(c): float((cnt > c).float().mean()) for c in (256, 512, 1024, 2048, 4096, 8192, 16384)}}
                if k == 1:
                    sp = ls.sparse_logits(refb, sel)
                    e["argmax_exact"] = bool((ls.first_argmax(sp) == top1).all())
                    e["top1_sparse_equals_top1_full"] = float((ls.first_argmax(sp) == top1).float().mean())
                r["k"][f"k{k}" + ("" if tile is None else f"_tile{tile}")] = e
                log(v, f"k={k} tile={tile}", json.dumps({kk: e[kk] for kk in ("contains_topk_set", "count",
                                                                                   "count_max")}))
        # informative only (NOT exact): candidate counts if the bound were scaled down to its observed tail
        for scale in (0.25, 0.1):
            Bs = B * scale
            T = ls.threshold(s - Bs, 1)
            sel = ls.up(s + Bs) >= ls.dn(T)[:, None]
            r[f"scaled_bound_{scale}_k1"] = {"count": pct(sel.sum(-1)), "contains_top1": float(
                sel.gather(1, top1[:, None]).float().mean())}
        res["variants"][v] = r
        log(v, "bound_ok", r["bound_ok"], "max err/B", round(r["max_err_over_bound"], 4))
    res["secs"] = round(time.time() - t0)
    json.dump(res, open(OUT, "w"), indent=1)
    log("wrote", OUT)


if __name__ == "__main__":
    main()
