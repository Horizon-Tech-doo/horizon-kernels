# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels exact-argmax output layer (lm_head screening), 30-09-2026. Switch HORIZON_LMHEAD_SCREEN, default 0.
#
# Idea (lm-head-screening (not in this release)): the BF16 lm_head (248,320 x 5,120, 2.54 GB per read) is read once per verify step
# and once per draft step above 16 rows. Greedy decoding, greedy draft verification and top-k sampling only need the
# few largest logits of each row. So:
#   1. screen: approximate logits s~ for every token from a quantized copy of the head (FP8 e4m3 per row: 1.27 GB);
#   2. certify: a rigorous per-(row, token) bound B with |s~ - rho| <= B, where rho is the fp32 value any BF16 GEMM
#      (cuBLAS in any summation order, or our recheck kernel) accumulates before its final BF16 rounding;
#   3. select every token whose largest possible BF16 value can reach the smallest possible BF16 value of the k-th
#      best token (k = 1 for greedy, k = top_k for top-k sampling). Every token of the exact top-k set, including
#      BF16 ties at its boundary, is provably inside the selection;
#   4. recheck the selected tokens exactly (BF16 weights, fp32 accumulation, one BF16 rounding) and hand vLLM a
#      "sparse logits" row: exact values at the selected tokens, -inf elsewhere. Argmax, top-k / top-p masks, the
#      greedy and probabilistic rejection sampler and the Gumbel sampler then see exactly the masked logits they see
#      today. Rows whose selection overflows the per-row capacity get their full row recomputed (device-side, no
#      host sync), so the result never depends on the capacity.
#
# Bound (all terms computed from data, no statistical assumption):
#   B[r, c] = ||x_r|| * P[c] + REL * |s~[r, c]| + ABS
#   P[c]    = ||w_c - w^_c||_2 + GAMMA * (||w_c||_2 + ||w^_c||_2)
# ||x_r|| * ||w_c - w^_c|| bounds the quantization error x.(w - w^) (Cauchy-Schwarz); GAMMA * ||x|| * ||w|| bounds
# the fp32 accumulation error of a K-term dot product in any order (gamma_K with unit roundoff 2^-22: 4x the IEEE
# 2^-24 to cover tensor-core truncation and the split-K / scale multiply roundings); REL * |s~| bounds the screen's
# own BF16 output rounding (half an ulp <= 2^-8 relative). BF16 rounding of the exact value is monotone, and one
# BF16 ulp is at most 2^-7 * |v|, so a token c can only tie or beat the k-th best token if
#   up(s~_c + B_c) >= dn(T_k),   up(v) = v + 2^-7 |v| + ABS,   dn(v) = v - 2^-7 |v| - ABS,
# with T_k = the k-th largest lower bound s~ - B over the row (at least k tokens have a value >= T_k).
#
# What "exact" means here: the selected set provably contains the production top-k set. The final values are
# production's only if the recheck reproduces production's summation order bit for bit (H1 in the design doc,
# measured on the GPU); otherwise HORIZON_LMHEAD_SCREEN_OWN=1 serves the full head from the same recheck kernel too
# (one arithmetic everywhere, batch-invariant) and screening is exact with respect to that head by construction.
import math
import os

import torch

K_HIDDEN = 5120
U_EFF = 2.0 ** -22                       # unit roundoff used for accumulation bounds (4x IEEE fp32)
REL = 2.0 ** -8 * (1.0 + 2.0 ** -6)       # screen BF16 output rounding, relative to the rounded value
ULP_REL = 2.0 ** -7                      # one BF16 ulp relative to |v| (normal range)
ABS = 1e-30
XNORM_SLACK = 1.0 + 1e-3                 # ||x|| is computed in fp32 on the GPU


def gamma(k: int, u: float = U_EFF) -> float:
    """gamma_k = k u / (1 - k u): bound of a k-term floating-point sum in any order, relative to sum |a_i|."""
    ku = k * u
    assert ku < 0.5, k
    return ku / (1.0 - ku) * 1.01          # + the per-column scale multiply and a margin


def token_terms(w_bf16: torch.Tensor, w_hat: torch.Tensor) -> torch.Tensor:
    """P[c] for a chunk of rows: ||w - w^|| + GAMMA (||w|| + ||w^||), computed in float64, returned as float32
    rounded up (never below the float64 value)."""
    w = w_bf16.double()
    wh = w_hat.double()
    g = gamma(w.shape[1])
    p = (w - wh).norm(dim=1) + g * (w.norm(dim=1) + wh.norm(dim=1))
    p32 = p.float()
    return torch.where(p32.double() < p, torch.nextafter(p32, torch.full_like(p32, math.inf)), p32)


def row_norms(x: torch.Tensor) -> torch.Tensor:
    return x.float().norm(dim=-1) * XNORM_SLACK


def up(v):
    return v + ULP_REL * v.abs() + ABS


def dn(v):
    return v - ULP_REL * v.abs() - ABS


def bounds(s: torch.Tensor, xnorm: torch.Tensor, P: torch.Tensor):
    """(lower, upper) float32 bounds on the pre-rounding fp32 logit, per (row, token)."""
    s = s.float()
    B = xnorm.float()[:, None] * P.float()[None, :] + REL * s.abs() + ABS
    return s - B, s + B


def threshold(lower: torch.Tensor, k: int) -> torch.Tensor:
    """T_k per row: the k-th largest lower bound."""
    return lower.topk(k, dim=-1).values[:, -1]


def tile_threshold(lower: torch.Tensor, k: int, tile: int) -> torch.Tensor:
    """What the kernels compute: the k-th largest of the per-tile maxima of the lower bounds. Never above
    threshold(lower, k) (the k largest tile maxima are k distinct lower bounds), so it selects a superset."""
    M, V = lower.shape
    nt = (V + tile - 1) // tile
    pad = nt * tile - V
    lw = torch.nn.functional.pad(lower, (0, pad), value=-math.inf) if pad else lower
    tmax = lw.view(M, nt, tile).amax(-1)
    return tmax.topk(min(k, nt), dim=-1).values[:, -1]


def select(s: torch.Tensor, xnorm: torch.Tensor, P: torch.Tensor, k: int, tile: int | None = None):
    """Reference selection: boolean mask (M, V) of the candidates, and the threshold used."""
    lower, upper = bounds(s, xnorm, P)
    T = threshold(lower, k) if tile is None else tile_threshold(lower, k, tile)
    return up(upper) >= dn(T)[:, None], T


def topk_set_mask(logits_bf16: torch.Tensor, k: int) -> torch.Tensor:
    """Exact top-k set of BF16 logits including every tie at the k-th value (what top-k masking keeps)."""
    kth = logits_bf16.topk(k, dim=-1).values[:, -1:]
    return logits_bf16 >= kth


def first_argmax(logits_bf16: torch.Tensor) -> torch.Tensor:
    """Smallest index among the maxima: vLLM's greedy rule (per-block tl.max, then tl.argmax over blocks, both
    first-index). torch.argmax is documented to return the first maximal index; no (M, V) temporaries."""
    return logits_bf16.argmax(-1)


def sparse_logits(exact_bf16: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reference output: exact values where selected, -inf elsewhere."""
    return torch.where(mask, exact_bf16, torch.full_like(exact_bf16, -math.inf))


# --------------------------------------------------------------------------------------------------------------------
# FP8 per-row screen copy (the S3 file format: weight e4m3 (V, K), weight_scale fp32 (V,), w^ = scale * weight).
def quant_fp8_rows(w_bf16: torch.Tensor):
    w = w_bf16.float()
    sc = w.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
    return (w / sc).to(torch.float8_e4m3fn), sc.reshape(-1)


def dequant_fp8_rows(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float()[:, None]


def screen_fp8_reference(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """What decode_w8a16 computes, up to fp32 summation order: bf16(scale * (x . q))."""
    return ((x.float() @ q.float().t()) * scale.float()[None, :]).to(torch.bfloat16)


# ====================================================================================================================
# vLLM 0.30 integration (V2 model runner). Every switch defaults off; nothing here runs unless HORIZON_LMHEAD_SCREEN=1.
#
#   HORIZON_LMHEAD_SCREEN=1          the screen is built for the served BF16 lm_head on its first eager use (vLLM's
#                                    dummy sampler run, before CUDA-graph capture): FP8 per-row copy (+1.27 GB), P
#   HORIZON_LMHEAD_SCREEN_TARGET=1   target verify / sampler rows use it when EVERY request of the step is eligible
#                                    (default 1): greedy, or top_k <= HORIZON_LMHEAD_SCREEN_TOPK (default 0 = greedy
#                                    only) with at most HORIZON_LMHEAD_SCREEN_TOPK_ROWS rows (default 32); no
#                                    logprobs, logprob token ids, logit bias / min_tokens / allowed ids, penalties,
#                                    bad words, thinking budget, min_p, grammar, watermark, trace replay, NaN count,
#                                    batch sharding, sampling mask, synthetic rejection sampling. Otherwise stock.
#   HORIZON_LMHEAD_SCREEN_DRAFT=1    the drafter's local-argmax fallback (LogitsProcessor.get_top_tokens: rows above
#                                    the K1a draft head's 16) uses it (default 1); runs inside the draft CUDA graphs
#   HORIZON_LMHEAD_SCREEN_KMAX       per-row candidate capacity, greedy (default 4096; overflow = full row, exact)
#   HORIZON_LMHEAD_SCREEN_TOPK_KMAX  per-row capacity in top-k mode (default 16384)
#   HORIZON_LMHEAD_SCREEN_MAXM       rows per call above which the stock head is used (default 64: from 65 rows the FP8
#                                    W8A16 screen is as slow as the BF16 head, race 28-09-2026: 1.6 vs 1.7 ms at 128)
#   HORIZON_LMHEAD_SCREEN_SELFTEST_MAXM  row counts the self-test covers (default 256; timing up to 256 too)
#   HORIZON_LMHEAD_SCREEN_OWN=1      ineligible sample steps use the recheck kernel on every row instead of cuBLAS
#                                    (one arithmetic for every served logit: "route B" in the design doc; default 0)
#   HORIZON_LMHEAD_SCREEN_SELFTEST=<json path>  at attach: compare the screen with the stock head (cuBLAS through the
#                                    layer's own quant_method) at every M in 1..SELFTEST_MAXM on real hidden states
#                                    (HORIZON_LMHEAD_SCREEN_SELFTEST_H=<.pt>, default synthetic): recomputed values
#                                    bitwise, argmax, top-k sets; write the JSON
#   HORIZON_LMHEAD_SCREEN_H1_GUARD=1 (default 1 with a self-test) screen only at the M values whose self-test values
#                                    were bit-identical to the stock head
#   HORIZON_LMHEAD_SCREEN_LOG_EVERY  counters line every N eligible target calls (default 2000)
_ENV = os.environ.get
_ST = {"screen": None, "tried": False, "ctx": None, "ok_m": None, "n": {}, "cand": None, "own": None}


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def on(name, default="1"):
    return _ENV(name, default) == "1"


def _cnt(key, v=1):
    _ST["n"][key] = _ST["n"].get(key, 0) + v


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def build_screen(w, kmax=None):
    """Screen for the served BF16 head w (V, K): FP8 per-row copy and P, computed on w's device in row chunks."""
    try:                                   # dev packages ship the kernels with the plugin (production TileLang unchanged)
        from horizon_kernels.lmscreen_ops.lmhead_screen import Screen
    except ImportError:
        from tilelang.contrib.sm120_fp8.lmhead_screen import Screen
    V, K = w.shape
    q = torch.empty(V, K, dtype=torch.float8_e4m3fn, device=w.device)
    sc = torch.empty(V, dtype=torch.float32, device=w.device)
    P = torch.empty(V, dtype=torch.float32, device=w.device)
    for r0 in range(0, V, 8192):
        qq, ss = quant_fp8_rows(w[r0:r0 + 8192])
        q[r0:r0 + 8192], sc[r0:r0 + 8192] = qq, ss
        P[r0:r0 + 8192] = token_terms(w[r0:r0 + 8192], dequant_fp8_rows(qq, ss))
    return Screen(w, q, sc, P, kmax=kmax or int(_ENV("HORIZON_LMHEAD_SCREEN_KMAX", "4096")),
                  topk_kmax=int(_ENV("HORIZON_LMHEAD_SCREEN_TOPK_KMAX", "16384")))


def _warm(screen, maxm):
    """Run every kernel configuration once outside capture: decode_w8a16's config changes with M (its buckets), our
    kernels are M-dynamic (one compile each)."""
    from horizon_kernels.lmhead import _configuration_boundary_rows
    from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8

    def key(rows):
        cfg = w8.config(screen.V, screen.K, rows)
        return tuple(cfg[:7]) + (cfg[7] if len(cfg) > 7 else 1,)

    rows = _configuration_boundary_rows(key, maxm)
    dev = screen.w.device
    for m in rows:
        x = torch.randn(m, screen.K, device=dev, dtype=torch.bfloat16)
        screen.logits(x)
    screen.full(torch.randn(2, screen.K, device=dev, dtype=torch.bfloat16))
    x = torch.randn(4, screen.K, device=dev, dtype=torch.bfloat16)
    screen.logits(x, k=torch.full((4,), 2, dtype=torch.int64, device=dev), k_host_max=2)
    torch.cuda.synchronize()
    return rows


def selftest(lm_head, screen, stock, maxm, path, hpath=""):
    """Screen vs the stock head at every M in 1..maxm. stock(x) must be exactly what vLLM computes (the layer's own
    quant_method.apply). Records, per M: values at every selected token bit-identical, first-index argmax equal,
    top-20 set (with ties) inside the selection and equal after masking, overflow rows."""
    import json
    dev = screen.w.device
    if hpath:
        H = torch.load(hpath, map_location="cpu").reshape(-1, screen.K).to(torch.bfloat16)
        H = H[H.float().abs().amax(-1) > 0]
    else:
        g = torch.Generator().manual_seed(0)
        H = (torch.randn(4096, screen.K, generator=g) * 1.7).to(torch.bfloat16)
    res, ok_m, off = {}, set(), 0
    for m in range(1, maxm + 1):
        x = H[torch.arange(off, off + m) % H.shape[0]].to(dev)
        off += m
        ref = stock(x)
        out = screen.logits(x)
        n_ovf = int((screen.last_cnt > screen.last_kmax).sum())
        sel = torch.isfinite(out)
        vals_equal = bool((out[sel] == ref[sel]).all())
        am = bool((first_argmax(out) == first_argmax(ref)).all())
        k20 = torch.full((m,), 20, dtype=torch.int64, device=dev)
        out20 = screen.logits(x, k=k20, k_host_max=20)
        sel20 = torch.isfinite(out20)
        in20 = bool((topk_set_mask(ref, 20) <= sel20).all())
        v20 = bool((out20[sel20] == ref[sel20]).all())
        n_ovf20 = int((screen.last_cnt > screen.last_kmax).sum())
        res[m] = {"values_bitwise": vals_equal, "argmax_equal": am, "top20_inside": in20, "top20_values_bitwise": v20,
                  "overflow_rows": n_ovf, "overflow_rows_top20": n_ovf20, "selected_mean": float(sel.sum(-1).float().mean())}
        if vals_equal and am and v20 and in20:
            ok_m.add(m)
    summary = {"M_tested": maxm, "M_ok": len(ok_m), "M_not_ok": sorted(set(range(1, maxm + 1)) - ok_m),
               "hidden": hpath or "synthetic", "rows_used": off}
    timing = {}
    for m in (1, 4, 8, 16, 17, 32, 33, 64, 65, 128, 256):
        if m > maxm:
            continue
        x = H[torch.arange(m) % H.shape[0]].to(dev)
        k20 = torch.full((m,), 20, dtype=torch.int64, device=dev)
        fns = {"stock": lambda: stock(x), "screen_greedy": lambda: screen.logits(x),
               "screen_top20": lambda: screen.logits(x, k=k20, k_host_max=20)}
        timing[m] = {}
        for name, fn in fns.items():          # eager, CUDA events; weights (2.5 / 1.3 GB) exceed the 128 MB L2
            for _ in range(3):
                fn()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(20):
                fn()
            b.record()
            torch.cuda.synchronize()
            timing[m][name + "_us"] = round(a.elapsed_time(b) * 1000 / 20, 1)
    summary["timing_us"] = timing
    with open(path, "w") as f:
        json.dump({"summary": summary, "per_M": res}, f, indent=1)
    return ok_m, summary


def _attach(lm_head, device):
    w = getattr(lm_head, "weight", None)
    if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or getattr(lm_head, "tp_size", 1) != 1:
        _log().warning("horizon_kernels lmhead screen: head is not a single-GPU BF16 weight; screen off")
        return None
    V = getattr(lm_head, "num_embeddings_per_partition", w.shape[0])
    w = w[:V]
    maxm = int(_ENV("HORIZON_LMHEAD_SCREEN_MAXM", "64"))
    screen = build_screen(w)
    rows = _warm(screen, maxm)
    st = _ENV("HORIZON_LMHEAD_SCREEN_SELFTEST", "")
    if st:
        def stock(x):
            return lm_head.quant_method.apply(lm_head, x, bias=None)
        ok_m, summ = selftest(lm_head, screen, stock, int(_ENV("HORIZON_LMHEAD_SCREEN_SELFTEST_MAXM", "256")), st,
                              _ENV("HORIZON_LMHEAD_SCREEN_SELFTEST_H", ""))
        _log().info("horizon_kernels lmhead screen self-test: %s", summ)
        if on("HORIZON_LMHEAD_SCREEN_H1_GUARD"):
            _ST["ok_m"] = ok_m
    _log().info("horizon_kernels lmhead screen: exact FP8 screen on the %dx%d BF16 head (kmax %d, warm-up rows %s, "
                "target=%s draft=%s topk=%s own=%s)", screen.V, screen.K, screen.kmax, list(rows),
                on("HORIZON_LMHEAD_SCREEN_TARGET"), on("HORIZON_LMHEAD_SCREEN_DRAFT"),
                _ENV("HORIZON_LMHEAD_SCREEN_TOPK", "0"), on("HORIZON_LMHEAD_SCREEN_OWN", "0"))
    return screen


def _screen_for(lm_head, x):
    s = _ST["screen"]
    if s is None and not _ST["tried"] and not _capturing() and x.is_cuda:
        _ST["tried"] = True
        _ST["screen"] = s = _attach(lm_head, x.device)
        _ST["head"] = lm_head
    if s is None or _ST.get("head") is not lm_head:
        return None
    return s


def _m_ok(m):
    ok = _ST["ok_m"]
    return m <= int(_ENV("HORIZON_LMHEAD_SCREEN_MAXM", "64")) and (ok is None or m in ok)


def _logits_processing_active(smp, idx):
    """True if any request in `idx` uses logit bias, penalties, bad words or a custom logits processor.
    vLLM 0.30: named states on the sampler. vLLM 0.32 moved them into smp.logits_processors (a list; the bias and bad-words
    states are no longer attributes), so walk the list, and treat any processor this does not know as active (stock path)."""
    lps = getattr(smp, "logits_processors", None)
    if lps is None:
        return bool(smp.logit_bias_state.use_logit_bias[idx].any() or smp.penalties_state.use_penalty[idx].any()
                    or (smp.bad_words_state.num_bad_words.np[idx] > 0).any())
    for p in lps:
        if hasattr(p, "use_logit_bias"):
            on_ = p.use_logit_bias[idx].any()
        elif hasattr(p, "use_penalty"):
            on_ = p.use_penalty[idx].any()
        elif hasattr(p, "num_bad_words"):
            on_ = (p.num_bad_words.np[idx] > 0).any()
        else:
            return True
        if on_:
            return True
    return False


def eligibility(runner, input_batch, grammar_output):
    """-> (mode, k_rows, k_max, reason): mode "greedy" | "topk" | None (stock). Host-side only (numpy state)."""
    import numpy as np
    if grammar_output is not None:
        return None, None, 0, "grammar"
    if getattr(runner, "batch_sharder", None) is not None:
        return None, None, 0, "batch_sharder"
    smp = getattr(runner, "sampler", None)
    if smp is None or type(smp).__name__ != "Sampler":
        return None, None, 0, "sampler_type"
    if smp.compute_nans or smp.trace_replay_state is not None or smp.return_sampling_mask:
        return None, None, 0, "sampler_flags"
    rs = getattr(runner, "rejection_sampler", None)
    if rs is not None and (rs.synthetic_conditional_rates is not None or getattr(rs, "watermark_key", None) is not None):
        return None, None, 0, "rejection_method"
    idx = input_batch.idx_mapping_np
    if smp.get_logprobs_dims(idx) is not None:
        return None, None, 0, "logprobs"
    tb = smp.thinking_budget_state
    if _logits_processing_active(smp, idx) or (tb.enabled and tb.use_thinking_budget[idx].any()):
        return None, None, 0, "logits_processing"
    ss = smp.sampling_states
    if (ss.min_p.np[idx] != 0.0).any():
        return None, None, 0, "min_p"
    temp = ss.temperature.np[idx]
    greedy = temp == 0.0
    counts = np.diff(input_batch.cu_num_logits_np[: input_batch.num_reqs + 1])
    rows = int(counts.sum())
    if not _m_ok(rows):
        return None, None, 0, "rows"
    if greedy.all():
        return "greedy", None, 1, "ok"
    kt = int(_ENV("HORIZON_LMHEAD_SCREEN_TOPK", "0"))
    topk = ss.top_k.np[idx].astype(np.int64)
    if kt <= 0 or rows > int(_ENV("HORIZON_LMHEAD_SCREEN_TOPK_ROWS", "32")) or (topk[~greedy] > kt).any():
        return None, None, 0, "sampling"
    k_req = np.where(greedy, 1, topk)
    k_rows = np.repeat(k_req, counts)
    return "topk", k_rows, int(k_rows.max()), "ok"


def install():
    """Hooks: LogitsProcessor._apply_head (target logits), LogitsProcessor.get_top_tokens (drafter local argmax) and
    GPUModelRunner.sample (per-step eligibility). Called from register() when HORIZON_LMHEAD_SCREEN=1."""
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    if getattr(LogitsProcessor._apply_head, "_horizon_screen", False):
        return
    orig_apply = LogitsProcessor._apply_head
    orig_top = LogitsProcessor.get_top_tokens

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        ctx, _ST["ctx"] = _ST["ctx"], None          # one use per sample() call
        if embedding_bias is None and hidden_states.dtype == torch.bfloat16 and self.head_dtype in (None, torch.bfloat16):
            s = _screen_for(lm_head, hidden_states)
            if s is not None and ctx is not None:
                x2 = hidden_states.reshape(-1, hidden_states.shape[-1])
                if ctx[0] == "greedy":
                    out = s.logits(x2)
                elif ctx[0] == "topk":
                    k = torch.as_tensor(ctx[1], dtype=torch.int64).to(x2.device, non_blocking=True)
                    out = s.logits(x2, k, ctx[2])
                elif ctx[0] == "own":
                    out = s.full(x2)
                else:
                    out = None
                if out is not None:
                    return out.reshape(*hidden_states.shape[:-1], s.V)
        return orig_apply(self, lm_head, hidden_states, embedding_bias)

    def get_top_tokens(self, lm_head, hidden_states, embedding_bias=None):
        if (on("HORIZON_LMHEAD_SCREEN_DRAFT") and embedding_bias is None and self.scale == 1.0 and self.soft_cap is None
                and hidden_states.dtype == torch.bfloat16 and getattr(lm_head, "tp_size", 1) == 1
                and self.head_dtype in (None, torch.bfloat16)):
            s = _screen_for(lm_head, hidden_states)
            if s is not None and _m_ok(hidden_states.shape[0]):
                num_pad = lm_head.shard_indices.num_org_vocab_padding
                if num_pad == 0:
                    out = s.logits(hidden_states.reshape(-1, hidden_states.shape[-1]))
                    return out.max(dim=-1).indices + lm_head.shard_indices.org_vocab_start_index
        return orig_top(self, lm_head, hidden_states, embedding_bias)

    _apply_head._horizon_screen = True
    get_top_tokens._horizon_screen = True
    LogitsProcessor._apply_head = _apply_head
    LogitsProcessor.get_top_tokens = get_top_tokens

    if on("HORIZON_LMHEAD_SCREEN_TARGET") or on("HORIZON_LMHEAD_SCREEN_OWN", "0"):
        import vllm.v1.worker.gpu.model_runner as mr
        orig_sample = mr.GPUModelRunner.sample

        def sample(self, hidden_states, input_batch, grammar_output):
            mode, k_rows, k_max, why = (None, None, 0, "target_off")
            if on("HORIZON_LMHEAD_SCREEN_TARGET"):
                try:
                    mode, k_rows, k_max, why = eligibility(self, input_batch, grammar_output)
                except Exception as e:  # noqa: BLE001  never break serving on an unexpected batch layout
                    mode, why = None, "error:" + type(e).__name__
            if mode is None and on("HORIZON_LMHEAD_SCREEN_OWN", "0") and hidden_states.shape[0] > 0:
                mode = "own"
            _cnt("calls")
            _cnt("mode:" + str(mode) if mode else "stock:" + why)
            _ST["ctx"] = (mode, k_rows, k_max) if mode else None
            try:
                return orig_sample(self, hidden_states, input_batch, grammar_output)
            finally:
                _ST["ctx"] = None
                every = int(_ENV("HORIZON_LMHEAD_SCREEN_LOG_EVERY", "2000"))
                if every > 0 and _ST["n"]["calls"] % every == 0:
                    _log().info("horizon_kernels lmhead screen counters: %s", dict(sorted(_ST["n"].items())))

        sample._horizon_screen = True
        mr.GPUModelRunner.sample = sample
    _log().info("horizon_kernels lmhead screen: hooks installed (target=%s, draft=%s, topk=%s, own=%s)",
                on("HORIZON_LMHEAD_SCREEN_TARGET"), on("HORIZON_LMHEAD_SCREEN_DRAFT"),
                _ENV("HORIZON_LMHEAD_SCREEN_TOPK", "0"), on("HORIZON_LMHEAD_SCREEN_OWN", "0"))
