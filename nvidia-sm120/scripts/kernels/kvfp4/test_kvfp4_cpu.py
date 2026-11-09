# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Horizon NVFP4 KV-cache format (hz-kvfp4 v1). No GPU, no TileLang, no vLLM: only torch.

usage: python test_kvfp4_cpu.py [PATH_TO_horizon_ops]  [--json OUT.json]
  PATH_TO_horizon_ops defaults to $HK_SRC/wt-kvfp4/tilelang/contrib/horizon_ops

What runs here (all synthetic data, CPU):
  T1 e4m3 encoder vs an exhaustive nearest-even search over all 256 codes (ties, subnormals, saturation)
  T2 e2m1 rounding vs an exhaustive nearest-even search, packing order (element 2j = low nibble)
  T3 format edge cases: exact grid values round-trip exactly, zero / tiny / huge blocks, no NaN
  T3b block-scale search never worse per block than the plain RN scale, deterministic
  T4 Hadamard: bit-exact butterfly vs the same op order, orthogonality, q'.k' == q.k (fp64)
  T5 round-trip error of K / V (random, outlier-heavy keys, sink token) for NVFP4 with and without Hadamard, next to
     FP8 e4m3 (scale 1, today), int4 per token-head (asymmetric) with / without Hadamard (vLLM int4_per_token_head
     is RHT + asymmetric zero point), and the attention-output error each format causes
  T6 paged write through vLLM's NVFP4 page geometry + reference attention == dense attention over the dequantised
     cache (paging, GQA mapping, causal draft window QL = 4)
  T8 the kernels' bit formulas (e4m3 decode, e2m1 values, nibble expansion, FWHT pair indexing) ported to Python and
     checked against the reference; code * scale is exact in bf16
  T7 emulation of the decode kernel's algorithm (split-KV, fp32 online softmax, bf16 P, fixed-order combine) vs the
     fp64 reference: how much of the 0.0025 relative-L2 gate the algorithm itself uses
"""
import importlib.util
import json
import math
import os
import sys

import torch

torch.manual_seed(0)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
JSON_OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
if JSON_OUT in ARGS:
    ARGS.remove(JSON_OUT)
PKG = ARGS[0] if ARGS else os.path.expanduser(
    "$HK_SRC/wt-kvfp4/tilelang/contrib/horizon_ops")
spec = importlib.util.spec_from_file_location("kv_fp4_ref", os.path.join(PKG, "kv_fp4_ref.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

RESULTS = {}


def record(name, ok, **kw):
    RESULTS[name] = dict(ok=bool(ok), **kw)
    print(json.dumps({"test": name, "ok": bool(ok), **kw}), flush=True)
    assert ok, name


# ------------------------------------------------------------------------------------------------------------ T1 e4m3
def t1():
    tab = R.e4m3_table()
    pos = torch.arange(0, 127)                                   # non-negative finite codes 0x00 .. 0x7e
    vals = tab[pos]
    # probes: every code, every midpoint (ties), points just around midpoints, random, above 448
    mids = (vals[:-1] + vals[1:]) / 2
    probes = torch.cat([vals, mids, torch.nextafter(mids.float(), torch.tensor(0.)).double(),
                        torch.nextafter(mids.float(), torch.tensor(1e9)).double(),
                        torch.rand(20000, dtype=torch.float64) * 460, torch.rand(20000, dtype=torch.float64) * 0.02,
                        torch.tensor([448., 460., 463.99, 464., 470., 1e6, 0.0, 2.0 ** -10, 2.0 ** -9 * 1.5])])
    probes = probes.float()                                      # the kernel input is fp32
    got = R.e4m3_encode(probes).long()
    p = probes.double().clamp(max=448.0)
    d = (p[:, None] - vals[None, :]).abs()
    best = d.min(dim=1, keepdim=True).values
    cand = (d == best)
    # ties -> even code (mantissa lsb 0)
    even = cand & (pos[None, :] % 2 == 0)
    exp = torch.where(even.any(1), even.float().argmax(1), cand.float().argmax(1))
    bad = (got != exp).sum().item()
    record("T1_e4m3_rne_satfinite", bad == 0, probes=probes.numel(), mismatches=bad)


# ------------------------------------------------------------------------------------------------------------ T2 e2m1
def t2():
    grid = torch.tensor(R.E2M1_GRID, dtype=torch.float64)
    mids = (grid[:-1] + grid[1:]) / 2
    y = torch.cat([grid, mids, torch.nextafter(mids.float(), torch.tensor(0.)).double(),
                   torch.nextafter(mids.float(), torch.tensor(9.)).double(),
                   torch.rand(50000, dtype=torch.float64) * 7.0, torch.tensor([6.5, 100.0])]).float()
    got = R.e2m1_code(y).long()
    d = (y.double().clamp(max=6.0)[:, None] - grid[None, :]).abs()
    best = d.min(1, keepdim=True).values
    cand = d == best
    codes = torch.arange(8)
    even = cand & (codes[None, :] % 2 == 0)
    exp = torch.where(even.any(1), even.float().argmax(1), cand.float().argmax(1))
    bad = (got != exp).sum().item()
    # packing: element 2j low nibble, 2j+1 high nibble, sign bit 3
    x = torch.tensor([[1.0, -2.0] + [0.0] * 14 + [6.0, 0.5] + [0.0] * 14], dtype=torch.float32)
    data, sf = R.quantize_rows(x, 1.0)
    # block 0 amax 2 -> s8 = e4m3(2/6) ; codes relative to that scale
    s0 = R.e4m3_decode(sf[0, :1]).item()
    pack_ok = (data[0, 0].item() & 15) == int(R.e2m1_code(torch.tensor([1.0 / s0])).item()) and \
              (data[0, 0].item() >> 4) == (8 | int(R.e2m1_code(torch.tensor([2.0 / s0])).item()))
    record("T2_e2m1_rne_and_packing", bad == 0 and pack_ok, probes=y.numel(), mismatches=bad, packing_ok=pack_ok)


# ------------------------------------------------------------------------------------------------------ T3 edge cases
def t3():
    D = 256
    # exact: values that are grid * power-of-two scale with the block max at 6 * s round-trip bit-exactly
    codes = torch.randint(0, 16, (64, D), dtype=torch.uint8)
    codes[:, ::16] = 7                                          # every block has a +6 so amax / 6 = s exactly
    s = 2.0 ** torch.randint(-6, 6, (64, D // 16)).float()
    x = (R.e2m1_value(codes).view(64, D // 16, 16) * s[..., None]).view(64, D)
    data, sf = R.quantize_rows(x, 1.0)
    exact = torch.equal(R.dequantize_rows(data, sf), x)
    # zero, tiny, huge, mixed
    z = torch.zeros(4, D)
    z[1, :16] = 1e-30
    z[2, :16] = 1e30
    z[3, 5] = -3.0
    dz, sz = R.quantize_rows(z, 1.0)
    rz = R.dequantize_rows(dz, sz)
    zero_ok = bool((dz[0] == 0).all() and (sz[0] == 0).all() and (rz[1] == 0).all())
    finite = bool(torch.isfinite(rz).all())
    huge_sat = rz[2, :16].abs().max().item() == 448.0 * 6.0     # saturates at 448 * 6 = 2688 (g = 1)
    neg_ok = rz[3, 5].item() == -3.0
    # global scale: with g = 2^-4 the same block quantises to the same codes, scales 16x
    g = 2.0 ** -4
    d2, s2 = R.quantize_rows(x * g, g)
    gscale_ok = torch.equal(d2, data) and torch.equal(s2, sf)
    record("T3_format_edges", exact and zero_ok and finite and huge_sat and neg_ok and gscale_ok, exact_grid=exact,
           zero_ok=zero_ok, finite=finite, huge_saturates_2688=huge_sat, negative_ok=neg_ok,
           global_scale_invariant=gscale_ok)


# ------------------------------------------------------------------------------------------------------- T4 Hadamard
def t4():
    D = 256
    sg = R.hadamard_signs(D)
    x = torch.randn(512, D)
    y = R.fwht(x.clone())
    # dense Sylvester matrix check (fp64)
    i = torch.arange(D)
    Hm = torch.tensor([[(-1) ** bin(a & b).count("1") for b in range(D)] for a in range(D)], dtype=torch.float64)
    dense = (x.double() @ Hm)
    close = (y.double() - dense).abs().max().item() / dense.abs().max().item()
    inv = (R.fwht(R.fwht(x.clone())) / D - x).abs().max().item()
    q, k = torch.randn(64, D), torch.randn(64, D)
    dq = (R.rotate(q, sg).double() * R.rotate(k, sg).double()).sum(-1)
    d0 = (q.double() * k.double()).sum(-1)
    dot = ((dq - d0).abs().max() / d0.abs().max()).item()
    # the butterfly is deterministic: two evaluations are bitwise equal, and fwht_scale(256) is exactly 1/16
    det = torch.equal(R.rotate(q, sg), R.rotate(q, sg)) and R.fwht_scale(256) == 1.0 / 16
    record("T4_hadamard", close < 1e-6 and inv < 1e-3 and dot < 1e-5 and det and float(sg.abs().min()) == 1.0,
           rel_vs_dense=close, inverse_err=inv, dot_rel_err=dot, deterministic=det)


# ------------------------------------------------------------------------------------------------ T5 error statistics
def int4_tokenhead(x, rot=None):
    """int4 per (token, head) asymmetric (scale + zero point per 256 values), optional rotation (RHT)"""
    y = R.rotate(x, rot) if rot is not None else x.float()
    lo, hi = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    sc = ((hi - lo) / 15).clamp_min(1e-12)
    qv = torch.round((y - lo) / sc).clamp(0, 15)
    deq = qv * sc + lo
    if rot is not None:          # undo the rotation to compare in the original space
        deq = R.fwht(deq / R.fwht_scale(x.shape[-1]) / x.shape[-1]) * rot
    return deq


def nvfp4_rt(x, g=1.0, rot=None, search=True):
    y = R.rotate(x, rot) if rot is not None else x.float()
    d, s = R.quantize_rows(y, g, search)
    deq = R.dequantize_rows(d, s, g)
    if rot is not None:
        deq = R.fwht(deq / R.fwht_scale(x.shape[-1]) / x.shape[-1]) * rot
    return deq


def fp8_rt(x):
    return x.float().clamp(-448, 448).to(torch.float8_e4m3fn).float()


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def make_kv(kind, N=4096, D=256, gen=None):
    k = torch.randn(N, D, generator=gen)
    v = torch.randn(N, D, generator=gen) * torch.exp(0.5 * torch.randn(N, 1, generator=gen))
    if kind == "outlier":
        ch = torch.tensor([0, 1, 64, 65, 126, 127, 200, 201])   # RoPE-pair-like outlier channels
        k[:, ch] = k[:, ch] * 6 + 10                              # big and biased, like low-frequency RoPE dims
        k[0] *= 8                                                 # an "attention sink" token (stays < 448: FP8-safe)
        v[0] *= 0.05
    return k.to(torch.bfloat16).float(), v.to(torch.bfloat16).float()


OUTLIER_CH = torch.tensor([0, 1, 64, 65, 126, 127, 200, 201])


def make_q(k, gen, kind, nq=64):
    q = torch.randn(nq, k.shape[1], generator=gen) * 2
    if kind == "outlier":                                         # queries lean on the same channels as the keys
        q[:, OUTLIER_CH] *= 3
    return q


def attn_err(k, v, kq, vq, q, sm=1 / 16):
    """relative L2 of the attention output over 4096 keys, the SAME queries for every format of a data kind"""
    def a(K, V):
        p = torch.softmax((q.double() @ K.double().T) * sm, -1)
        return p @ V.double()
    return rel(a(kq, vq), a(k, v))


def t5():
    """mean over 5 seeds (4096 keys, 64 queries each); the same queries for every format of a seed"""
    sg = R.hadamard_signs(256)
    rows = {}
    for kind in ("gaussian", "outlier"):
        acc = {}
        for seed in range(5):
            gen = torch.Generator().manual_seed(100 + seed)
            k, v = make_kv(kind, gen=gen)
            q = make_q(k, gen, kind)
            fmts = {
                "fp8_e4m3_scale1 (today)": (fp8_rt(k), fp8_rt(v)),
                "nvfp4 RN scale (K,V plain)": (nvfp4_rt(k, search=False), nvfp4_rt(v, search=False)),
                "nvfp4 (K,V plain)": (nvfp4_rt(k), nvfp4_rt(v)),
                "nvfp4 (K Hadamard, V plain)": (nvfp4_rt(k, rot=sg), nvfp4_rt(v)),
                "fp8 K + nvfp4 V (K8V4-like)": (fp8_rt(k), nvfp4_rt(v)),
                "int4 token-head asym": (int4_tokenhead(k), int4_tokenhead(v)),
                "int4 token-head asym + RHT": (int4_tokenhead(k, sg), int4_tokenhead(v, sg)),
            }
            for name, (kq, vq) in fmts.items():
                a_ = acc.setdefault(name, {"k_rel_l2": [], "v_rel_l2": [], "attn_out_rel_l2": []})
                a_["k_rel_l2"].append(rel(kq, k))
                a_["v_rel_l2"].append(rel(vq, v))
                a_["attn_out_rel_l2"].append(attn_err(k, v, kq, vq, q))
        for name, a_ in acc.items():
            rows[f"{kind}/{name}"] = {m: round(sum(x) / len(x), 4) for m, x in a_.items()}
            print(json.dumps({"kind": kind, "format": name, **rows[f"{kind}/{name}"]}), flush=True)
    had_helps = rows["outlier/nvfp4 (K Hadamard, V plain)"]["k_rel_l2"] < rows["outlier/nvfp4 (K,V plain)"]["k_rel_l2"]
    beats_int4 = rows["gaussian/nvfp4 (K,V plain)"]["k_rel_l2"] < rows["gaussian/int4 token-head asym"]["k_rel_l2"]
    search_helps = all(rows[f"{kd}/nvfp4 (K,V plain)"]["k_rel_l2"] < rows[f"{kd}/nvfp4 RN scale (K,V plain)"]["k_rel_l2"]
                       for kd in ("gaussian", "outlier"))
    # measurement, not a pass/fail gate: synthetic data only; the real gate is the model log-prob / NLL / needle run
    record("T5_roundtrip_error_stats", search_helps, hadamard_reduces_outlier_key_error=had_helps,
           nvfp4_beats_int4_tokenhead_gaussian=beats_int4, scale_search_reduces_error=search_helps, table=rows)


def t3b():
    """block-scale search: never worse than the RN scale per block (c0 is a candidate), deterministic"""
    x = torch.randn(4096, 256) * torch.exp(torch.randn(4096, 1))
    xb = x.view(-1, 16, 16)
    worse = 0
    d1, s1 = R.quantize_rows(x, 1.0, True)
    d0, s0 = R.quantize_rows(x, 1.0, False)
    e1 = ((R.dequantize_rows(d1, s1) - x) ** 2).view(-1, 16, 16).sum(-1)
    e0 = ((R.dequantize_rows(d0, s0) - x) ** 2).view(-1, 16, 16).sum(-1)
    worse = int((e1 > e0 * (1 + 1e-6)).sum())
    det = all(torch.equal(a, b) for a, b in zip(R.quantize_rows(x, 1.0, True), (d1, s1)))
    changed = float((s1 != s0).float().mean())
    record("T3b_scale_search", worse == 0 and det, blocks=e1.numel(), blocks_worse=worse, deterministic=det,
           fraction_of_blocks_with_other_scale=round(changed, 4))


# ------------------------------------------------------------------------------------------ T6 paged write + attention
def build_case(B, lens, QL, PS=64, KVH=4, G=6, D=256, rot=True, gen=None, g=(1.0, 1.0)):
    NT = sum(lens)
    NP = sum(math.ceil(L / PS) for L in lens) + 3
    cache = R.alloc_cache(NP, PS, KVH, D, fill=0xAA)            # garbage outside written slots
    views = R.split_views(cache, KVH)
    perm = torch.randperm(NP - 1, generator=gen) + 1
    MP = max(math.ceil(L / PS) for L in lens)
    bt = torch.zeros(B, MP, dtype=torch.int32)
    o = 0
    for b, L in enumerate(lens):
        n = math.ceil(L / PS)
        bt[b, :n] = perm[o:o + n].int()
        o += n
    slots = []
    for b, L in enumerate(lens):
        for p in range(L):
            slots.append(int(bt[b, p // PS]) * PS + p % PS)
    slots = torch.tensor(slots, dtype=torch.int64)
    key = torch.randn(NT, KVH, D, generator=gen).to(torch.bfloat16)
    val = torch.randn(NT, KVH, D, generator=gen).to(torch.bfloat16)
    gk = torch.full((KVH,), g[0])
    gv = torch.full((KVH,), g[1])
    sg = R.hadamard_signs(D) if rot else None
    # pad slot (-1) must be skipped
    R.write_reference(torch.cat([key, key[:1]]), torch.cat([val, val[:1]]), views,
                      torch.cat([slots, torch.tensor([-1])]), gk, gv, sg)
    q = torch.randn(B * QL, KVH * G, D, generator=gen).to(torch.bfloat16)
    qr = R.rotate_q(q, sg) if rot else q
    return dict(cache=cache, views=views, bt=bt, sl=torch.tensor(lens, dtype=torch.int32), key=key, val=val, q=q,
                qr=qr, gk=gk, gv=gv, sg=sg, QL=QL, KVH=KVH, G=G, D=D, lens=lens)


def dense_attention(c, sm):
    """dense per-request attention over dequantised K (in rotated space) / V from the token arrays, no paging"""
    KVH, G, D, QL = c["KVH"], c["G"], c["D"], c["QL"]
    k = R.rotate(c["key"], c["sg"]) if c["sg"] is not None else c["key"].float()
    dk, sk = R.quantize_rows(k, c["gk"])
    dv, sv = R.quantize_rows(c["val"].float(), c["gv"])
    K = R.dequantize_rows(dk, sk, c["gk"].double(), torch.float64)
    V = R.dequantize_rows(dv, sv, c["gv"].double(), torch.float64)
    out = torch.empty(c["q"].shape, dtype=torch.float64)
    o = 0
    for b, L in enumerate(c["lens"]):
        for t in range(QL):
            lim = L - QL + t + 1
            qq = c["qr"][b * QL + t].double().view(KVH, G, D)
            s = torch.einsum("hgd,lhd->hgl", qq, K[o:o + lim]) * sm
            out[b * QL + t] = torch.einsum("hgl,lhd->hgd", torch.softmax(s, -1), V[o:o + lim]).reshape(-1, D)
        o += L
    return out


def t6():
    gen = torch.Generator().manual_seed(11)
    worst = 0.0
    for lens, QL, g in (([1, 70, 129], 1, (1.0, 1.0)), ([5, 64, 300], 4, (1.0, 0.5)), ([1000], 4, (0.25, 2.0))):
        c = build_case(len(lens), lens, QL, gen=gen, g=g)
        sm = 256 ** -0.5
        ref = R.attention_reference(c["qr"], c["views"], c["bt"], c["sl"], QL, sm, c["gk"], c["gv"])
        den = dense_attention(c, sm)
        worst = max(worst, rel(ref, den))
    # the rotation leaves q.k unchanged: attention with the original q over un-rotated dequantised keys is the same
    # up to the quantisation of the (rotated vs plain) keys only -> compare rotated-space vs plain-space attention
    record("T6_paged_reference_equals_dense", worst < 1e-12, worst_rel_l2=worst)


# --------------------------------------------------------------------------------- T7 decode-kernel algorithm emulation
def emulate_kernel(c, sm, splits=4, block_N=32, min_chunk=256, v_scale=1.0):
    """what attn_decode_fp4 computes: bf16 K/V tiles (code * sf exact), fp32 scores * (sm * gk * log2e), exp2 online
    softmax per split, P rounded to bf16, fp32 accumulate, per-split normalise * gv, fixed-order combine"""
    KVH, G, D, QL = c["KVH"], c["G"], c["D"], c["QL"]
    LOG2E = 1.4426950408889634
    out = torch.empty(c["q"].shape, dtype=torch.float32)
    for b, L in enumerate(c["lens"]):
        K, V = R.gather(c["views"], c["bt"][b], L)
        K, V = K.float().to(torch.bfloat16).float(), V.float().to(torch.bfloat16).float()   # exact
        chunk = max(min_chunk, math.ceil(math.ceil(L / splits) / block_N) * block_N)
        for h in range(KVH):
            qrows = torch.stack([c["qr"][b * QL + i // G, h * G + i % G].float() for i in range(G * QL)])
            sc = torch.tensor(sm * float(c["gk"][h]) * LOG2E, dtype=torch.float32)
            parts, lses = [], []
            for s in range(splits):
                st, en = s * chunk, min(L, (s + 1) * chunk)
                m = torch.full((G * QL,), -math.inf)
                l = torch.zeros(G * QL)
                acc = torch.zeros(G * QL, D)
                for p0 in range(st, max(en, st), block_N):
                    kk = K[p0:min(en, p0 + block_N), h]
                    sco = (qrows @ kk.T) * sc
                    pos = torch.arange(p0, p0 + kk.shape[0])
                    lim = L - QL + torch.arange(G * QL) // G + 1
                    sco = torch.where(pos[None, :] < lim[:, None], sco, torch.tensor(-math.inf))
                    mn = torch.maximum(m, sco.max(1).values)
                    alpha = torch.where(mn == -math.inf, torch.ones_like(mn), torch.exp2(m - mn))
                    p = torch.where(mn[:, None] == -math.inf, torch.zeros_like(sco), torch.exp2(sco - mn[:, None]))
                    l = l * alpha + p.sum(1)
                    acc = acc * alpha[:, None] + p.to(torch.bfloat16).float() @ V[p0:p0 + kk.shape[0], h]
                    m = mn
                parts.append(torch.where(l[:, None] > 0, acc / l.clamp_min(1e-30)[:, None], torch.zeros_like(acc))
                             * float(c["gv"][h]))
                lses.append(torch.where(l > 0, torch.log2(l.clamp_min(1e-30)) + m, torch.tensor(-math.inf)))
            lse = torch.stack(lses)                           # (S, R)
            mx = lse.max(0).values
            w = torch.where(lse == -math.inf, torch.zeros_like(lse), torch.exp2(lse - mx))
            w = w / w.sum(0)
            o = sum(parts[s] * w[s][:, None] for s in range(splits)) * v_scale
            for i in range(G * QL):
                out[b * QL + i // G, h * G + i % G] = o[i]
    return out.to(torch.bfloat16)


def t7():
    gen = torch.Generator().manual_seed(13)
    worst = 0.0
    for lens, QL, splits in (([1, 200], 1, 4), ([700, 33], 4, 3), ([2048], 4, 8)):
        c = build_case(len(lens), lens, QL, gen=gen, g=(1.0, 1.0))
        sm = 256 ** -0.5
        ref = R.attention_reference(c["qr"], c["views"], c["bt"], c["sl"], QL, sm, c["gk"], c["gv"])
        emu = emulate_kernel(c, sm, splits=splits)
        worst = max(worst, rel(emu, ref))
    record("T7_kernel_algorithm_emulation", worst <= 0.0025, worst_rel_l2=worst, gate=0.0025)


# ------------------------------------------------------------------ T8 kernel helper formulas (Python ports of the C / TIR)
def t8():
    """the kernels' bit formulas, ported line by line, against the reference tables"""
    import struct

    def u2f(u):
        return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
    tab = R.e4m3_table()
    # hz_e4m3_dec: e ? bits((e + 120) << 23 | m << 20) : m * 2^-9
    dec_ok = all((u2f(((c >> 3 & 15) + 120) << 23 | (c & 7) << 20) if (c >> 3 & 15) else (c & 7) * 0.001953125)
                 == tab[c].item() for c in range(127))
    # hz_e2m1_val: e ? bits((e + 126) << 23 | l << 22) : (l ? .5 : 0)
    val_ok = all((u2f(((m >> 1 & 3) + 126) << 23 | (m & 1) << 22) if (m >> 1 & 3) else (0.5 if m & 1 else 0.0))
                 == R.E2M1_GRID[m] for m in range(8))
    # _fp4_val (decode kernel TIR): 2|v| = m (e == 0) else (2 + m) << (e - 1); sign bit 3; times 0.5 * sf
    ok_nib = True
    for byte in range(256):
        for j in (0, 1):
            c = (byte >> (4 * j)) & 15
            e, m = (c >> 1) & 3, c & 1
            two_v = m if e == 0 else (2 + m) << max(e - 1, 0)
            v = two_v * (-0.5 if c & 8 else 0.5)
            ref = R.e2m1_value(torch.tensor([c], dtype=torch.uint8)).item()
            ok_nib = ok_nib and v == ref
    # FWHT pair index (kv_write_fp4.fwht_body): i = ((r >> st) << (st + 1)) + (r & ((1 << st) - 1)), partner i + 2^st
    D = 256
    pair_ok = True
    for st in range(8):
        h = 1 << st
        idx = sorted([((r >> st) << (st + 1)) + (r & (h - 1)) for r in range(D // 2)])
        pair_ok = pair_ok and idx == [i for i in range(D) if not i & h]
    # full kernel-order dequant of random bytes == dequantize_rows
    data = torch.randint(0, 256, (64, 128), dtype=torch.uint8)
    sf = torch.randint(0, 127, (64, 16), dtype=torch.uint8)
    ref = R.dequantize_rows(data, sf)
    port = torch.empty(64, 256)
    lut2 = [0, 1, 2, 3, 4, 6, 8, 12]
    for r in range(64):
        for j in range(256):
            c = (int(data[r, j // 2]) >> (4 * (j % 2))) & 15
            port[r, j] = lut2[c & 7] * (-0.5 if c & 8 else 0.5) * tab[int(sf[r, j // 16])].item()
    deq_ok = torch.equal(port, ref)
    # bf16 exactness of code * sf (why the tensor-core tile is exact)
    bf_ok = torch.equal(ref.to(torch.bfloat16).float(), ref)
    record("T8_kernel_formulas", dec_ok and val_ok and ok_nib and pair_ok and deq_ok and bf_ok, e4m3_dec=dec_ok,
           e2m1_val=val_ok, nibble_formula=ok_nib, fwht_pairs=pair_ok, dequant_port=deq_ok, bf16_exact=bf_ok)


if __name__ == "__main__":
    for f in (t1, t2, t3, t3b, t4, t5, t6, t7, t8):
        f()
    print(json.dumps({"ALL_CPU_TESTS": "PASS", "n": len(RESULTS)}), flush=True)
    if JSON_OUT:
        with open(JSON_OUT, "w") as fh:
            json.dump(RESULTS, fh, indent=1)
