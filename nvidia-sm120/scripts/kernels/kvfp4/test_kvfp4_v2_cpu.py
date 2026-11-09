# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for attn_decode_fp4_v2 (register-level FP4 unpack). No GPU, no TileLang: numpy + torch only.

usage: python test_kvfp4_v2_cpu.py [PATH_TO_horizon_ops] [--json OUT.json]

  V1 sf117: all 256 scale bytes -> bf16 bits of e4m3(byte & 0x7f) * 2^117, exact, finite
  V2 K unpack, exhaustive: every packed byte (in every byte position of the word) x every scale code 0..127: the
     Marlin placement + ONE bf16x2 multiply gives e2m1 * e4m3 * 2^-9 exactly, in the pair order the kernel assumes
     (P3 = (e3, e7), P2 = (e2, e6), P1 = (e1, e5), P0 = (e0, e4))
  V3 V unpack, exhaustive: every byte of token A and token B in every byte position x every scale code: byte_perm
     + Marlin + bf16x2 multiply give (A.lo, B.lo) / (A.hi, B.hi) exactly
  V4 lane-level emulation of the whole kernel (Q/K/V fragment layouts of mma.m16n8k16 per the PTX ISA, movmatrix,
     shuffles, cp.async staging layout, split chunking, fixed-order slice sum and combine) against the fp64
     reference over the same dequantised cache: relative L2 <= 0.0025, dense and MoE geometry, QL 1 and 4, NCW 2/4,
     masked / partial blocks, multi-page requests, empty splits
  V5 shared-memory bank conflicts of every LDS in the emulated main loop (addresses recorded per lane)
The emulation checks the index math and numerics model; it cannot prove the PTX layouts or the hardware's bf16
subnormal handling (mul.rn.bf16x2 on the 0.5 code), which the GPU gate (test_kvfp4_gpu_v2.py) covers.
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
JSON_OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
if JSON_OUT in ARGS:
    ARGS.remove(JSON_OUT)
PKG = ARGS[0] if ARGS else os.path.expanduser("$HK_SRC/wt-kvfp4/tilelang/contrib/horizon_ops")
spec = importlib.util.spec_from_file_location("kv_fp4_ref", os.path.join(PKG, "kv_fp4_ref.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
RESULTS = {}
U32 = np.uint32


def record(name, ok, **kw):
    RESULTS[name] = dict(ok=bool(ok), **kw)
    print(json.dumps({"test": name, "ok": bool(ok), **kw}), flush=True)
    assert ok, name


# ------------------------------------------------------------------------------------------------ bit-level model
def bf16_to_f64(bits):
    return (np.asarray(bits, dtype=U32) << U32(16)).view(np.float32).astype(np.float64)


def f32_to_bf16(x):
    """RNE float32 -> bf16 bits (finite inputs)"""
    u = np.asarray(x, dtype=np.float32).view(U32)
    return ((u + U32(0x7FFF) + ((u >> U32(16)) & U32(1))) >> U32(16)).astype(U32)


def mar(y):
    y = np.asarray(y, dtype=U32)
    return (y & U32(0x80008000)) | ((y & U32(0x70007000)) >> U32(6))


def hmul2(a, b, check=None):
    """mul.rn.bf16x2 with IEEE subnormals: exact product rounded to bf16 (records whether rounding was needed)"""
    a, b = np.asarray(a, dtype=U32), np.asarray(b, dtype=U32)
    out = np.zeros(np.broadcast(a, b).shape, dtype=U32)
    for sh in (0, 16):
        pa = bf16_to_f64((a >> U32(sh)) & U32(0xFFFF))
        pb = bf16_to_f64((b >> U32(sh)) & U32(0xFFFF))
        p = pa * pb
        r = f32_to_bf16(p.astype(np.float32))        # products here are exact in bf16, so no double rounding
        if check is not None:
            check.append(bool(np.all(bf16_to_f64(r) == p)))
        out |= (r & U32(0xFFFF)) << U32(sh)
    return out


def sf117(byte):
    byte = np.asarray(byte, dtype=U32)
    e, m = (byte >> U32(3)) & U32(15), byte & U32(7)
    normal = (((e + U32(237)) << U32(23)) | (m << U32(20))).view(np.float32)
    sub = m.astype(np.float32) * np.float32(2.0 ** 108)
    f = np.where(e != 0, normal, sub).astype(np.float32)
    return (f.view(U32) >> U32(16)).astype(U32)


def byte_perm(a, b, sel):
    a, b = np.asarray(a, dtype=U32), np.asarray(b, dtype=U32)
    src = [(a >> U32(8 * i)) & U32(0xFF) for i in range(4)] + [(b >> U32(8 * i)) & U32(0xFF) for i in range(4)]
    out = np.zeros(np.broadcast(a, b).shape, dtype=U32)
    for i in range(4):
        out |= src[(sel >> (4 * i)) & 7] << U32(8 * i)
    return out


E4M3 = R.e4m3_table().numpy()
GRID = np.array(R.E2M1_GRID)


def sf_val(code):
    """what the kernel decodes a scale byte to: e4m3(byte & 0x7f), with 0x7f (NaN in e4m3fn) read as 480"""
    code = np.asarray(code) & 0x7F
    return np.where(code == 0x7F, 1.875 * 2.0 ** 8, np.abs(np.nan_to_num(E4M3[code])))


def nib_val(n):
    n = np.asarray(n)
    v = GRID[n & 7]
    return np.where(n & 8, -v, v)


# --------------------------------------------------------------------------------------------------------- V1..V3
def v1():
    b = np.arange(256, dtype=U32)
    got = bf16_to_f64(sf117(b))
    exp = np.array([abs(E4M3[x & 0x7F]) if (x & 0x7F) != 0x7F else 1.875 * 2.0 ** 8 for x in range(256)]) * 2.0 ** 117
    ok = bool(np.all(got == exp) and np.all(np.isfinite(got)))
    record("V1_sf117_exhaustive", ok, bytes=256)


def v2():
    bytes_ = np.arange(256, dtype=U32)
    words = (bytes_ * U32(0x01010101))[:, None]                     # the byte in all 4 positions
    codes = np.arange(128, dtype=U32)[None, :]
    sc = sf117(codes)
    scp = sc | (sc << U32(16))
    chk = []
    P = {3: hmul2(mar(words), scp, chk), 2: hmul2(mar(words << U32(4)), scp, chk),
         1: hmul2(mar(words << U32(8)), scp, chk), 0: hmul2(mar(words << U32(12)), scp, chk)}
    sfv = sf_val(codes)
    ok = True
    for k, reg in P.items():
        for half, elem in ((0, k), (1, k + 4)):
            nib = (words >> U32(4 * elem)) & U32(15)
            exp = nib_val(nib) * sfv * 2.0 ** -9
            got = bf16_to_f64((reg >> U32(16 * half)) & U32(0xFFFF))
            ok = ok and bool(np.all(got == exp))
    record("V2_k_unpack_exhaustive", ok and all(chk), combos=256 * 128 * 8, exact_products=all(chk))


def v3():
    rng = np.random.default_rng(3)
    ok, chk, n = True, [], 0
    for jj in range(4):
        sel = jj | (jj << 4) | ((4 + jj) << 8) | ((4 + jj) << 12)
        for side in ("A", "B"):
            by = np.arange(256, dtype=U32)[:, None]
            other = rng.integers(0, 256, size=(256, 1)).astype(U32)
            codes = np.arange(128, dtype=U32)[None, :]
            ocodes = rng.integers(0, 127, size=(1, 128)).astype(U32)
            rnd = rng.integers(0, 2 ** 32, size=(256, 1), dtype=np.uint64).astype(U32)
            wA = (rnd & ~U32(0xFF << (8 * jj))) | ((by if side == "A" else other) << U32(8 * jj))
            wB = (rnd[::-1] & ~U32(0xFF << (8 * jj))) | ((other if side == "A" else by) << U32(8 * jj))
            sA = sf117(codes if side == "A" else ocodes)
            sB = sf117(ocodes if side == "A" else codes)
            sAB = sA | (sB << U32(16))
            y = byte_perm(wA, wB, sel)
            lo = hmul2(mar(y << U32(4)), sAB, chk)
            hi = hmul2(mar(y), sAB, chk)
            for reg, nibsh in ((lo, 0), (hi, 4)):
                for half, (w_, sc_) in enumerate(((wA, codes if side == "A" else ocodes),
                                                  (wB, ocodes if side == "A" else codes))):
                    nib = (w_ >> U32(8 * jj + nibsh)) & U32(15)
                    exp = nib_val(nib) * sf_val(sc_) * 2.0 ** -9
                    got = bf16_to_f64((reg >> U32(16 * half)) & U32(0xFFFF))
                    ok = ok and bool(np.all(got == exp))
                    n += got.size
    record("V3_v_unpack_exhaustive", ok and all(chk), checked_values=n, exact_products=all(chk))


# ------------------------------------------------------------------------------------------ V4 lane-level emulation
class Banks:
    def __init__(self):
        self.worst = {}

    def lds(self, name, addrs, width):
        """addrs: (32,) byte addresses of one warp-wide load of `width` bytes per lane"""
        per_phase = {16: 8, 8: 16}.get(width, 32)
        worst = 1
        for p0 in range(0, 32, per_phase):
            banks = {}
            for a in addrs[p0:p0 + per_phase]:
                for wd in range(max(1, width // 4)):
                    word = (int(a) // 4) + wd
                    banks.setdefault(word % 32, set()).add(word)
            worst = max(worst, max(len(v) for v in banks.values()))
        self.worst[name] = max(self.worst.get(name, 1), worst)


def mma_emul(acc, a, b):
    """acc (4, 32) f32; a (4, 32) u32; b (2, 32) u32 -> acc + A @ B with the m16n8k16 fragment layouts"""
    lane = np.arange(32)
    g, c = lane >> 2, lane & 3
    A = np.zeros((16, 16))
    B = np.zeros((16, 8))
    for e in range(2):
        A[g, 2 * c + e] = bf16_to_f64((a[0] >> U32(16 * e)) & U32(0xFFFF))
        A[g + 8, 2 * c + e] = bf16_to_f64((a[1] >> U32(16 * e)) & U32(0xFFFF))
        A[g, 2 * c + 8 + e] = bf16_to_f64((a[2] >> U32(16 * e)) & U32(0xFFFF))
        A[g + 8, 2 * c + 8 + e] = bf16_to_f64((a[3] >> U32(16 * e)) & U32(0xFFFF))
        B[2 * c + e, g] = bf16_to_f64((b[0] >> U32(16 * e)) & U32(0xFFFF))
        B[2 * c + 8 + e, g] = bf16_to_f64((b[1] >> U32(16 * e)) & U32(0xFFFF))
    Dm = (A @ B).astype(np.float32)
    out = acc.copy()
    out[0] += Dm[g, 2 * c]
    out[1] += Dm[g, 2 * c + 1]
    out[2] += Dm[g + 8, 2 * c]
    out[3] += Dm[g + 8, 2 * c + 1]
    return out.astype(np.float32)


def movt_emul(x):
    lane = np.arange(32)
    g, c = lane >> 2, lane & 3
    M = np.zeros((8, 8), dtype=U32)
    M[g, 2 * c] = x & U32(0xFFFF)
    M[g, 2 * c + 1] = x >> U32(16)
    Tm = M.T
    return (Tm[g, 2 * c] | (Tm[g, 2 * c + 1] << U32(16))).astype(U32)


def shfl_xor(v, m):
    return v[np.arange(32) ^ m]


def emulate(qbits, kd, ksf, vd, vsf, bt, sl, gk, gv, scale_log2, KVH, G, QL, NCW, NST, MINC, PS, S, banks,
            mutate=None):
    """one launch: returns opart (B, KVH, S, 32, 256) f32 and lse (B, KVH, S, 32) f32 as the kernel writes them"""
    D, HQ, Rr = 256, KVH * G, G * QL
    NQT = (Rr + 7) // 8
    DW = D // NCW
    RB, VRS, SB, CPL, CPG = DW // 2, DW // 2 + 16, DW // 16, DW // 4, DW // 8
    KW, KS, MT = CPL // 8, DW // 16, DW // 16
    STAGE = 16 * RB + 16 * VRS + 32 * SB
    Bn = sl.shape[0]
    opart = np.full((Bn, KVH, S, 32, D), np.nan, dtype=np.float32)
    lse = np.full((Bn, KVH, S, 32), np.nan, dtype=np.float32)
    lane = np.arange(32)
    g, c = lane >> 2, lane & 3
    NEG = np.float32(-np.inf)
    for b in range(Bn):
        L = int(sl[b])
        for h in range(KVH):
            for s in range(S):
                chunk = max(MINC, ((L + S - 1) // S + 15) // 16 * 16)
                st, en = s * chunk, min(L, s * chunk + chunk)
                nblk = (en - st + 15) // 16 if en > st else 0
                sc = np.float32(np.float32(scale_log2) * np.float32(gk[h]) * np.float32(512.0))
                W = []
                for cs in range(NCW):
                    cb = cs * DW
                    qf = np.zeros((NQT, KS, 2, 32), dtype=U32)
                    for nt in range(NQT):
                        row = nt * 8 + g
                        okr = row < Rr
                        tq, hq = np.where(okr, row // G, 0), np.where(okr, row % G, 0)
                        for w in range(KW):
                            for u in range(2):
                                ch = cb + CPL * c + 8 * w + 2 * u
                                qr = qbits[b * QL + tq, h * G + hq]
                                v0 = qr[lane, ch].astype(U32) | (qr[lane, ch + 4].astype(U32) << U32(16))
                                v1 = qr[lane, ch + 1].astype(U32) | (qr[lane, ch + 5].astype(U32) << U32(16))
                                qf[nt, 2 * w + u, 0] = np.where(okr, v0, 0)
                                qf[nt, 2 * w + u, 1] = np.where(okr, v1, 0)
                    W.append(dict(cb=cb, qf=qf, m=np.full((NQT, 2, 32), NEG, np.float32),
                                  l=np.zeros((NQT, 2, 32), np.float32), acc=np.zeros((MT, NQT, 4, 32), np.float32)))
                for k in range(nblk):
                    p0 = st + 16 * k
                    page, pos0 = int(bt[b, p0 // PS]), p0 % PS
                    sacc_all = []
                    stages = []
                    for cs in range(NCW):
                        cb = W[cs]["cb"]
                        # the cp.async stage image: K rows RB, V rows VRS (16 padding bytes), K / V scale slices
                        stg = np.zeros(STAGE, dtype=np.uint8)
                        for r in range(16):
                            stg[r * RB:(r + 1) * RB] = kd[page, pos0 + r, h, cb // 2:cb // 2 + RB]
                            stg[16 * RB + r * VRS:16 * RB + r * VRS + RB] = vd[page, pos0 + r, h, cb // 2:cb // 2 + RB]
                            o = 16 * RB + 16 * VRS
                            stg[o + r * SB:o + (r + 1) * SB] = ksf[page, pos0 + r, h, cs * SB:(cs + 1) * SB]
                            stg[o + 16 * SB + r * SB:o + 16 * SB + (r + 1) * SB] = vsf[page, pos0 + r, h,
                                                                                       cs * SB:(cs + 1) * SB]
                        stages.append(stg)
                        base = cs * NST * STAGE + (k % NST) * STAGE      # absolute smem address for bank checks
                        sKS = 16 * RB + 16 * VRS

                        def word(off):
                            return stg[off[:, None] + np.arange(4)].astype(U32) @ (U32(1) << (U32(8) * np.arange(4, dtype=U32)))
                        kw0 = [word(g * RB + (CPL // 2) * c + 4 * w) for w in range(KW)]
                        kw1 = [word((g + 8) * RB + (CPL // 2) * c + 4 * w) for w in range(KW)]
                        banks.lds(f"K_data_ncw{NCW}", base + g * RB + (CPL // 2) * c, CPL // 2)
                        banks.lds(f"K_data_ncw{NCW}_row8", base + (g + 8) * RB + (CPL // 2) * c, CPL // 2)
                        sacc = np.zeros((NQT, 4, 32), np.float32)
                        for w in range(KW):
                            blk = (CPL * c + 8 * w) // 16
                            banks.lds(f"K_scale_ncw{NCW}", base + sKS + g * SB + blk, 1)
                            s0 = sf117(stg[sKS + g * SB + blk])
                            s1 = sf117(stg[sKS + (g + 8) * SB + blk])
                            s0p, s1p = s0 | (s0 << U32(16)), s1 | (s1 << U32(16))
                            y0, y1 = kw0[w], kw1[w]
                            Pa = {3: hmul2(mar(y0), s0p), 2: hmul2(mar(y0 << U32(4)), s0p),
                                  1: hmul2(mar(y0 << U32(8)), s0p), 0: hmul2(mar(y0 << U32(12)), s0p)}
                            Pb = {3: hmul2(mar(y1), s1p), 2: hmul2(mar(y1 << U32(4)), s1p),
                                  1: hmul2(mar(y1 << U32(8)), s1p), 0: hmul2(mar(y1 << U32(12)), s1p)}
                            if mutate == "k_pairs":                   # negative control: wrong k-slot mapping
                                Pa[1], Pa[2] = Pa[2], Pa[1]
                            for nt in range(NQT):
                                qfw = W[cs]["qf"]
                                sacc[nt] = mma_emul(sacc[nt], np.stack([Pa[0], Pb[0], Pa[1], Pb[1]]),
                                                    np.stack([qfw[nt, 2 * w, 0], qfw[nt, 2 * w, 1]]))
                                sacc[nt] = mma_emul(sacc[nt], np.stack([Pa[2], Pb[2], Pa[3], Pb[3]]),
                                                    np.stack([qfw[nt, 2 * w + 1, 0], qfw[nt, 2 * w + 1, 1]]))
                        sacc_all.append(sacc)
                    tot = sacc_all[0].copy()
                    for cs in range(1, NCW):
                        tot = (tot + sacc_all[cs]).astype(np.float32)
                    for cs in range(NCW):
                        Wc = W[cs]
                        x = np.zeros((NQT, 4, 32), np.float32)
                        for nt in range(NQT):
                            for e in range(4):
                                tok = g if e < 2 else g + 8
                                row = nt * 8 + 2 * c + (e & 1)
                                pos = p0 + tok
                                ok = (row < Rr) & (pos < en) & (pos < L - QL + row // G + 1)
                                x[nt, e] = np.where(ok, (tot[nt, e] * sc).astype(np.float32), NEG)
                        alpha = np.zeros((NQT, 2, 32), np.float32)
                        resc = False
                        for nt in range(NQT):
                            for par in range(2):
                                bm = np.maximum(x[nt, par], x[nt, par + 2])
                                for msk in (4, 8, 16):
                                    bm = np.maximum(bm, shfl_xor(bm, msk))
                                mn = np.maximum(Wc["m"][nt, par], bm)
                                with np.errstate(invalid="ignore"):
                                    a = np.where(mn == NEG, np.float32(1), np.exp2(Wc["m"][nt, par] - mn)).astype(np.float32)
                                    pa = np.where(mn == NEG, np.float32(0), np.exp2(x[nt, par] - mn)).astype(np.float32)
                                    pb_ = np.where(mn == NEG, np.float32(0), np.exp2(x[nt, par + 2] - mn)).astype(np.float32)
                                Wc["l"][nt, par] = (Wc["l"][nt, par] * a + (pa + pb_)).astype(np.float32)
                                Wc["m"][nt, par] = mn
                                alpha[nt, par] = a
                                resc = resc or bool(np.any(a != 1))
                                x[nt, par], x[nt, par + 2] = pa, pb_
                        if resc:
                            for e in range(4):
                                Wc["acc"][:, :, e] = (Wc["acc"][:, :, e] * alpha[None, :, e & 1]).astype(np.float32)
                        pbf = np.zeros((NQT, 2, 32), U32)
                        for nt in range(NQT):
                            pbf[nt, 0] = movt_emul(f32_to_bf16(x[nt, 0]) | (f32_to_bf16(x[nt, 1]) << U32(16)))
                            pbf[nt, 1] = movt_emul(f32_to_bf16(x[nt, 2]) | (f32_to_bf16(x[nt, 3]) << U32(16)))
                        stg = stages[cs]
                        base = cs * NST * STAGE + (k % NST) * STAGE
                        sV, sVS = 16 * RB, 16 * RB + 16 * VRS + 16 * SB
                        VB = CPG // 2

                        def vword(tokrow, wi):
                            off = sV + tokrow * VRS + VB * g + 4 * wi
                            return stg[off[:, None] + np.arange(4)].astype(U32) @ (U32(1) << (U32(8) * np.arange(4, dtype=U32)))
                        nvw = (MT + 3) // 4
                        va = [vword(2 * c, i) for i in range(nvw)]
                        vb = [vword(2 * c + 1, i) for i in range(nvw)]
                        vc = [vword(2 * c + 8, i) for i in range(nvw)]
                        vdw = [vword(2 * c + 9, i) for i in range(nvw)]
                        for nm, tr in (("A", 2 * c), ("B", 2 * c + 1), ("C", 2 * c + 8), ("D", 2 * c + 9)):
                            banks.lds(f"V_data_ncw{NCW}", base + sV + tr * VRS + VB * g, VB)
                        vblk = (CPG * g) // 16
                        banks.lds(f"V_scale_ncw{NCW}", base + sVS + (2 * c) * SB + vblk, 1)
                        sA = sf117(stg[sVS + (2 * c) * SB + vblk])
                        sB = sf117(stg[sVS + (2 * c + 1) * SB + vblk])
                        sC = sf117(stg[sVS + (2 * c + 8) * SB + vblk])
                        sD = sf117(stg[sVS + (2 * c + 9) * SB + vblk])
                        sAB, sCD = sA | (sB << U32(16)), sC | (sD << U32(16))
                        for j in range(MT):
                            sel = (j & 3) | ((j & 3) << 4) | ((4 + (j & 3)) << 8) | ((4 + (j & 3)) << 12)
                            if mutate == "v_tokens":                  # negative control: token pair swapped
                                yab = byte_perm(vb[j >> 2], va[j >> 2], sel)
                            else:
                                yab = byte_perm(va[j >> 2], vb[j >> 2], sel)
                            ycd = byte_perm(vc[j >> 2], vdw[j >> 2], sel)
                            a4 = np.stack([hmul2(mar(yab << U32(4)), sAB), hmul2(mar(yab), sAB),
                                           hmul2(mar(ycd << U32(4)), sCD), hmul2(mar(ycd), sCD)])
                            for nt in range(NQT):
                                Wc["acc"][j, nt] = mma_emul(Wc["acc"][j, nt], a4, pbf[nt])
                gvs = np.float32(np.float32(gv[h]) * np.float32(512.0))
                for cs in range(NCW):
                    Wc = W[cs]
                    lt = Wc["l"].copy()
                    for msk in (4, 8, 16):
                        lt = (lt + lt[:, :, np.arange(32) ^ msk]).astype(np.float32)
                    for nt in range(NQT):
                        for par in range(2):
                            row = nt * 8 + 2 * c + par
                            sel_ = (g == 0) & (row < Rr)
                            if cs == 0:
                                with np.errstate(divide="ignore"):
                                    v = np.where(lt[nt, par] > 0, np.log2(np.maximum(lt[nt, par], 1e-38)) + Wc["m"][nt, par], NEG)
                                lse[b, h, s, row[sel_]] = v[sel_]
                    for j in range(MT):
                        for nt in range(NQT):
                            for e in range(4):
                                row = nt * 8 + 2 * c + (e & 1)
                                ch = Wc["cb"] + CPG * g + 2 * j + (e >> 1)
                                l_ = lt[nt, e & 1]
                                val = np.where(l_ > 0, Wc["acc"][j, nt, e] / np.where(l_ > 0, l_, 1) * gvs, 0).astype(np.float32)
                                okr = row < Rr
                                opart[b, h, s, row[okr], ch[okr]] = val[okr]
    return opart, lse


def combine_emul(opart, lse, R_, QL, G, v_scale):
    """attn_decode.attn_combine_kernel: fixed-order merge -> (B*QL, HQ, D) bf16 (as f64)"""
    Bn, KVH, S = lse.shape[:3]
    out = np.zeros((Bn * QL, KVH * G, 256))
    for b in range(Bn):
        for h in range(KVH):
            for r in range(R_):
                ls = lse[b, h, :, r].astype(np.float32)
                mx = np.float32(max(ls))
                w = np.where(ls == -np.inf, 0, np.exp2(ls - mx)).astype(np.float32)
                tot = np.float32(w.sum())
                acc = np.zeros(256, np.float32)
                for k in range(S):
                    acc = acc + opart[b, h, k, r] * (w[k] / tot if w[k] else np.float32(0))
                o = torch.tensor(acc * np.float32(v_scale)).to(torch.bfloat16).double().numpy()
                out[b * QL + r // G, h * G + r % G] = o
    return out


def build(B, lens, QL, KVH, G, seed, PS=64, had=True, gk=None, gv=None):
    gen = torch.Generator().manual_seed(seed)
    pages = [math.ceil(L / PS) for L in lens]
    NP = sum(pages) + 3
    perm = torch.randperm(NP - 1, generator=gen) + 1
    bt = torch.zeros(B, max(pages), dtype=torch.int32)
    o = 0
    for i, p in enumerate(pages):
        bt[i, :p] = perm[o:o + p].int()
        o += p
    slots = torch.tensor([int(bt[i, p // PS]) * PS + p % PS for i, L in enumerate(lens) for p in range(L)])
    NT = sum(lens)
    key = torch.randn(NT, KVH, 256, generator=gen)
    key[:, :, [0, 1, 64, 65]] = key[:, :, [0, 1, 64, 65]] * 6 + 10
    val = torch.randn(NT, KVH, 256, generator=gen) * torch.exp(0.5 * torch.randn(NT, 1, 1, generator=gen))
    key, val = key.to(torch.bfloat16), val.to(torch.bfloat16)
    cache = R.alloc_cache(NP, PS, KVH, 256, fill=0x7F)            # garbage (0x7f = NaN in e4m3!) outside slots
    views = R.split_views(cache, KVH)
    gk = torch.ones(KVH) if gk is None else gk
    gv = torch.ones(KVH) if gv is None else gv
    sg = R.hadamard_signs(256) if had else None
    R.write_reference(key, val, views, slots, gk, gv, sg)
    q = torch.randn(B * QL, KVH * G, 256, generator=gen).to(torch.bfloat16)
    qr = R.rotate_q(q, sg) if had else q
    return dict(views=views, bt=bt, sl=torch.tensor(lens, dtype=torch.int32), qr=qr, gk=gk, gv=gv)


def v4():
    banks = Banks()
    worst = 0.0
    cases = [(4, 6, 1, [37, 200], 2, 2), (4, 6, 4, [300, 17], 3, 2), (4, 6, 4, [150], 8, 2),
             (2, 8, 1, [130, 64], 2, 2), (2, 8, 4, [96, 250], 3, 4), (4, 6, 4, [1000], 4, 2)]
    recs = []
    for i, (KVH, G, QL, lens, S, NCW) in enumerate(cases):
        gk = torch.tensor([1.0, 0.5, 2.0, 1.0][:KVH])
        gv = torch.tensor([1.0, 2.0, 1.0, 0.25][:KVH])
        cse = build(len(lens), lens, QL, KVH, G, 40 + i, had=(i % 2 == 0), gk=gk, gv=gv)
        kd, ksf, vd, vsf = (v.numpy() for v in cse["views"])
        qbits = cse["qr"].view(torch.int16).numpy().view(np.uint16)
        sm = 256 ** -0.5
        op, ls = emulate(qbits, kd, ksf, vd, vsf, cse["bt"].numpy(), cse["sl"].numpy(), cse["gk"].numpy(),
                         cse["gv"].numpy(), sm * 1.4426950408889634, KVH, G, QL, NCW, 4, 64, 64, S, banks)
        out = combine_emul(op, ls, G * QL, QL, G, 1.0)
        ref = R.attention_reference(cse["qr"], cse["views"], cse["bt"], cse["sl"], QL, sm, cse["gk"],
                                    cse["gv"]).numpy()
        r = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
        fin = bool(np.isfinite(out).all())
        worst = max(worst, r)
        recs.append(dict(KVH=KVH, G=G, QL=QL, lens=lens, S=S, NCW=NCW, rel_l2=round(r, 6), finite=fin))
        print(json.dumps({"case": recs[-1]}), flush=True)
        assert fin
    # negative controls: the same emulation with a deliberately wrong K k-slot or V token mapping must fail clearly
    ctl = {}
    for mut in ("k_pairs", "v_tokens"):
        cse = build(2, [300, 17], 4, 4, 6, 99, had=False)
        kd, ksf, vd, vsf = (v.numpy() for v in cse["views"])
        qbits = cse["qr"].view(torch.int16).numpy().view(np.uint16)
        op, ls = emulate(qbits, kd, ksf, vd, vsf, cse["bt"].numpy(), cse["sl"].numpy(), cse["gk"].numpy(),
                         cse["gv"].numpy(), 256 ** -0.5 * 1.4426950408889634, 4, 6, 4, 2, 4, 64, 64, 3, Banks(),
                         mutate=mut)
        out = combine_emul(op, ls, 24, 4, 6, 1.0)
        ref = R.attention_reference(cse["qr"], cse["views"], cse["bt"], cse["sl"], 4, 256 ** -0.5, cse["gk"],
                                    cse["gv"]).numpy()
        ctl[mut] = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
    detects = all(v > 0.05 for v in ctl.values())
    record("V4_lane_emulation_vs_reference", worst <= 0.0025 and detects, worst_rel_l2=worst, gate=0.0025,
           negative_controls_rel_l2=ctl, negative_controls_detected=detects, cases=recs)
    return banks


def v5(banks):
    ok = all(v == 1 for k, v in banks.worst.items() if "data" in k)
    record("V5_bank_conflicts", ok, worst_ways=banks.worst,
           note="data loads must be conflict-free; 1-byte scale loads are same-word broadcasts or small")


if __name__ == "__main__":
    v1()
    v2()
    v3()
    bk = v4()
    v5(bk)
    print(json.dumps({"ALL_V2_CPU_TESTS": "PASS", "n": len(RESULTS)}), flush=True)
    if JSON_OUT:
        with open(JSON_OUT, "w") as fh:
            json.dump(RESULTS, fh, indent=1)
