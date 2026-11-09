"""GQA-packed flash-attention forward for gfx90a (MI250X), TileLang.

The upstream AMD example gives every query head its own workgroup, so with GQA each K/V tile is loaded once per query
head (6x for Qwen's 24 q / 4 kv heads). Here one workgroup takes P query positions x all G query heads of one kv head:
the GEMM M dimension is P*G rows (row r = position r // G, head r % G), and each K/V tile is loaded once for all of them.
Q can stay in shared memory or go to registers (frees LDS for a larger key tile; register-A GEMM verified exact on
gfx90a by fa_diag.py).

Layout: Q [B, L, H, D], K/V [B, L, Hkv, D], H = Hkv * G, fp16 in/out, fp32 accumulate. Causal or not.
Usage: python gqa_fa.py [L [cfg D [B [G]]]]   (correctness vs float64 + speed vs torch SDPA and vs the upstream example's best tile). cfg: JSON tile, '-' for the default list, 'auto' for d128_tile, d256_tile or mha_tile. G 1 is MHA.

gqa_flash_fast is the same kernel with C edits on the generated HIP source (see evidence/N5/RESULTS.md; none of this is expressible in TileLang 0.1.15):
- one load block per K step, 2 rows of K and 2 of V per lane, 16 bytes each; next step's rows are loaded to registers before the V GEMM, so they overlap it
- two barriers per K step: store K, V | barrier | read K, V | barrier. race_free() checks that every write->read and read->write of an LDS array crosses a barrier
- row sum reduced once after the K loop (exact up to summation order); O rescale skipped when sf == 1 (exact)
- vt=True (bN 32): V stored to LDS transposed, so the PV GEMM reads 8 bytes per MFMA operand instead of four 2-byte reads (_vt_edit; d256_tile turns it on)
Form JTP (D 128, d128_tile on clang 22, B 1, L 1024 to 16384, L % 64 == 0, except L in _D128_JTP_NOT) is form J with the same transposed V plus a register prefetch of the next K/V step;
d128_tile returns it as form JTPR, the same source with the P cast by v_cvt_pkrtz_f16_f32 (round toward zero). Form M4HL/M4HLR (32x32x8 MFMA) is kept for gqa_flash_j but not dispatched.
"""
import json
import os
import re
import subprocess
import sys

import torch
import torch.nn.functional as F
import tilelang
import tilelang.rocm.language as T
from tilelang.tileop.base import GemmWarpPolicy


def make(B, L, Hkv, G, D, causal, P, bN, threads, q_regs, k_pack=2, edge=False, rev=False, recip=False, swap=False, rows=0):
    """rows != 0: rows per block need not be a multiple of G. The flat row r = position * G + head is split into blocks of `rows`, so P is ignored and the number of waves
    (rows / 16) can be chosen freely; a block may start in the middle of a position's heads."""
    H = Hkv * G
    bM = rows if rows else P * G
    scale = (1.0 / D) ** 0.5
    dtype, acc = T.float16, T.float32
    NB = -(-L * G // rows) if rows else -(-L // P)
    has_edge = causal or L % bN != 0
    d0, d1 = (B * Hkv, NB) if swap else (NB, B * Hkv)

    @T.prim_func
    def main(Q: T.Tensor((B, L, H, D), dtype), K: T.Tensor((B, L, Hkv, D), dtype), V: T.Tensor((B, L, Hkv, D), dtype),
             O: T.Tensor((B, L, H, D), dtype)):
        with T.Kernel(d0, d1, threads=threads) as (b0, b1):
            bx = b1 if swap else b0
            byz = b0 if swap else b1
            b = byz // Hkv
            hk = byz % Hkv
            if rows:
                r0 = (NB - 1 - bx) * bM if rev else bx * bM
                p0 = r0 // G
            else:
                p0 = (NB - 1 - bx) * P if rev else bx * P
            K_s = T.alloc_shared((bN, D), dtype)
            V_s = T.alloc_shared((bN, D), dtype)
            acc_o = T.alloc_fragment((bM, D), acc)
            acc_s = T.alloc_fragment((bM, bN), acc)
            p_cast = T.alloc_fragment((bM, bN), dtype)
            m_i = T.alloc_fragment((bM,), acc)
            m_prev = T.alloc_fragment((bM,), acc)
            l_i = T.alloc_fragment((bM,), acc)
            row_sum = T.alloc_fragment((bM,), acc)
            sf = T.alloc_fragment((bM,), acc)
            if recip:
                rl = T.alloc_fragment((bM,), acc)
            if q_regs:
                Q_x = T.alloc_fragment((bM, D), dtype)
            else:
                Q_x = T.alloc_shared((bM, D), dtype)
            if rows:
                for i, d in T.Parallel(bM, D):
                    Q_x[i, d] = T.if_then_else((r0 + i) // G < L, Q[b, (r0 + i) // G, hk * G + (r0 + i) % G, d], T.cast(0, dtype))
            else:
                for i, d in T.Parallel(bM, D):
                    Q_x[i, d] = T.if_then_else(p0 + i // G < L, Q[b, p0 + i // G, hk * G + i % G, d], T.cast(0, dtype))
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(acc))
            T.fill(l_i, 0)
            if edge:
                T.clear(acc_s)
            if rows:
                n_kv = T.ceildiv(T.min((r0 + bM - 1) // G + 1, L), bN) if causal else T.ceildiv(L, bN)
            else:
                n_kv = T.ceildiv(T.min(p0 + P, L), bN) if causal else T.ceildiv(L, bN)
            for k in T.serial(n_kv):
                kv0 = k * bN
                T.copy(K[b, kv0:kv0 + bN, hk, :], K_s)
                T.copy(V[b, kv0:kv0 + bN, hk, :], V_s)
                if not edge:                                     # mask written before the GEMM on every tile
                    if causal and rows:
                        for i, j in T.Parallel(bM, bN):
                            acc_s[i, j] = T.if_then_else(((r0 + i) // G >= kv0 + j) & (kv0 + j < L), 0, -T.infinity(acc))
                    elif causal:
                        for i, j in T.Parallel(bM, bN):
                            acc_s[i, j] = T.if_then_else((p0 + i // G >= kv0 + j) & (kv0 + j < L), 0, -T.infinity(acc))
                    else:
                        for i, j in T.Parallel(bM, bN):
                            acc_s[i, j] = T.if_then_else(kv0 + j < L, 0, -T.infinity(acc))
                T.gemm(Q_x, K_s, acc_s, transpose_B=True, clear_accum=edge, k_pack=k_pack, policy=GemmWarpPolicy.FullRow)
                if edge and has_edge:                            # mask only on tiles the diagonal or the end of L crosses
                    if causal:
                        if kv0 + bN - 1 > p0:
                            if rows:
                                for i, j in T.Parallel(bM, bN):
                                    acc_s[i, j] = T.if_then_else(((r0 + i) // G >= kv0 + j) & (kv0 + j < L), acc_s[i, j], -T.infinity(acc))
                            else:
                                for i, j in T.Parallel(bM, bN):
                                    acc_s[i, j] = T.if_then_else((p0 + i // G >= kv0 + j) & (kv0 + j < L), acc_s[i, j], -T.infinity(acc))
                    else:
                        if kv0 + bN > L:
                            for i, j in T.Parallel(bM, bN):
                                acc_s[i, j] = T.if_then_else(kv0 + j < L, acc_s[i, j], -T.infinity(acc))
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for i in T.Parallel(bM):
                    m_i[i] = T.max(m_i[i], m_prev[i])
                for i in T.Parallel(bM):
                    sf[i] = T.exp(m_prev[i] * scale - m_i[i] * scale)
                    l_i[i] *= sf[i]
                for i, d in T.Parallel(bM, D):
                    acc_o[i, d] *= sf[i]
                for i, j in T.Parallel(bM, bN):
                    acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)
                T.reduce_sum(acc_s, row_sum, dim=1)
                for i in T.Parallel(bM):
                    l_i[i] += row_sum[i]
                T.copy(acc_s, p_cast)
                T.gemm(p_cast, V_s, acc_o, policy=GemmWarpPolicy.FullRow)
            if rows and recip:
                for i in T.Parallel(bM):
                    rl[i] = 1.0 / T.if_then_else(l_i[i] > 0, l_i[i], 1.0)
                for i, d in T.Parallel(bM, D):
                    if (r0 + i) // G < L:
                        O[b, (r0 + i) // G, hk * G + (r0 + i) % G, d] = acc_o[i, d] * rl[i]
            elif rows:
                for i, d in T.Parallel(bM, D):
                    if (r0 + i) // G < L:
                        O[b, (r0 + i) // G, hk * G + (r0 + i) % G, d] = acc_o[i, d] / T.if_then_else(l_i[i] > 0, l_i[i], 1.0)
            elif recip:
                for i in T.Parallel(bM):
                    rl[i] = 1.0 / T.if_then_else(l_i[i] > 0, l_i[i], 1.0)
                for i, d in T.Parallel(bM, D):
                    if p0 + i // G < L:
                        O[b, p0 + i // G, hk * G + i % G, d] = acc_o[i, d] * rl[i]
            else:
                for i, d in T.Parallel(bM, D):
                    if p0 + i // G < L:
                        O[b, p0 + i // G, hk * G + i % G, d] = acc_o[i, d] / T.if_then_else(l_i[i] > 0, l_i[i], 1.0)

    return main


def _block_end(s, i):
    d = 0
    j = s.index("{", i)
    while True:
        if s[j] == "{":
            d += 1
        elif s[j] == "}":
            d -= 1
            if d == 0:
                return j + 1
        j += 1


def _swz(r, c, n1, n2):
    """C expression of TileLang's swizzled LDS index of element (r, c) of a tile (n1, n2), read off its generated code."""
    C = min(64, n2 & -n2)
    e = f"((({c}) / {C}) * {C * n1}) + ((({r}) >> 4) * {16 * C}) + ((({r}) & 15) * {C})"
    top = C.bit_length() - 2
    rb = 2
    for b in range(top, 2, -1):
        e += f" + (((((({c}) >> {b}) ^ (({r}) >> {rb})) & 1)) * {1 << b})"
        rb -= 1
    return "(" + e + f" + (({c}) & 7))"


def _lds_events(s):
    k = s.index("for (int k = 0;")
    ev = []
    for ln in s[k:_block_end(s, k)].split("\n"):
        if "__syncthreads" in ln:
            ev.append(("B", ""))
            continue
        for a in ("K_s", "V_s"):
            if "half_t*)" + a in ln:
                ev.append(("W" if ("half_t*)" + a) in ln.split(" = ")[0] else "R", a))
    return ev


def race_free(s):
    """Every write->read and read->write of one LDS array, cyclically over the K step, must cross a barrier. Threads write disjoint elements."""
    ev = _lds_events(s)
    bad = set()
    for i, (t, a) in enumerate(ev):
        if t == "B":
            continue
        for d in range(1, len(ev) + 1):
            t2, a2 = ev[(i + d) % len(ev)]
            if t2 == "B":
                break
            if a2 == a and t2 != t:
                bad.add(f"{t}({a})->{t2}({a2})")
                break
    return sorted(bad)


def _fast_edit(s, c):
    """Applies the N5 edits to generated source. c: L, Hkv, B, D, bN, threads, swap."""
    L, Hkv, D, bN, th = c["L"], c["Hkv"], c["D"], c["bN"], c["threads"]
    by = "(int)blockIdx.x" if c["swap"] else "(int)blockIdx.y"
    k_loop = s.index("for (int k = 0;")
    n_kv = re.match(r"for \(int k = 0; k < (.*); \+\+k\) ", s[k_loop:s.index("{", k_loop)]).group(1)

    def span(a):
        """Span of the load of array a: its '#pragma unroll' block, or the statement when the copy has one step and no loop."""
        i = k_loop
        for ln in s[k_loop:].split("\n"):
            if " = " in ln and ("half_t*)" + a) in ln.split(" = ")[0]:
                i += ln.index("half_t*)" + a)
                break
            i += len(ln) + 1
        else:
            raise ValueError("no store to " + a)
        p = s.rfind("#pragma unroll", k_loop, i)
        if p >= 0 and _block_end(s, p) > i:
            return p, _block_end(s, p)
        return s.rfind("\n", 0, i) + 1, s.index("\n", i) + 1

    def row(A, w, kx):
        r = f"(({kx}) * {bN}) + kk_v + {w}"
        if c["B"] == 1:
            return f"*(uint4*)({A} + ((((({r}) * {Hkv}) + ({by})) * {D}) + d0_v))"
        return f"*(uint4*)({A} + (((((({by}) / {Hkv}) * {L} + ({r})) * {Hkv} + (({by}) % {Hkv})) * {D}) + d0_v))"

    cpr = D // 8
    slots = bN // 2 * cpr
    if slots > th:
        raise ValueError(f"one load step per thread needed: {slots} chunks, {th} threads")
    names = [("k0_v", "K", 0), ("k1_v", "K", 1), ("v0_v", "V", 0), ("v1_v", "V", 1)]
    loads = lambda kx: "\n".join(f"      {n} = {row(A, w, kx)};" for n, A, w in names)  # noqa: E731
    stores = "\n".join(f"      *(uint4*)(((half_t*){a}) + {_swz(f'kk_v + {w}', 'd0_v', bN, D)}) = {n};" for n, a, w in (("k0_v", "K_s", 0), ("k1_v", "K_s", 1), ("v0_v", "V_s", 0), ("v1_v", "V_s", 1)))
    p, e = span("K_s")
    s = s[:p] + s[e:]
    p, e = span("V_s")
    s = s[:p] + f"""
    if ((c_v < {slots}) && (((k * {bN}) + kk_v) < {L})) {{
{stores}
    }}
""" + s[e:]
    # next step's rows, before the V GEMM
    i = s.index("half_t B_local_1[", k_loop)
    j = s.rfind("{", 0, i)
    s = s[:j] + f"""
    if (((k + 1) < ({n_kv})) && (c_v < {slots}) && ((((k + 1) * {bN}) + kk_v) < {L})) {{
{loads("k + 1")}
    }}
    """ + s[j:]
    decl = f"""
  int c_v = (int)threadIdx.x;
  int kk_v = (c_v / {cpr}) * 2;
  int d0_v = (c_v & {cpr - 1}) * 8;
""" + "".join(f"  uint4 {n} = make_uint4(0, 0, 0, 0);\n" for n, _, _ in names) + f"""
  if ((0 < ({n_kv})) && (c_v < {slots}) && (kk_v < {L})) {{
{loads("0")}
  }}
  """
    s = s[:k_loop] + decl + s[k_loop:]
    # barriers
    k_loop = s.index("for (int k = 0;")
    e = _block_end(s, k_loop)
    body = s[k_loop:e].replace("__syncthreads();\n", "")
    j = body.rfind("{", 0, body.index("half_t B_local["))
    body = body[:j] + "__syncthreads();\n    " + body[j:]
    s = s[:k_loop] + body[:-1] + "  __syncthreads();\n  }" + s[e:]
    # row sum once after the K loop
    m = re.search(r"row_sum\[0\] = tl::AllReduce<tl::SumOp, (\d+), (\d+), 0>::run\(row_sum\[0\], [^;]*;\n", s)
    if m:
        s = s[:m.start()] + s[m.end():]
        k_loop = s.index("for (int k = 0;")
        e = _block_end(s, k_loop)
        s = s[:e] + f"\n  l_i[0] = tl::AllReduce<tl::SumOp, {m.group(1)}, {m.group(2)}, 0>::run(l_i[0], (float*)nullptr);\n" + s[e:]
    # no O rescale when sf == 1
    i = s.find("l_i[0] = (l_i[0] * sf[0]);\n")
    if i >= 0:
        p = s.index("#pragma unroll", i)
        e = _block_end(s, p)
        if "acc_o" in s[p:e]:
            s = s[:p] + "if (sf[0] != 1.0f) {\n" + s[p:e] + "\n}\n" + s[e:]
    n_w = sum(1 for t, _ in _lds_events(s) if t == "W")
    if n_w != 4:
        raise RuntimeError(f"gqa_flash_fast: expected 4 LDS stores in the K step, found {n_w}")
    bad = race_free(s)
    if bad:
        raise RuntimeError(f"gqa_flash_fast: LDS accesses not separated by a barrier: {bad}")
    return s


def _vt_ev(expr, **v):
    return eval(expr.replace("(int)threadIdx.x", "T").replace("/", "//"), {}, v)  # noqa: S307  (generated index arithmetic only)


def _vt_edit(s, c):
    """Transposed V for the fast path, bN 32 only (PREFILL-MAX round 1): V goes to LDS as [d][key pair] dwords at d * 16 + (c ^ 2 ((d >> 1) & 7)), so the PV GEMM's
    A operand is one 8-byte read per MFMA instead of four 2-byte reads. V's global loads get their own thread map (key pair fastest). Applied after _fast_edit;
    the generated V read index is checked numerically against the assumed (d, key) map, and anything else raises."""
    bN, D, L = c["bN"], c["D"], c["L"]
    if bN != 32 or L % bN:
        raise ValueError("vt: bN 32 and L % 32 == 0 only")
    npair = bN // 2
    m = re.search(r"( *)for \(int j_2 = 0; j_2 < (\d+); \+\+j_2\) \{\n *for \(int local_id = 0; local_id < 4; \+\+local_id\) \{\n *B_local_1\[\(\(j_2 \* 4\) \+ local_id\)\] = \(\(half_t\*\)V_s\)\[(.*)\];\n *\}\n *\}\n", s)
    if not m or int(m.group(2)) != D // 16:
        raise RuntimeError("vt: V read nest not found")
    idx, swz = m.group(3), _swz("(R)", "(Cc)", bN, D)
    for T in (0, 1, 7, 15, 16, 37, 63, 64, 100, 511):
        for ki_1 in range(bN // 16):
            for j_2 in range(D // 16):
                for lid in range(4):
                    d, key = 16 * j_2 + (T & 15), 16 * ki_1 + 4 * ((T & 63) >> 4) + lid
                    if _vt_ev(idx, T=T, ki_1=ki_1, j_2=j_2, local_id=lid) != _vt_ev(swz, R=key, Cc=d):
                        raise RuntimeError(f"vt: read map differs at T {T} ki {ki_1} j {j_2} l {lid}")
    ind = m.group(1)
    new = (f"{ind}int vr_t = (int)threadIdx.x;\n{ind}asm volatile(\"\" : \"+v\"(vr_t));\n{ind}const int vr_l = vr_t & 15;\n"
           f"{ind}const int vr_b = (vr_l * {npair}) + (((ki_1 * 8) + (((vr_t & 63) >> 4) * 2)) ^ (((vr_l >> 1) & 7) << 1));\n"
           f"{ind}#pragma unroll\n{ind}for (int j_2 = 0; j_2 < {D // 16}; ++j_2) {{\n"
           f"{ind}  *(uint2*)(B_local_1 + (j_2 * 4)) = *(uint2*)(((uint*)V_s) + (vr_b + (j_2 * {16 * npair})));\n{ind}}}\n")
    s = s[:m.start()] + new + s[m.end():]
    n = 0
    for w in (0, 1):
        s, k = re.subn(r"(v%d_v = \*\(uint4\*\)\(V \+ .*?)kk_v( \+ %d\).*?)d0_v(\)\);)" % (w, w), r"\1kv_v\2dv_v\3", s)
        n += k
    if n != 4:
        raise RuntimeError(f"vt: expected 4 V loads, found {n}")
    s = s.replace("  int d0_v = (c_v & ", f"  int kv_v = (c_v % {npair}) * 2;\n  int dv_v = (c_v / {npair}) * 8;\n  int d0_v = (c_v & ", 1)
    if "int kv_v" not in s:
        raise RuntimeError("vt: map declaration not found")
    lines = [ln for ln in s.split("\n") if "((half_t*)V_s) +" in ln and "= v" in ln]
    if len(lines) != 2:
        raise RuntimeError(f"vt: expected 2 V stores, found {len(lines)}")
    ind = lines[0][:len(lines[0]) - len(lines[0].lstrip())]
    st = (f"{ind}{{\n{ind}  int vt_t = (int)threadIdx.x;\n{ind}  asm volatile(\"\" : \"+v\"(vt_t));\n"
          f"{ind}  const int vt_c = vt_t % {npair};\n{ind}  const int vt_d = (vt_t / {npair}) * 8;\n"
          f"{ind}  const uint vw[4] = {{v0_v.x, v0_v.y, v0_v.z, v0_v.w}};\n{ind}  const uint vh[4] = {{v1_v.x, v1_v.y, v1_v.z, v1_v.w}};\n"
          f"{ind}  #pragma unroll\n{ind}  for (int e = 0; e < 8; ++e) {{\n{ind}    const int d = vt_d + e;\n"
          f"{ind}    const uint lo = (e & 1) ? (vw[e >> 1] >> 16) : (vw[e >> 1] & 0xffffu);\n"
          f"{ind}    const uint hi = (e & 1) ? (vh[e >> 1] & 0xffff0000u) : (vh[e >> 1] << 16);\n"
          f"{ind}    ((uint*)V_s)[(d * {npair}) + (vt_c ^ (((d >> 1) & 7) << 1))] = lo | hi;\n{ind}  }}\n{ind}}}")
    s = s.replace(lines[0] + "\n", st + "\n", 1).replace(lines[1] + "\n", "", 1)
    if "((half_t*)V_s)" in s:
        raise RuntimeError("vt: half_t V_s access left")
    return s


_FAST = {}          # context of the fast kernel being compiled. Set only for the duration of a gqa_flash_fast call (try/finally), so a disk-cache hit cannot leave it set for a later plain kernel.


def _install_hook():
    from tilelang.contrib import hipcc
    if getattr(hipcc.compile_hip, "_gqa_fast", False):
        return
    orig = hipcc.compile_hip

    def comp(code, *a, **k):
        if _J:                                   # form J: the stub's device code is replaced by the J source, compiled with the wheel's own options
            if "K_s" not in code or "acc_s" in code:
                raise RuntimeError("gqa_flash_j: hook reached a kernel that is not the J stub")
            return orig(_j_source(_J["L"], _J.get("form", "J")), *a, **k)
        if _FAST:
            code = _fast_edit(code, _FAST)
            if _FAST.get("vt"):
                code = _vt_edit(code, _FAST)
            if _FAST.get("lb"):
                code, n = re.subn(r"__launch_bounds__\((\d+)\)", lambda m: f"__launch_bounds__({m.group(1)}, {_FAST['lb']})", code)
                if n != 1:
                    raise RuntimeError("gqa_flash_fast: launch bounds not found")
            if _FAST.get("sched"):               # same switch as evidence/N5/src4/flagsets4.sh, so what landed is what was timed
                prev = os.environ.get("HIPCC_COMPILE_FLAGS_APPEND")
                os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = ((prev + " ") if prev else "") + "-mllvm -misched=" + _FAST["sched"]
                try:
                    return orig(code, *a, **k)
                finally:
                    if prev is None:
                        del os.environ["HIPCC_COMPILE_FLAGS_APPEND"]
                    else:
                        os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = prev
        return orig(code, *a, **k)
    comp._gqa_fast = True
    hipcc.compile_hip = comp


@tilelang.jit(out_idx=[3], compile_flags=["-DGQA_FAST_N5_V2"])     # flag: ignored on the HIP path, keeps fast and plain kernels apart in the cache. V2: entries from V1 may hold a stale-context edit.
def _gqa_flash_fast_jit(B, L, Hkv, G, D, causal, P, bN, threads, k_pack, edge, rev, recip, swap, rows):
    return make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows)


_CLANG = []


def clang_major():
    """Major version of the clang behind hipcc, the compiler kernels are built with (18 on the LUMI stack, 22 on the vLLM image)."""
    if not _CLANG:
        from tilelang.contrib import hipcc
        out = subprocess.run([hipcc.find_rocm_path() + "/bin/hipcc", "--version"], capture_output=True, text=True).stdout
        _CLANG.append(int(re.search(r"clang version (\d+)", out).group(1)))
    return _CLANG[0]


def _check_tuning(G, P, threads, rows, lb, sched):
    """lb and sched were only gains on clang 18 at these tiles. lb = 4 at 12 waves loses 22 to 39% and clang 22 is flat to negative (evidence/N5/RESULTS.md sections 13, 14)."""
    if not lb and not sched:
        return
    if clang_major() != 18:
        raise ValueError(f"gqa_flash_fast: lb and sched are for clang 18, this is clang {clang_major()}")
    if threads > 640 or not (rows == 128 or (G == 1 and not rows and P in (128, 160))):
        raise ValueError(f"gqa_flash_fast: lb only at rows 128 or MHA P128, P160, up to 10 waves; got G {G} P {P} rows {rows} threads {threads}")
    if sched and not (G == 1 and not rows):
        raise ValueError("gqa_flash_fast: sched only at MHA tiles")


def mha_tile(L, B=1, Hkv=8):
    """Tile for G = 1: keyword arguments for gqa_flash_fast. A lookup of races, not a model: P160 wins only at B 1, L 2048 and 3584 to 4320, and at B 2, L 2048; P128 everywhere else, because P128 stays near 0.92 to 1.03 of CK where P160 ranges from 0.68 to 1.25.
    Timed P128 against P160 in one process on both stacks at L 512 to 4096 and at 2 x 1024, 2 x 2048, 4 x 512, 4 x 1024, 16 x 1024, 2 x 8192 (evidence/N5/RESULTS.md section 15; Hkv 8, randn0; those batches also on ar and big on clang 22). 4160, 4320 and clang 22 lengths from 4608 up are section 14; other L and B are untimed and get P128. On clang 18 add lb = 4 and sched gcn-max-ilp."""
    P = 160 if Hkv == 8 and ((B == 1 and (L == 2048 or 3584 <= L <= 4320)) or (B == 2 and L == 2048)) else 128
    t = dict(P=P, bN=32, threads=P * 4, rev=False, swap=False, recip=True)
    if clang_major() == 18:
        t.update(lb=4, sched="gcn-max-ilp")
    return t


_D128_R128 = {22: (1024, 2048, 2560, 2816, 3072, 3584, 4096, 5000, 5120), 18: (1024, 2560)}
_D128_R256 = {22: (6144, 8192, 10240, 12288, 16384), 18: ()}
_D128_J = (2816, 3584)        # form J, clang 22, B 1, L % 64 == 0 (gate in RESEARCH-LOG JPORT entry). Superseded by JTP below; kept for gqa_flash_j(L, "J").
_D128_JTP = (1024, 16384)     # form JTP, clang 22, B 1, L % 64 == 0 (PREFILL-MAX round 1; races at 1024-16384, RESEARCH-LOG)
_D128_JTP_NOT = (1088, 1536, 3200)   # no JTP win in every round-3 job (PREFILL-MAX round 3): 1088 loses (0.959-1.021), 1536 and 3200 win by less than one job's identical-code spread
_D256_VT_NOT = (1056, 5152)          # vt not shown above one job's identical-code spread there (PREFILL-MAX round 3)
_D128_PK = (1024, 16384)     # form JTPR (JTP + pkrtz P cast) replaces JTP over this range (PREFILL-MAX round 4: won at all 22 timed lengths, 4 inputs, 2 opposite-order jobs)
# Not dispatched (PREFILL-MAX round 4, RESEARCH-LOG): form M4HLR (loses to JTPR on ar by more than the noise at every timed length 8192-16384) and pk on the d256 vt path (0.970-1.003 of vt).
_D128_TIMED = (1024, 1536, 2048, 2560, 2816, 3072, 3584, 4096, 4160, 5000, 5120, 6144, 8192, 10240, 12288, 16384)


def d128_tile(L, B=1):
    """Tile for D 128, G 6, Hkv 4, causal: keyword arguments for gqa_flash_fast. rows 128 bN32, rows 256 bN32 or P32 bN96, whichever won the race at the timed length nearest L, per compiler (clang 22 and clang 18 differ). The table is a race result, not a model: rows 128 has a hole at 4160 on clang 22.
    Timed at the lengths in _D128_TIMED, B 1, randn0 (evidence/N5/RESULTS.md sections 15, 16; rows 256 raced only at 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384 on clang 22 and not picked on clang 18). Other L and B > 1 are untimed. On clang 22 this hand-edited path beats CK only up to L 2560; 2816 is a tie and 3072 up is a loss.
    Form J (fork-main's rows 128 bN64 J source, through gqa_flash_j) is returned with form "J" on clang 22, B 1, L 2816 to 3584 with L % 64 == 0 only: over the old pick 1.030 to 1.049 at 2816, 3072, 3328, 3584
    on six inputs in 2 jobs of opposite order, over CK 0.995 to 1.040 (JPORT, jobs 23561796, 23561797).
    Form JTP (form J with V stored to LDS transposed and next step's K/V loads issued before the PV GEMM, _jtp_source) replaces J and is returned with form "JTP" on clang 22, B 1,
    L 1024 to 16384 with L % 64 == 0, except L in _D128_JTP_NOT: over the previous dispatch 1.011 to 1.126 at 13 lengths from 1024 to 16384, 3 inputs, 2 jobs of opposite order (PREFILL-MAX round 1);
    round 3 (2 pairs of opposite-order jobs with an identical-code arm, 3 inputs): wins at every L % 64 == 0 from 1024 to 2048 except 1088 and 1536, every one from 2816 to 3584 except 3200, and at 2240, 4608, 5632, 7168, 9216,
    10240, 11264, 13312, 14336, 15360, 16320; fp64 check at all 241 dispatched lengths. Other lengths in the range are interpolated between those. The excluded lengths keep the previous dispatch (J at 3200).
    Form JTPR (JTP with the P cast by v_cvt_pkrtz, decision 9 Nov on its error) is returned instead of JTP over _D128_PK (PREFILL-MAX round 4): over JTP 1.005 to 1.020 in every cell at 22 lengths
    from 1024 to 16384 (13 below 8192, 9 multiples of 1024 from 8192), inputs randn0, big, ramp, ar, 2 jobs of opposite order with an identical-code arm (noise 0.36% and 0.63%); fp64 check at every
    dispatched length (rel at most 4.7e-4, CK's level). Form M4HLR (32x32x8 MFMA, gqa_flash_j(L, "M4HLR")) is not dispatched: it loses to JTPR on ar (0.982 to 0.991) at every timed length from 8192.
    Clang 18, B > 1 and other L keep the table below."""
    c = clang_major()
    if c == 22 and B == 1 and _D128_JTP[0] <= L <= _D128_JTP[1] and L % 64 == 0:
        if L not in _D128_JTP_NOT:
            f = "JTPR" if _D128_PK[0] <= L <= _D128_PK[1] else "JTP"
            return dict(P=0, bN=64, threads=512, rows=128, rev=True, swap=True, recip=True, form=f)
        if _D128_J[0] <= L <= _D128_J[1]:
            return dict(P=0, bN=64, threads=512, rows=128, rev=True, swap=True, recip=True, form="J")
    near = min(_D128_TIMED, key=lambda x: (abs(x - L), x))
    if near in _D128_R256.get(c, _D128_R256[22]):
        t = dict(P=0, bN=32, threads=1024, rows=256, rev=True, swap=True, recip=True)
    elif near in _D128_R128.get(c, _D128_R128[22]):
        t = dict(P=0, bN=32, threads=512, rows=128, rev=True, swap=True, recip=True)
        if c == 18:
            t["lb"] = 4                # +3 to +6% at 1024 and 2560, same process (job 23525788)
    else:
        t = dict(P=32, bN=96, threads=768, rev=True, swap=True, recip=True)
    return t


def d256_tile(L, B=1):
    """Tile for D 256, G 6, Hkv 4, causal: keyword arguments for gqa_flash_fast. P32 bN32 768 threads (the tile in the CLI's list), with vt=True (transposed V)
    on clang 22, B 1, L 1024 to 16384 with L % 32 == 0, except L in _D256_VT_NOT: over the same tile without vt 1.016 to 1.058 at L 1024, 2048, 4096, 8192, 16384, 3 inputs,
    2 jobs of opposite order (PREFILL-MAX round 1); round 3 (identical-code arm, 4 inputs, 2 jobs each) 1.020 to 1.059 at 1088, 1568, 2080, 2592, 3104, 4128, 6176, 7200, 8224, 10240, 10272, 12320, 14368, fp64 check at 17 more lengths.
    Elsewhere vt=False. pk (the pkrtz P cast) is not returned: with vt it read 0.970 to 1.003 of vt alone at 9 lengths (PREFILL-MAX round 4). No served caller uses d256_tile (only this module's CLI)."""
    t = dict(P=32, bN=32, threads=768, rev=True, swap=True, recip=True)
    if clang_major() == 22 and B == 1 and 1024 <= L <= 16384 and L % 32 == 0 and L not in _D256_VT_NOT:
        t["vt"] = True
    return t


def gqa_flash_fast(B, L, Hkv, G, D, causal, P, bN, threads, k_pack=2, edge=True, rev=True, recip=False, swap=False, rows=0, lb=0, sched="", form="", vt=False):
    """Plain wrapper, not a jit: it binds the edit to this one compile. Not thread-safe with a concurrent compile in the same process.
    lb: second argument of __launch_bounds__, the occupancy the compiler targets (0 = none). sched: LLVM -misched value. Both checked by _check_tuning.
    form "J", "JTP", "JTPR", "M4HL" or "M4HLR": gqa_flash_j, only for the exact tile d128_tile returns with it (512 threads for the J forms, 256 for the M forms).
    vt: transposed V in LDS (_vt_edit), bN 32 and L % 32 == 0 only; compiled with its own cache flag."""
    if form in _J_FORMS:
        if vt or (B, Hkv, G, D, causal, P, bN, threads, rows, rev, swap, recip, lb, sched) != (1, 4, 6, 128, True, 0, 64, _J_FORMS[form], 128, True, True, True, 0, ""):
            raise ValueError(f"gqa_flash_fast: form {form} covers B 1, Hkv 4, G 6, D 128, causal, rows 128, bN 64, {_J_FORMS[form]} threads, rev, swap, recip only")
        return gqa_flash_j(L, form)
    if form:
        raise ValueError(f"gqa_flash_fast: unknown form {form!r}")
    if L % 2 or bN % 2 or not edge:
        raise ValueError("gqa_flash_fast needs even L and bN and the edge-only mask")
    _check_tuning(G, P, threads, rows, lb, sched)
    _install_hook()
    _FAST.clear()
    _FAST.update(L=L, Hkv=Hkv, B=B, D=D, bN=bN, threads=threads, swap=swap, lb=lb, sched=sched, vt=vt)
    try:
        if vt:
            if lb or sched:
                raise ValueError("gqa_flash_fast: vt was raced without lb and sched")
            f = make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows)
            return tilelang.compile(f, out_idx=[3], compile_flags=["-DGQA_FAST_N5_V2", "-DGQA_VT_1"])
        if lb or sched:
            f = make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows)
            flags = ["-DGQA_FAST_N5_V2", f"-DGQA_LB_{lb}"] + ([f"-DGQA_SCHED_{sched.replace('-', '_')}"] if sched else [])
            return tilelang.compile(f, out_idx=[3], compile_flags=flags)
        return _gqa_flash_fast_jit(B, L, Hkv, G, D, causal, P, bN, threads, k_pack, edge, rev, recip, swap, rows)
    finally:
        _FAST.clear()


# Form J (rows 128, bN 64, 512 threads, K/V staged through registers, row sum once after the K loop, reciprocal, reversed and swapped grid) for D 128, G 6, Hkv 4, causal, B 1.
# Source: TileLang on fork-main (patches 0001 to 0007) generated this HIP at L 3072 (evidence/D128/bytes/d128_3072_r128_b64_t512_JDRF.cc); the wheel cannot generate it.
# Only two constants depend on L: @R0@ = 6 L - 128 (first flat row of block 0) and @NK@ = 6 L + 383 (K-loop bound numerator); equal to fork-main's own output at 2816, 3584, 4096, 8192, 16384 [observed, text compare].
# The RegVec definition is patch 0007's, added after the includes so the wheel's headers compile it (same instruction stream as fork-main, evidence/D128 freeze3).
# Valid only for L % 64 == 0: the source has no bounds check on rows or keys.
_J_SRC = r"""#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/scan.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>


namespace tl {
template <class S> struct RegVec;
template <int N> struct RegVecBase { typedef uint Vec __attribute__((ext_vector_type(N))); Vec v; };
#define TL_DEFINE_REG_VEC(S, N) template <> struct RegVec<S> : RegVecBase<N> { TL_DEVICE operator S() const { return __builtin_bit_cast(S, v); } TL_DEVICE RegVec &operator=(const S &x) { v = __builtin_bit_cast(Vec, x); return *this; } TL_DEVICE RegVec &operator=(const RegVec &x) { v = x.v; return *this; } };
TL_DEFINE_REG_VEC(uint2, 2)
TL_DEFINE_REG_VEC(uint4, 4)
#undef TL_DEFINE_REG_VEC
}
extern "C" __global__ void __launch_bounds__(512) main_kernel(half_t* __restrict__ K, half_t* __restrict__ O, half_t* __restrict__ Q, half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* K_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* V_s = ((void*)((char*)buf_dyn_shmem + 16384));
  half_t Q_x[32];
  float acc_o[32];
  float m_i[1];
  float l_i[1];
  float l_r[1];
  float acc_s[16];
  half_t K_r[16];
  half_t V_r[16];
  float m_prev[1];
  float m_i_clear[1];
  float sf[1];
  half_t p_cast[16];
  float rl[1];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(Q_x + (i * 8)) = *(tl::RegVec<uint4>*)(Q + ((((((((((((((int)threadIdx.x) >> 6) * 16) + (((int)threadIdx.x) & 15)) + @R0@) - (((int)blockIdx.y) * 128)) / 6) * 3072) + (((int)blockIdx.x) * 768)) + ((6 & (((((((((int)threadIdx.x) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 4) - (((int)blockIdx.y) * 128)) % 6) >> 31)) * 128)) + (((((((((int)threadIdx.x) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 4) - (((int)blockIdx.y) * 128)) % 6) * 128)) + (i * 32)) + (((((int)threadIdx.x) & 63) >> 4) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float broadcast_var = 0.000000e+00f;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  m_i[0] = -HUGE_VALF;
  l_i[0] = 0.000000e+00f;
  l_r[0] = 0.000000e+00f;
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    float broadcast_var_1 = 0.000000e+00f;
    *(float4*)(acc_s + (i_2 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  for (int k = 0; k < ((@NK@ - (((int)blockIdx.y) * 128)) / 384); ++k) {
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      *(tl::RegVec<uint4>*)(K_r + (i_3 * 8)) = *(tl::RegVec<uint4>*)(K + (((((k * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (i_3 * 512)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      *(tl::RegVec<uint4>*)(V_r + (i_4 * 8)) = *(tl::RegVec<uint4>*)(V + (((((k * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (i_4 * 512)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      *(tl::RegVec<uint4>*)(((half_t*)K_s) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + (i_5 * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((i_5 + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(tl::RegVec<uint4>*)(K_r + (i_5 * 8));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      *(tl::RegVec<uint4>*)(((half_t*)V_s) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + ((((int)threadIdx.x) >> 4) * 128)) + (i_6 * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((i_6 + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(tl::RegVec<uint4>*)(V_r + (i_6 * 8));
    }
    {
      half_t B_local[32];
      #pragma unroll
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        float broadcast_var_2 = 0.000000e+00f;
        *(float4*)(acc_s + (i_7 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
      }
      __syncthreads();
      for (int ki = 0; ki < 4; ++ki) {
        for (int j = 0; j < 4; ++j) {
          *(uint4*)(B_local + (j * 8)) = *(tl::RegVec<uint4>*)(((half_t*)K_s) + (((((((ki >> 1) * 4096) + (j * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        }
        for (int kp = 0; kp < 2; ++kp) {
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            {
      *(((float32x4*)acc_s) + j_1) = __builtin_amdgcn_mfma_f32_16x16x16f16(*(((float16x4*)B_local) + ((j_1 * 2) + kp)),
                    *(((float16x4*)Q_x) + ((ki * 2) + kp)),
                    *(((float32x4*)acc_s) + j_1), 0, 0, 0);
    };
          }
        }
      }
    }
    if (((@R0@ - (((int)blockIdx.y) * 128)) / 6) < ((k * 64) + 63)) {
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        float condval;
        if (((((((k * 64) + ((i_8 >> 2) * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4)) + (i_8 & 3)) <= ((((((((int)threadIdx.x) >> 6) * 16) + (((int)threadIdx.x) & 15)) + @R0@) - (((int)blockIdx.y) * 128)) / 6)) & (bool)1)) {
          condval = acc_s[i_8];
        } else {
          condval = -HUGE_VALF;
        }
        acc_s[i_8] = condval;
      }
    }
    m_prev[0] = m_i[0];
    m_i_clear[0] = -HUGE_VALF;
    #pragma unroll
    for (int rv = 0; rv < 16; ++rv) {
      m_i_clear[0] = max(m_i_clear[0], acc_s[(((rv & 3) * 4) + (rv >> 2))]);
    }
    m_i_clear[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(m_i_clear[0]);
    m_i[0] = max(m_i[0], m_i_clear[0]);
    m_i[0] = max(m_i[0], m_prev[0]);
    sf[0] = __expf(((m_prev[0] * 8.838835e-02f) - (m_i[0] * 8.838835e-02f)));
    l_r[0] = (l_r[0] * sf[0]);
    if (sf[0] != 1.000000e+00f) {
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        float4 __1;
          float4 v_ = *(float4*)(acc_o + (i_9 * 4));
          float4 v__1 = make_float4(sf[0], sf[0], sf[0], sf[0]);
          __1.x = (v_.x*v__1.x);
          __1.y = (v_.y*v__1.y);
          __1.z = (v_.z*v__1.z);
          __1.w = (v_.w*v__1.w);
        *(float4*)(acc_o + (i_9 * 4)) = __1;
      }
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      acc_s[i_10] = __expf(((acc_s[i_10] * 8.838835e-02f) - (m_i[0] * 8.838835e-02f)));
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      l_r[0] = (l_r[0] + acc_s[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      uint2 __2;
      float4 v__2 = *(float4*)(acc_s + (i_12 * 4));
      *((half_t*)(&(((half2*)(&(__2.x)))->x))) = (half_t)(v__2.x);
      *((half_t*)(&(((half2*)(&(__2.x)))->y))) = (half_t)(v__2.y);
      *((half_t*)(&(((half2*)(&(__2.y)))->x))) = (half_t)(v__2.z);
      *((half_t*)(&(((half2*)(&(__2.y)))->y))) = (half_t)(v__2.w);
      *(uint2*)(p_cast + (i_12 * 4)) = __2;
    }
    {
      half_t B_local_1[32];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        for (int j_2 = 0; j_2 < 8; ++j_2) {
          for (int local_id = 0; local_id < 4; ++local_id) {
            B_local_1[((j_2 * 4) + local_id)] = ((half_t*)V_s)[(((((((((j_2 >> 2) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((j_2 & 3) >> 1)) & 1) * 32)) + ((((local_id >> 1) + (j_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int j_3 = 0; j_3 < 8; ++j_3) {
          {
      *(((float32x4*)acc_o) + j_3) = __builtin_amdgcn_mfma_f32_16x16x16f16(*(((float16x4*)B_local_1) + j_3),
                    *(((float16x4*)p_cast) + ki_1),
                    *(((float32x4*)acc_o) + j_3), 0, 0, 0);
    };
        }
      }
    }
  }
  l_r[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(l_r[0]);
  l_i[0] = l_r[0];
  float condval_1;
  if ((0.000000e+00f < l_i[0])) {
    condval_1 = l_i[0];
  } else {
    condval_1 = 1.000000e+00f;
  }
  rl[0] = (1.000000e+00f / condval_1);
  #pragma unroll
  for (int i_13 = 0; i_13 < 8; ++i_13) {
    half_t O_local_cast[4];
    uint2 __3;
    float4 __4;
      float4 v__3 = *(float4*)(acc_o + (i_13 * 4));
      float4 v__4 = make_float4(rl[0], rl[0], rl[0], rl[0]);
      __4.x = (v__3.x*v__4.x);
      __4.y = (v__3.y*v__4.y);
      __4.z = (v__3.z*v__4.z);
      __4.w = (v__3.w*v__4.w);
    *((half_t*)(&(((half2*)(&(__3.x)))->x))) = (half_t)(__4.x);
    *((half_t*)(&(((half2*)(&(__3.x)))->y))) = (half_t)(__4.y);
    *((half_t*)(&(((half2*)(&(__3.y)))->x))) = (half_t)(__4.z);
    *((half_t*)(&(((half2*)(&(__3.y)))->y))) = (half_t)(__4.w);
    *(tl::RegVec<uint2>*)(O_local_cast + 0) = __3;
    *(tl::RegVec<uint2>*)(O + ((((((((((((((int)threadIdx.x) >> 6) * 16) + (((int)threadIdx.x) & 15)) + @R0@) - (((int)blockIdx.y) * 128)) / 6) * 3072) + (((int)blockIdx.x) * 768)) + ((6 & (((((((((int)threadIdx.x) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 4) - (((int)blockIdx.y) * 128)) % 6) >> 31)) * 128)) + (((((((((int)threadIdx.x) >> 6) * 16) + (((int)threadIdx.x) & 15)) + 4) - (((int)blockIdx.y) * 128)) % 6) * 128)) + (i_13 * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4))) = *(tl::RegVec<uint2>*)(O_local_cast + 0);
  }
}

"""


# Form JTP = J with two changes, both text edits of _J_SRC (PREFILL-MAX round 1):
# (1) V goes to LDS transposed, [d][key pair] dwords at d * 32 + (c ^ 2 (d & 15)); the PV GEMM's A operand is then one 8-byte read per MFMA (J: four 2-byte reads,
#     128 ds_read_u16 per K step; JTP: 16 ds_read2st64_b64). V's global-load thread map changes (key pair c = (l & 15) | ((w & 1) << 4), d chunk j = (l >> 4) | ((w >> 1) << 2), lane l, wave w)
#     so each LDS write instruction hits 32 banks. The empty asm keeps address math in the loop (126 VGPRs, occupancy 4 kept).
# (2) next step's K and V global loads are issued before the PV GEMM (step 0 before the loop); the loop top keeps only the barrier before the LDS store.
_JTP_VLOAD = (r"    #pragma unroll\n    for \(int i_4 = 0; i_4 < 2; \+\+i_4\) \{\n.*?\n    \}\n",
              "    #pragma unroll\n    for (int i_4 = 0; i_4 < 2; ++i_4) {\n      *(tl::RegVec<uint4>*)(V_r + (i_4 * 8)) = *(tl::RegVec<uint4>*)(V + ((k * 32768) + ((((((((int)threadIdx.x) & 15) | ((((int)threadIdx.x) >> 2) & 16))) * 2) + i_4) * 512) + (((int)blockIdx.x) * 128) + (((((((int)threadIdx.x) & 63) >> 4) | ((((int)threadIdx.x) >> 5) & 12))) * 8)));\n    }\n")
_JTP_VSTORE = (r"    #pragma unroll\n    for \(int i_6 = 0; i_6 < 2; \+\+i_6\) \{\n.*?\n    \}\n",
               """    {
      int vt_t = (int)threadIdx.x;
      asm volatile("" : "+v"(vt_t));
      const int vt_c = (vt_t & 15) | ((vt_t >> 2) & 16);
      const int vt_j = ((vt_t & 63) >> 4) | ((vt_t >> 5) & 12);
      const uint4 va = *(uint4*)(V_r + 0);
      const uint4 vb = *(uint4*)(V_r + 8);
      const uint vw[4] = {va.x, va.y, va.z, va.w};
      const uint vh[4] = {vb.x, vb.y, vb.z, vb.w};
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int d = (vt_j * 8) + e;
        const uint lo = (e & 1) ? (vw[e >> 1] >> 16) : (vw[e >> 1] & 0xffffu);
        const uint hi = (e & 1) ? (vh[e >> 1] & 0xffff0000u) : (vh[e >> 1] << 16);
        ((uint*)V_s)[(d * 32) + (vt_c ^ ((d & 15) << 1))] = lo | hi;
      }
    }
""")
_JTP_VREAD = (r"        for \(int j_2 = 0; j_2 < 8; \+\+j_2\) \{\n          for \(int local_id = 0; local_id < 4; \+\+local_id\) \{\n.*?\n          \}\n        \}\n",
              """        int vr_t = (int)threadIdx.x;
        asm volatile("" : "+v"(vr_t));
        const int vr_l = vr_t & 15;
        const int vr_b = (vr_l * 32) + (((ki_1 * 8) + (((vr_t & 63) >> 4) * 2)) ^ (vr_l << 1));
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; ++j_2) {
          *(uint2*)(B_local_1 + (j_2 * 4)) = *(uint2*)(((uint*)V_s) + (vr_b + (j_2 * 512)));
        }
""")
_JTP_LOADS = r"(    #pragma unroll\n    for \(int i_3 = 0; i_3 < 2; \+\+i_3\) \{\n.*?\n    \}\n    #pragma unroll\n    for \(int i_4 = 0; i_4 < 2; \+\+i_4\) \{\n.*?\n    \}\n)(    __syncthreads\(\);\n)"
_JTP_NK = "((@NK@ - (((int)blockIdx.y) * 128)) / 384)"


def _jtp_template():
    s = _J_SRC
    for pat, new in (_JTP_VLOAD, _JTP_VSTORE, _JTP_VREAD):
        s, n = re.subn(pat, lambda m, new=new: new, s, flags=re.S)
        if n != 1:
            raise RuntimeError("JTP: J source anchor not found")
    m = re.search(_JTP_LOADS, s, re.S)
    ld = m.group(1)
    s = s[:m.start()] + m.group(2) + s[m.end():]
    loop = s.index("  for (int k = 0; k < " + _JTP_NK)
    s = s[:loop] + ld.replace("(k * 32768)", "0").replace("\n    ", "\n  ") + s[loop:]
    i = s.index("    {\n      half_t B_local_1[32];")
    return s[:i] + f"    if ((k + 1) < {_JTP_NK}) {{\n" + ld.replace("(k * 32768)", "((k + 1) * 32768)") + "    }\n" + s[i:]


def _j_source(L, form="J"):
    if L % 64 or L <= 0:
        raise ValueError(f"form J needs L % 64 == 0, got {L}")
    return _j_template(form).replace("@R0@", str(6 * L - 128)).replace("@NK@", str(6 * L + 383))


_JTP_SRC = _jtp_template()

# Form JTPR = JTP with the P cast by v_cvt_pkrtz_f16_f32 (round toward zero, two values per instruction) instead of four round-to-nearest casts
# (PREFILL-MAX round 1 form JTPR, evidence/PREFILL-MAX/src/jvar.py; decision 9 Nov accepts its error, about 4.7e-4 against fp64 on randn0, CK's level).
_JTPR_PC = re.compile(r"      uint2 __2;\n      float4 v__2 = \*\(float4\*\)\(acc_s \+ \(i_12 \* 4\)\);\n(?:.*\n){4}      \*\(uint2\*\)\(p_cast \+ \(i_12 \* 4\)\) = __2;\n")
_JTPR_NEW = """      uint2 __2;
      float4 v__2 = *(float4*)(acc_s + (i_12 * 4));
      __2.x = __builtin_bit_cast(uint, __builtin_amdgcn_cvt_pkrtz(v__2.x, v__2.y));
      __2.y = __builtin_bit_cast(uint, __builtin_amdgcn_cvt_pkrtz(v__2.z, v__2.w));
      *(uint2*)(p_cast + (i_12 * 4)) = __2;
"""


def _jtpr_template():
    s, n = _JTPR_PC.subn(lambda m: _JTPR_NEW, _JTP_SRC)
    if n != 1:
        raise RuntimeError("JTPR: P cast anchor not found")
    return s


_JTPR_SRC = _jtpr_template()

# Form M4HL (PREFILL-MAX round 2, evidence/PREFILL-MAX/r2/src/m32.py 1c8554db, form M4HL = TMPL with NT 256, HOIST 1, EXP2 0, LD 1, LZ 0, DG 0; branches resolved by unifdef):
# 32x32x8 MFMA, 4 waves, rows 128 (32 flat rows f = 6 q + g per wave), grid (Hkv, 6 L / 128). S^T = K Q^T in 32x32 tiles (A = K rows from LDS, B = Q from registers);
# P stays in registers as the B operand of O^T = V^T P^T (A = V^T from LDS, transposed [d][key slot], 16-byte chunks XOR-swizzled by d & 7). One row per lane:
# row max by one cross-lane step (lane ^ 32), row sum reduced once after the loop. K/V global loads from a uniform base + 32-bit offset, next step's loads right after the LDS barrier.
# Only @R0@ depends on L. Valid only for L % 64 == 0 (no bounds checks). Same LDS as J (K_s, V_s, 16384 B each).
_M4HL_SRC = r"""#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>

#define NT 256
#define KEEP(x_)
#define ROWS (NT / 2)
#define LL ((@R0@ + 128) / 6)
typedef _Float16 hv4 __attribute__((ext_vector_type(4)));
typedef float fv16 __attribute__((ext_vector_type(16)));
typedef uint uv4 __attribute__((ext_vector_type(4)));
typedef float fv2 __attribute__((ext_vector_type(2)));
typedef uint uv2 __attribute__((ext_vector_type(2)));
// halves 0..3 / 4..7 of a 16-byte vector as an MFMA operand. Not __builtin_bit_cast(hv4, v.zw): clang 22 lowered that to the low half (PREFILL-MAX r2, M4 v4 asm).
#define LO4(v_) __builtin_bit_cast(hv4, (uv2)__builtin_shufflevector((v_), (v_), 0, 1))
#define HI4(v_) __builtin_bit_cast(hv4, (uv2)__builtin_shufflevector((v_), (v_), 2, 3))
#define U4C(u_, i_) ((i_) == 0 ? (u_).x : (i_) == 1 ? (u_).y : (i_) == 2 ? (u_).z : (u_).w)

extern "C" __global__ void __launch_bounds__(NT) __attribute__((amdgpu_waves_per_eu(2, 2))) main_kernel(half_t* __restrict__ K, half_t* __restrict__ O, half_t* __restrict__ Q, half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  uv4* K_s = (uv4*)buf_dyn_shmem;               // 64 keys x 16 chunks
  uv4* V_s = (uv4*)(buf_dyn_shmem + 16384);     // 128 d x 8 chunks
  const int t = (int)threadIdx.x;
  const int lane = t & 63, w = t >> 6, r = lane & 31, h = lane >> 5;
  const int kvh = (int)blockIdx.x;
  const int F0 = 6 * LL - ROWS * ((int)blockIdx.y + 1);
  const int f = F0 + 32 * w + r;
  const int q = f / 6;
  const int g = f - 6 * q;
  const int nk = ((F0 + ROWS - 1) / 6) / 64 + 1;
  const int qmin = F0 / 6;
  const float sc = 8.838835e-02f;

  hv4 Qx[16];
  {
    const uv4* qp = (const uv4*)(Q + (q * 3072 + (kvh * 6 + g) * 128 + 64 * h));
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
      uv4 x = qp[m];
      Qx[2 * m] = LO4(x);
      Qx[2 * m + 1] = HI4(x);
    }
  }
  fv16 acc_o[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) acc_o[i] = (fv16)0.0f;
  float m_i = -HUGE_VALF, l_r = 0.0f;

  // staging map. K: chunk kc = t & 15 of key (t >> 4) + (NT / 16) i. V: key group vg = t & 15 (keys 4 vg ..), NT 256: 4 keys, chunk vc = t >> 4; NT 512: 2 keys (pair vp), chunk vc = t >> 5.
  constexpr int KN = 1024 / NT;           // K chunks per thread
  constexpr int VN = 1024 / NT;           // V rows (keys) per thread
  const int kc = t & 15, kk0 = t >> 4;
  const int vg = t & 15;
  const int vp = (NT == 512) ? ((t >> 4) & 1) : 0;
  const int vc = (NT == 512) ? (t >> 5) : (t >> 4);
  const int vkey0 = 4 * vg + 2 * vp;
  uv4 Kr[KN], Vr[VN];
  const half_t* Kb = K + kvh * 128;
  const half_t* Vb = V + kvh * 128;
  // uniform base (scalar) + 32-bit unsigned byte offset per thread: lets the compiler use the saddr form, no 64-bit address math per load
  const char* Kc = (const char*)(K + kvh * 128);
  const char* Vc = (const char*)(V + kvh * 128);
  const unsigned koff = (unsigned)((kk0 * 512 + 8 * kc) * 2), voff = (unsigned)((vkey0 * 512 + 8 * vc) * 2);
  auto loadkv = [&](int ks) {
    const char* kb_ = Kc + (size_t)ks * 65536;
    const char* vb_ = Vc + (size_t)ks * 65536;
    #pragma unroll
    for (int i = 0; i < KN; ++i) Kr[i] = *(const uv4*)(kb_ + (koff + (unsigned)(i * (NT / 16) * 1024)));
    #pragma unroll
    for (int i = 0; i < VN; ++i) Vr[i] = *(const uv4*)(vb_ + (voff + (unsigned)(i * 1024)));
  };
  loadkv(0);
  // V slot of the thread's first key: nb = vg >> 3, j = (vg >> 1) & 3, hh = vg & 1
  const int vchunk = 4 * (vg >> 3) + 2 * (((vg >> 1) & 3) >> 1) + (vg & 1);
  const int vin = 4 * ((vg >> 1) & 1) + 2 * vp;    // half offset inside the chunk

  for (int k = 0; k < nk; ++k) {
    if (k > 0) __syncthreads();
    {
      int tt = t;
      KEEP(tt);
      #pragma unroll
      for (int i = 0; i < KN; ++i) {
        const int key = (tt >> 4) + (NT / 16) * i;
        K_s[key * 16 + ((tt & 15) ^ (key & 7))] = Kr[i];
      }
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int d = 8 * vc + e;
        // v_perm_b32: low (even e) or high (odd e) halves of two dwords in one instruction
        const uint psel = (e & 1) ? 0x07060302u : 0x05040100u;
        const uint w01 = __builtin_amdgcn_perm(U4C(Vr[1], e >> 1), U4C(Vr[0], e >> 1), psel);
        uint* dst = (uint*)(V_s + d * 8 + (vchunk ^ (d & 7))) + (vin >> 1);
        const uint w23 = __builtin_amdgcn_perm(U4C(Vr[3], e >> 1), U4C(Vr[2], e >> 1), psel);
        *(uv2*)dst = (uv2){w01, w23};
      }
    }
    __syncthreads();
    if (k + 1 < nk) loadkv(k + 1);

    fv16 acc_s[2];
    acc_s[0] = (fv16)0.0f;
    acc_s[1] = (fv16)0.0f;
    {
      int rr = r;
      KEEP(rr);
      const uv4* kp = K_s + rr * 16;
      #pragma unroll
      for (int m = 0; m < 8; ++m) {
        const int ch = (8 * h + m) ^ (rr & 7);
        uv4 a0 = kp[ch];
        uv4 a1 = kp[512 + ch];
        acc_s[0] = __builtin_amdgcn_mfma_f32_32x32x8f16(LO4(a0), Qx[2 * m], acc_s[0], 0, 0, 0);
        acc_s[1] = __builtin_amdgcn_mfma_f32_32x32x8f16(LO4(a1), Qx[2 * m], acc_s[1], 0, 0, 0);
        acc_s[0] = __builtin_amdgcn_mfma_f32_32x32x8f16(HI4(a0), Qx[2 * m + 1], acc_s[0], 0, 0, 0);
        acc_s[1] = __builtin_amdgcn_mfma_f32_32x32x8f16(HI4(a1), Qx[2 * m + 1], acc_s[1], 0, 0, 0);
      }
    }
    if (qmin < k * 64 + 63) {
      #pragma unroll
      for (int nb = 0; nb < 2; ++nb) {
        #pragma unroll
        for (int v = 0; v < 16; ++v) {
          const int key = k * 64 + 32 * nb + 8 * (v >> 2) + 4 * h + (v & 3);
          if (key > q) acc_s[nb][v] = -HUGE_VALF;
        }
      }
    }
    float mx = -HUGE_VALF;
    #pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      #pragma unroll
      for (int v = 0; v < 16; ++v) mx = max(mx, acc_s[nb][v]);
    }
    mx = max(mx, __shfl_xor(mx, 32));
    const float m_prev = m_i;
    m_i = max(m_i, mx);
    const float sf = __expf((m_prev * sc) - (m_i * sc));
#define PEXP(x_) __expf(((x_) * sc) - (m_i * sc))
    l_r = l_r * sf;
    if (sf != 1.0f) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc_o[i] = acc_o[i] * sf;
    }
    hv4 P[8];
    #pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float p0 = PEXP(acc_s[nb][4 * j]);
        float p1 = PEXP(acc_s[nb][4 * j + 1]);
        float p2 = PEXP(acc_s[nb][4 * j + 2]);
        float p3 = PEXP(acc_s[nb][4 * j + 3]);
        l_r = l_r + p0 + p1 + p2 + p3;
        hv4 pv;
        pv[0] = (_Float16)p0; pv[1] = (_Float16)p1; pv[2] = (_Float16)p2; pv[3] = (_Float16)p3;
        P[4 * nb + j] = pv;
      }
    }
    {
      int rr = r;
      KEEP(rr);
      const uv4* vb = V_s + rr * 8;
      #pragma unroll
      for (int nb = 0; nb < 2; ++nb) {
        #pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int ch = (4 * nb + 2 * jj + h) ^ (rr & 7);
          uv4 a[4];
          #pragma unroll
          for (int dt = 0; dt < 4; ++dt) a[dt] = vb[dt * 256 + ch];
          #pragma unroll
          for (int dt = 0; dt < 4; ++dt)
            acc_o[dt] = __builtin_amdgcn_mfma_f32_32x32x8f16(LO4(a[dt]), P[4 * nb + 2 * jj], acc_o[dt], 0, 0, 0);
          #pragma unroll
          for (int dt = 0; dt < 4; ++dt)
            acc_o[dt] = __builtin_amdgcn_mfma_f32_32x32x8f16(HI4(a[dt]), P[4 * nb + 2 * jj + 1], acc_o[dt], 0, 0, 0);
        }
      }
    }
  }
  l_r = l_r + __shfl_xor(l_r, 32);
  const float rl = 1.0f / ((0.0f < l_r) ? l_r : 1.0f);
  half_t* op = O + (q * 3072 + (kvh * 6 + g) * 128 + 4 * h);
  #pragma unroll
  for (int dt = 0; dt < 4; ++dt) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      hv4 o;
      o[0] = (_Float16)(acc_o[dt][4 * j] * rl); o[1] = (_Float16)(acc_o[dt][4 * j + 1] * rl);
      o[2] = (_Float16)(acc_o[dt][4 * j + 2] * rl); o[3] = (_Float16)(acc_o[dt][4 * j + 3] * rl);
      *(uv2*)(op + 32 * dt + 8 * j) = __builtin_bit_cast(uv2, o);
    }
  }
}
"""
_M4HL_PC = "        hv4 pv;\n        pv[0] = (_Float16)p0; pv[1] = (_Float16)p1; pv[2] = (_Float16)p2; pv[3] = (_Float16)p3;\n        P[4 * nb + j] = pv;\n"
# Form M4HLR = M4HL with the same pkrtz P cast as JTPR (PREFILL-MAX round 4).
_M4HLR_NEW = "        P[4 * nb + j] = __builtin_bit_cast(hv4, (uv2){__builtin_bit_cast(uint, __builtin_amdgcn_cvt_pkrtz(p0, p1)), __builtin_bit_cast(uint, __builtin_amdgcn_cvt_pkrtz(p2, p3))});\n"
if _M4HL_SRC.count(_M4HL_PC) != 1:
    raise RuntimeError("M4HLR: P cast anchor not found")
_M4HLR_SRC = _M4HL_SRC.replace(_M4HL_PC, _M4HLR_NEW)
_J_FORMS = {"J": 512, "JTP": 512, "JTPR": 512, "M4HL": 256, "M4HLR": 256}     # form -> threads (rows 128, bN 64 for all)


def _j_template(form):
    return {"J": _J_SRC, "JTP": _JTP_SRC, "JTPR": _JTPR_SRC, "M4HL": _M4HL_SRC, "M4HLR": _M4HLR_SRC}[form]


def _j_stub(L, B=1, Hkv=4, G=6, D=128, bN=64, threads=512, rows=128):
    """Launch shell for form J. Its device code is never compiled (the hook swaps in _j_source); it fixes the wrapper's signature, the grid (B Hkv, 6 L / 128) and the launch LDS:
    only K_s and V_s, 2 x 16384 = 32768 B, so two blocks fit per CU. The wheel's own lowering of J asks 36864 B (two reduce workspaces) and reads 0.898 to 0.920 of native (D128)."""
    NB = -(-L * G // rows)
    dt = T.float16

    @T.prim_func
    def main(Q: T.Tensor((B, L, Hkv * G, D), dt), K: T.Tensor((B, L, Hkv, D), dt), V: T.Tensor((B, L, Hkv, D), dt), O: T.Tensor((B, L, Hkv * G, D), dt)):
        with T.Kernel(B * Hkv, NB, threads=threads) as (b0, b1):
            K_s = T.alloc_shared((bN, D), dt)
            V_s = T.alloc_shared((bN, D), dt)                       # both read so the wrapper counts them
            T.copy(K[0, 0:bN, 0, :], K_s)
            T.copy(V[0, 0:bN, 0, :], V_s)
            T.sync_threads()
            for i, d in T.Parallel(bN, D):
                O[0, i, 0, d] = K_s[i, d] + V_s[i, d] + Q[0, i, 0, d]
    return main


_J = {}             # set only during a gqa_flash_j compile (try/finally), like _FAST
_J_K = {}


def gqa_flash_j(L, form="J"):
    """Form J or JTP at length L (B 1, Hkv 4, G 6, D 128, causal), clang 22 only: the only stack they were raced on. Returns a callable (Q, K, V) -> O like gqa_flash_fast.
    Cache key: the stub plus a flag holding the source template's md5, so a changed template cannot reuse old bytes."""
    if (L, form) in _J_K:
        return _J_K[(L, form)]
    if L % 64 or L <= 0:
        raise ValueError(f"form J needs L % 64 == 0, got {L}")
    if form not in _J_FORMS:
        raise ValueError(f"gqa_flash_j: unknown form {form!r}")
    if clang_major() != 22:
        raise ValueError(f"form J was raced on clang 22 only, this is clang {clang_major()}")
    import hashlib
    _install_hook()
    _J.clear()
    _J.update(L=L, form=form)
    try:
        t = _j_template(form)
        k = tilelang.compile(_j_stub(L, threads=_J_FORMS[form]), out_idx=[3], compile_flags=["-DGQA_J_" + hashlib.md5(t.encode()).hexdigest()[:12]])
    finally:
        _J.clear()
    _J_K[(L, form)] = k
    return k


@tilelang.jit(out_idx=[3])
def gqa_flash_fwd(B, L, Hkv, G, D, causal, P, bN, threads, q_regs, k_pack=2, edge=False, rev=False, recip=False, swap=False, rows=0):
    return make(B, L, Hkv, G, D, causal, P, bN, threads, q_regs, k_pack, edge, rev, recip, swap, rows)


def ref64(Q, K, V, causal, G):
    q, k, v = Q.double(), K.double().repeat_interleave(G, 2), V.double().repeat_interleave(G, 2)
    s = torch.einsum("bqhd,bkhd->bhqk", q, k) / q.size(-1) ** 0.5
    if causal:
        n = s.size(-1)
        s = s.masked_fill(torch.ones(n, n, device=s.device).tril() == 0, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", F.softmax(s, -1), v)


if __name__ == "__main__":
    from tilelang.profiler import do_bench
    torch.manual_seed(0)
    B, L, Hkv, G, D, causal = 1, 4096, 4, 6, 256, True
    if len(sys.argv) > 5:            # gqa_fa.py L cfg D B G; G = 1 is MHA: 8 heads, not causal, the N5 MHA cases
        G = int(sys.argv[5])
        if G == 1:
            Hkv, causal = 8, False
    if len(sys.argv) > 1:
        L = int(sys.argv[1])
    if len(sys.argv) > 3:            # gqa_fa.py L '[cfg]' D B, or '-' as cfg for the default list
        D = int(sys.argv[3])
    if len(sys.argv) > 4:
        B = int(sys.argv[4])
    Q = torch.randn(B, L, Hkv * G, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
    R = ref64(Q, K, V, causal, G) if B * L <= 8192 else None   # float64 reference needs ~50 GB at 16k
    flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
    qt, kt, vt = Q.transpose(1, 2), K.transpose(1, 2).repeat_interleave(G, 1), V.transpose(1, 2).repeat_interleave(G, 1)
    sms = do_bench(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal), warmup=20, rep=100)
    print(json.dumps({"impl": "torch_sdpa", "B": B, "D": D, "L": L, "ms": round(sms, 3), "tflops": round(flops / sms * 1e-9, 1)}), flush=True)
    # (P, bN, threads, q_regs, k_pack, [edge, rev, recip, swap, fast, rows]): LDS = [bM*D*2 if Q in shared] + 2*bN*D*2 + reduce workspaces. The check below skips LDS >= 64 KB for these kernels; 65536 B alone does dispatch on hand kernels (K1)
    # Round 1 (5 Nov): best (16, 32, 384, Q in registers) = 0.965x SDPA. N5 (6 Nov, evidence/N5/RESULTS.md): the tiles below, plain and fast.
    configs = {256: [(32, 48, 768, True, 2, True, True, False, False, False, 0), (32, 32, 768, True, 2, True, True, True, True, True, 0)],
               128: [(32, 96, 768, True, 2, True, True, True, False, False, 0), (0, 32, 512, True, 2, True, True, True, True, True, 128),
                     (32, 96, 768, True, 2, True, True, True, True, True, 0)]}.get(D, [(32, 16, 768, True, 2)])
    auto = len(sys.argv) > 2 and sys.argv[2] == "auto"    # 'auto': the tile the rules below pick; the default list at G = 1, where the list above does not apply
    if auto or G == 1:
        if D not in (128, 256) or (G == 1 and D != 128) or (G != 1 and (G, Hkv, causal) != (6, 4, True)):
            raise SystemExit("auto: the tile rules cover D 128 at G 6 and G 1, and D 256 at G 6 only")
        configs = [mha_tile(L, B, Hkv) if G == 1 else (d128_tile(L, B) if D == 128 else d256_tile(L, B))] if auto or len(sys.argv) <= 2 or sys.argv[2] == "-" else [tuple(json.loads(sys.argv[2]))]
    elif len(sys.argv) > 2 and sys.argv[2] != "-":
        configs = [tuple(json.loads(sys.argv[2]))]
    for cfg in configs:
        if isinstance(cfg, dict):                            # tile rule: fast kernel, with the lb and sched the rule picks
            t = cfg
            r = {"impl": "tilelang_gqa", "B": B, "D": D, "L": L, "G": G, "rule": True, "P": t["P"], "rows": t.get("rows", 0), "bN": t["bN"], "threads": t["threads"], "lb": t.get("lb", 0), "sched": t.get("sched", ""), "fast": True}
            if t.get("form"):
                r["form"] = t["form"]
            if t.get("vt"):
                r["vt"] = True
            try:
                k = gqa_flash_fast(B, L, Hkv, G, D, causal, t["P"], t["bN"], t["threads"], 2, True, t["rev"], t["recip"], t["swap"], t.get("rows", 0), t.get("lb", 0), t.get("sched", ""), t.get("form", ""), t.get("vt", False))
                O = k(Q, K, V)
                if R is None:
                    so = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2)
                    r["rel_vs_sdpa"] = round(((O.float() - so.float()).norm() / so.float().norm()).item(), 6)
                    good = r["rel_vs_sdpa"] < 2e-3
                else:
                    r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 6)
                    good = r["rel_vs_fp64"] < 2e-3
                if good:
                    ms = do_bench(lambda: k(Q, K, V), warmup=20, rep=100)
                    r.update(ms=round(ms, 3), tflops=round(flops / ms * 1e-9, 1), vs_sdpa=round(sms / ms, 3))
            except Exception as e:  # noqa: BLE001
                r["err"] = (type(e).__name__ + ": " + str(e))[-200:]
            print(json.dumps(r), flush=True)
            continue
        P, bN, th, qr, kp, edge, rev, recip, swap, fast, rows = tuple(cfg) + (False, False, False, False, False, 0)[len(cfg) - 5:]
        lds = (0 if qr else (rows or P * G) * D * 2) + 2 * bN * D * 2
        r = {"impl": "tilelang_gqa", "B": B, "D": D, "L": L, "P": P, "bM": rows or P * G, "bN": bN, "threads": th, "q_regs": qr, "k_pack": kp, "lds_kb": lds // 1024,
             "edge": edge, "rev": rev, "recip": recip, "swap": swap, "fast": fast}
        if lds >= 64 * 1024:              # bN 64 at d256: 65536 B plus about 6 KB of workspaces, hipModuleLaunchKernel rejects it (invalid argument, job 23503844; cause inferred)
            r["err"] = "skipped: LDS >= 64 KB"
            print(json.dumps(r), flush=True)
            continue
        try:
            if fast:
                k = gqa_flash_fast(B, L, Hkv, G, D, causal, P, bN, th, kp, edge, rev, recip, swap, rows)
            else:
                k = gqa_flash_fwd(B, L, Hkv, G, D, causal, P, bN, th, qr, kp, edge, rev, recip, swap, rows)
            O = k(Q, K, V)
            if R is None:                                          # long sequences: check against torch SDPA (fp32 acc)
                so = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2)
                r["rel_vs_sdpa"] = round(((O.float() - so.float()).norm() / so.float().norm()).item(), 6)
                good = r["rel_vs_sdpa"] < 2e-3
            else:
                r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 6)
                good = r["rel_vs_fp64"] < 2e-3
            if good:
                ms = do_bench(lambda: k(Q, K, V), warmup=20, rep=100)
                r.update(ms=round(ms, 3), tflops=round(flops / ms * 1e-9, 1), vs_sdpa=round(sms / ms, 3))
        except Exception as e:  # noqa: BLE001
            r["err"] = (type(e).__name__ + ": " + str(e))[-200:]
        print(json.dumps(r), flush=True)
