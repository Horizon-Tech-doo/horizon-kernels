"""GQA-packed flash-attention forward for gfx90a (MI250X), TileLang.

The upstream AMD example gives every query head its own workgroup, so with GQA each K/V tile is loaded once per query
head (6x for Qwen's 24 q / 4 kv heads). Here one workgroup takes P query positions x all G query heads of one kv head:
the GEMM M dimension is P*G rows (row r = position r // G, head r % G), and each K/V tile is loaded once for all of them.
Q can stay in shared memory or go to registers (frees LDS for a larger key tile; register-A GEMM verified exact on
gfx90a by fa_diag.py).

Layout: Q [B, L, H, D], K/V [B, L, Hkv, D], H = Hkv * G, fp16 in/out, fp32 accumulate. Causal or not.
Usage: python gqa_fa.py [L [cfg D [B [G]]]]   (correctness vs float64 + speed vs torch SDPA and vs the upstream example's best tile). cfg: JSON tile, '-' for the default list, 'auto' for d128_tile or mha_tile. G 1 is MHA.

gqa_flash_fast is the same kernel with C edits on the generated HIP source (see evidence/N5/RESULTS.md; none of this is expressible in TileLang 0.1.15):
- one load block per K step, 2 rows of K and 2 of V per lane, 16 bytes each; next step's rows are loaded to registers before the V GEMM, so they overlap it
- two barriers per K step: store K, V | barrier | read K, V | barrier. race_free() checks that every write->read and read->write of an LDS array crosses a barrier
- row sum reduced once after the K loop (exact up to summation order); O rescale skipped when sf == 1 (exact)
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


_FAST = {}          # context of the fast kernel being compiled. Set only for the duration of a gqa_flash_fast call (try/finally), so a disk-cache hit cannot leave it set for a later plain kernel.


def _install_hook():
    from tilelang.contrib import hipcc
    if getattr(hipcc.compile_hip, "_gqa_fast", False):
        return
    orig = hipcc.compile_hip

    def comp(code, *a, **k):
        if _FAST:
            code = _fast_edit(code, _FAST)
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
_D128_TIMED = (1024, 1536, 2048, 2560, 2816, 3072, 3584, 4096, 4160, 5000, 5120, 6144, 8192, 10240, 12288, 16384)


def d128_tile(L, B=1):
    """Tile for D 128, G 6, Hkv 4, causal: keyword arguments for gqa_flash_fast. rows 128 bN32, rows 256 bN32 or P32 bN96, whichever won the race at the timed length nearest L, per compiler (clang 22 and clang 18 differ). The table is a race result, not a model: rows 128 has a hole at 4160 on clang 22.
    Timed at the lengths in _D128_TIMED, B 1, randn0 (evidence/N5/RESULTS.md sections 15, 16; rows 256 raced only at 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384 on clang 22 and not picked on clang 18). Other L and B > 1 are untimed. On clang 22 this hand-edited path beats CK only up to L 2560; 2816 is a tie and 3072 up is a loss. The TileLang source form rows 128 bN64 J on fork-main reads above CK to L 3584 (section 16); this function and gqa_flash_fast do not reach it."""
    c = clang_major()
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


def gqa_flash_fast(B, L, Hkv, G, D, causal, P, bN, threads, k_pack=2, edge=True, rev=True, recip=False, swap=False, rows=0, lb=0, sched=""):
    """Plain wrapper, not a jit: it binds the edit to this one compile. Not thread-safe with a concurrent compile in the same process.
    lb: second argument of __launch_bounds__, the occupancy the compiler targets (0 = none). sched: LLVM -misched value. Both checked by _check_tuning."""
    if L % 2 or bN % 2 or not edge:
        raise ValueError("gqa_flash_fast needs even L and bN and the edge-only mask")
    _check_tuning(G, P, threads, rows, lb, sched)
    _install_hook()
    _FAST.clear()
    _FAST.update(L=L, Hkv=Hkv, B=B, D=D, bN=bN, threads=threads, swap=swap, lb=lb, sched=sched)
    try:
        if lb or sched:
            f = make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows)
            flags = ["-DGQA_FAST_N5_V2", f"-DGQA_LB_{lb}"] + ([f"-DGQA_SCHED_{sched.replace('-', '_')}"] if sched else [])
            return tilelang.compile(f, out_idx=[3], compile_flags=flags)
        return _gqa_flash_fast_jit(B, L, Hkv, G, D, causal, P, bN, threads, k_pack, edge, rev, recip, swap, rows)
    finally:
        _FAST.clear()


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
        if D != 128 or (G != 1 and (G, Hkv, causal) != (6, 4, True)):
            raise SystemExit("auto: the tile rules cover D 128 at G 6 and G 1 only")
        configs = [mha_tile(L, B, Hkv) if G == 1 else d128_tile(L, B)] if auto or len(sys.argv) <= 2 or sys.argv[2] == "-" else [tuple(json.loads(sys.argv[2]))]
    elif len(sys.argv) > 2 and sys.argv[2] != "-":
        configs = [tuple(json.loads(sys.argv[2]))]
    for cfg in configs:
        if isinstance(cfg, dict):                            # tile rule: fast kernel, with the lb and sched the rule picks
            t = cfg
            r = {"impl": "tilelang_gqa", "B": B, "D": D, "L": L, "G": G, "rule": True, "P": t["P"], "rows": t.get("rows", 0), "bN": t["bN"], "threads": t["threads"], "lb": t.get("lb", 0), "sched": t.get("sched", ""), "fast": True}
            try:
                k = gqa_flash_fast(B, L, Hkv, G, D, causal, t["P"], t["bN"], t["threads"], 2, True, t["rev"], t["recip"], t["swap"], t.get("rows", 0), t.get("lb", 0), t.get("sched", ""))
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
