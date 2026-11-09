"""GQA-packed flash-attention forward for gfx90a (MI250X), TileLang.

The upstream AMD example gives every query head its own workgroup, so with GQA each K/V tile is loaded once per query
head (6x for Qwen's 24 q / 4 kv heads). Here one workgroup takes P query positions x all G query heads of one kv head:
the GEMM M dimension is P*G rows (row r = position r // G, head r % G), and each K/V tile is loaded once for all of them.
Q can stay in shared memory or go to registers (frees LDS for a larger key tile; register-A GEMM verified exact on
gfx90a by fa_diag.py).

Layout: Q [B, L, H, D], K/V [B, L, Hkv, D], H = Hkv * G, fp16 in/out, fp32 accumulate. Causal or not.
Usage: python gqa_fa.py   (correctness vs float64 + speed vs torch SDPA and vs the upstream example's best tile)

gqa_flash_fast is the same kernel with C edits on the generated HIP source (see evidence/N5/RESULTS.md; none of this is expressible in TileLang 0.1.15):
- one load block per K step, 2 rows of K and 2 of V per lane, 16 bytes each; next step's rows are loaded to registers before the V GEMM, so they overlap it
- two barriers per K step: store K, V | barrier | read K, V | barrier. race_free() checks that every write->read and read->write of an LDS array crosses a barrier
- row sum reduced once after the K loop (exact up to summation order); O rescale skipped when sf == 1 (exact)
"""
import json
import re
import sys

import torch
import torch.nn.functional as F
import tilelang
import tilelang.rocm.language as T
from tilelang.tileop.base import GemmWarpPolicy


def make(B, L, Hkv, G, D, causal, P, bN, threads, q_regs, k_pack=2, edge=False, rev=False, recip=False, swap=False, rows=0, lds_pad=0):
    """rows != 0: rows per block need not be a multiple of G. The flat row r = position * G + head is split into blocks of `rows`, so P is ignored and the number of waves
    (rows / 16) can be chosen freely; a block may start in the middle of a position's heads.
    lds_pad != 0: reserve that many more bytes of LDS at launch, for the padded layout of gqa_flash_fast. The launch size comes from alloc_shared, so a buffer has to stay alive to enlarge it."""
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
        with T.Kernel(d0, d1, threads=threads, **({"prelude": "#include <hip/hip_runtime.h>\nstatic inline __device__ void gqa_keep(void* p) { (void)p; }\n"} if lds_pad else {})) as (b0, b1):
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
            if lds_pad:
                pad_s = T.alloc_shared((lds_pad,), T.uint8)
                T.call_extern("handle", "gqa_keep", T.address_of(pad_s[0]))
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
            if lds_pad:
                T.call_extern("handle", "gqa_keep", T.address_of(pad_s[0]))

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


def race_free(s, ring=False):
    """Every write->read and read->write of one LDS array, cyclically over the K step, must cross a barrier. Threads write disjoint elements.
    ring: the arrays alternate by k & 1, so the cycle is two K steps long and each step has its own copy of each array."""
    ev = _lds_events(s)
    if ring:
        ev = [(t, a + "0") for t, a in ev] + [(t, a + "1") for t, a in ev]
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
    if c.get("ring"):                # K and V tiles alternate by k & 1: tile k + 1 is written while tile k may still be read, so the barrier at the end of the K step goes
        T_h = 2 * bN * D
        body = body.replace("((half_t*)K_s)", "(((half_t*)K_s) + ring_o)").replace("((half_t*)V_s)", "(((half_t*)V_s) + ring_o)")
        j = body.index("{") + 1
        s = s[:k_loop] + body[:j] + f"\n    int ring_o = (k & 1) * {T_h};" + body[j:] + s[e:]
    else:
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
    bad = race_free(s, c.get("ring", False))
    if bad:
        raise RuntimeError(f"gqa_flash_fast: LDS accesses not separated by a barrier: {bad}")
    return s


def _lds_offsets(s, v_off):
    """K_s at LDS 0, V_s at v_off bytes. The pad buffer and the reduce workspaces overlap them on purpose: AllReduce<64, 16> never touches its workspace (shfl_xor only, tl_templates/hip/reduce.h)."""
    for a, off in (("K_s", 0), ("V_s", v_off)):
        s, n = re.subn(rf"(void\* {a} = \(\(void\*\)\(\(char\*\)buf_dyn_shmem \+ )\d+(\)\);)", lambda m: f"{m.group(1)}{off}{m.group(2)}", s)
        if n != 1:
            raise RuntimeError(f"gqa_flash_fast: {a} offset not found")
    if "AllReduce<tl::MaxOp, 64, 16, 0>" not in s or re.search(r"AllReduce<tl::\w+, (128|256|512|1024),", s):
        raise RuntimeError("gqa_flash_fast: a reduce may use its LDS workspace")
    return s


def _stride_edit(s, c):
    """LDS row stride of K_s and V_s from 128 B to c["stride"] B (K8). A row of the swizzled layout is C halves; every multiplier that is a whole number of rows scales, bank terms below C stay. Returns (source, dynamic LDS bytes)."""
    D, bN, S = c["D"], c["bN"], c["stride"] // 2
    C = min(64, D & -D)
    dyn = 2 * (D // C) * bN * S * 2

    def scale(m):
        n = int(m.group(1))
        return f"* {n // C * S})" if n >= C and n % C == 0 else m.group(0)

    out = []
    for ln in s.split("\n"):
        if "half_t*)K_s" in ln or "half_t*)V_s" in ln:
            a = ln.index("half_t*)K_s") if "half_t*)K_s" in ln else ln.index("half_t*)V_s")
            ln = ln[:a] + re.sub(r"\* (\d+)\)", scale, ln[a:])
        out.append(ln)
    s = "\n".join(out)
    return _lds_offsets(s, dyn // 2), dyn


_FAST = {}          # context of the fast kernel being compiled. Set only for the duration of a gqa_flash_fast call (try/finally), so a disk-cache hit cannot leave it set for a later plain kernel.


def _install_hook():
    from tilelang.contrib import hipcc
    if getattr(hipcc.compile_hip, "_gqa_fast", False):
        return
    orig = hipcc.compile_hip

    def comp(code, *a, **k):
        if _FAST:
            code = _fast_edit(code, _FAST)
            if _FAST.get("stride"):
                code = _stride_edit(code, _FAST)[0]
            if _FAST.get("ring"):
                code = _lds_offsets(code, 2 * _FAST["bN"] * _FAST["D"])
            if _FAST.get("lb"):
                code, n = re.subn(r"__launch_bounds__\((\d+)\)", lambda m: f"__launch_bounds__({m.group(1)}, {_FAST['lb']})", code)
                if n != 1:
                    raise RuntimeError("gqa_flash_fast: launch bounds not found")
        return orig(code, *a, **k)
    comp._gqa_fast = True
    hipcc.compile_hip = comp


@tilelang.jit(out_idx=[3], compile_flags=["-DGQA_FAST_N5_V2"])     # flag: ignored on the HIP path, keeps fast and plain kernels apart in the cache. V2: entries from V1 may hold a stale-context edit.
def _gqa_flash_fast_jit(B, L, Hkv, G, D, causal, P, bN, threads, k_pack, edge, rev, recip, swap, rows):
    return make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows)


def gqa_flash_fast(B, L, Hkv, G, D, causal, P, bN, threads, k_pack=2, edge=True, rev=True, recip=False, swap=False, rows=0, stride=0, lb=0, ring=False):
    """Plain wrapper, not a jit: it binds the edit to this one compile. Not thread-safe with a concurrent compile in the same process.
    stride: LDS row stride of K and V in bytes (0 = 128 B, the generated layout). lb: second argument of __launch_bounds__, the occupancy the compiler targets (0 = none).
    ring: two K tiles and two V tiles in LDS, alternating by K step, one barrier per K step instead of two."""
    if L % 2 or bN % 2 or not edge:
        raise ValueError("gqa_flash_fast needs even L and bN and the edge-only mask")
    _install_hook()
    _FAST.clear()
    _FAST.update(L=L, Hkv=Hkv, B=B, D=D, bN=bN, threads=threads, swap=swap, lb=lb, ring=ring)
    try:
        if ring:
            if stride:
                raise ValueError("ring with stride is not implemented")
            dyn = 8 * bN * D
            if dyn > 65536:
                raise ValueError(f"ring: two tiles of K and V need {dyn} B of LDS")
            f = make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows, max(dyn - 4 * bN * D - 2 * threads * 4, 16))
            return tilelang.compile(f, out_idx=[3], compile_flags=["-DGQA_FAST_N5_V2", "-DGQA_RING", f"-DGQA_LB_{lb}"])
        if stride:
            _FAST["stride"] = stride
            C = min(64, D & -D)
            dyn = 2 * (D // C) * bN * (stride // 2) * 2
            if stride % 16 or stride < 2 * C or dyn > 65536:
                raise ValueError(f"stride {stride}: multiple of 16, at least {2 * C} B, and K + V ({dyn} B) must fit 64 KB of LDS")
            pad = dyn - 2 * bN * D * 2 - 2 * threads * 4      # alloc_shared already gives K + V + 2 reduce workspaces of 4 B per thread
            f = make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows, max(pad, 16))
            return tilelang.compile(f, out_idx=[3], compile_flags=["-DGQA_FAST_N5_V2", f"-DGQA_STRIDE_{stride}", f"-DGQA_LB_{lb}"])
        if lb:
            f = make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows)
            return tilelang.compile(f, out_idx=[3], compile_flags=["-DGQA_FAST_N5_V2", f"-DGQA_LB_{lb}"])
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
    if len(sys.argv) > 2 and sys.argv[2] != "-":
        configs = [tuple(json.loads(sys.argv[2]))]
    for cfg in configs:
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
