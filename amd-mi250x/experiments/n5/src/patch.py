"""C-level patches on TileLang's generated HIP source, not expressible in TileLang 0.1.15. Flavour letters in E3_PATCH:
A: drop the 2 ThreadSync barriers that exist only because reduce gets an LDS workspace; AllReduce<..,64,..> never touches
   LDS (offset < wave size uses ds_bpermute only, reduce.h), V_s is not written after barrier 2 of the K step.
B: keep l_i per thread during the K loop, reduce it once after the loop instead of every K step.
K: K_s loaded with 16-byte lane loads (T.copy emits 4-byte lane loads here).
X: V likewise: row-major V_s with 16-byte lane loads, or with a transposed tile V_t 16-byte lane loads plus 8 16-bit LDS
   writes per lane, so the V read of the second GEMM becomes ds_read_b64.
D: skip the acc_o rescale when no lane of the wave has sf != 1 (the compare is lane-wise; execz skips the block).
"""
import os
import re


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


CTX = {}
LAST = {}


def _swz(r, c, n1, n2):
    """C expression for TileLang's swizzled LDS index of element (r, c) of a tile (n1, n2); read off generated reads:
    pow2 chunk C = min(64, largest power of 2 dividing n2); column bits log2(C)-1 down to 3 are XORed with row bits 2,1,0."""
    C = min(64, n2 & -n2)
    e = f"((({c}) / {C}) * {C * n1}) + ((({r}) >> 4) * {16 * C}) + ((({r}) & 15) * {C})"
    top = C.bit_length() - 2
    rb = 2
    for b in range(top, 2, -1):
        e += f" + (((((({c}) >> {b}) ^ (({r}) >> {rb})) & 1)) * {1 << b})"
        rb -= 1
    e += f" + (({c}) & 7)"
    return "(" + e + ")"


def _find_load(s, needle):
    """(start, end) of the '#pragma unroll' block in the K loop holding the first store through needle, or None."""
    k_loop = s.index("for (int k = 0;")
    i = s.find(needle, k_loop)
    if i < 0:
        return None
    p = s.rfind("#pragma unroll", k_loop, i)
    if p < 0:
        return None
    return p, _block_end(s, p)


def _kuload(s, name):
    src = "K" if name == "K_s" else "V"
    r = _find_load(s, "*(uint1*)(((half_t*)" + name + ") +")
    if r is None:
        print(f"[patch {name}] no load in K loop, skipped", flush=True)
        return s
    c = CTX
    cpr, steps = c["D"] // 8, -(-c["bN"] * (c["D"] // 8) // c["th"])
    code = f"""
    #pragma unroll
    for (int s_u = 0; s_u < {steps}; ++s_u) {{
      int c_u = s_u * {c["th"]} + ((int)threadIdx.x);
      if (c_u < {c["bN"] * cpr}) {{
        int kk_u = c_u / {cpr};
        int d0_u = (c_u & {cpr - 1}) * 8;
        if (((k * {c["bN"]}) + kk_u) < {c["L"]}) {{
          *(uint4*)(((half_t*){name}) + {_swz("kk_u", "d0_u", c["bN"], c["D"])}) = *(uint4*)({src} + ((((((k * {c["bN"]}) + kk_u) * {c["Hkv"]}) + ((int)blockIdx.y)) * {c["D"]}) + d0_u));
        }}
      }}
    }}
"""
    print(f"[patch {name}] 16-byte load, steps {steps}", flush=True)
    return s[:r[0]] + code + s[r[1]:]


def _vload(s):
    r = _find_load(s, "((half_t*)V_t)[")
    if r is None:
        print("[patch V] no V_t store in K loop, skipped", flush=True)
        return s
    c = CTX
    cpr, steps = c["D"] // 8, -(-c["bN"] * (c["D"] // 8) // c["th"])
    code = f"""
    #pragma unroll
    for (int s_v = 0; s_v < {steps}; ++s_v) {{
      int c_v = s_v * {c["th"]} + ((int)threadIdx.x);
      if (c_v < {c["bN"] * cpr}) {{
        int kk_v = c_v / {cpr};
        int d0_v = (c_v & {cpr - 1}) * 8;
        if (((k * {c["bN"]}) + kk_v) < {c["L"]}) {{
          half_t x_v[8];
          *(uint4*)(x_v) = *(uint4*)(V + ((((((k * {c["bN"]}) + kk_v) * {c["Hkv"]}) + ((int)blockIdx.y)) * {c["D"]}) + d0_v));
          #pragma unroll
          for (int i_v = 0; i_v < 8; ++i_v) {{
            ((half_t*)V_t)[{_swz("d0_v + i_v", "kk_v", c["D"], c["bN"])}] = x_v[i_v];
          }}
        }}
      }}
    }}
"""
    print(f"[patch V] staged V load, steps {steps}", flush=True)
    return s[:r[0]] + code + s[r[1]:]


def _vload2(s):
    """N6, CK's V path: 2 rows loaded 16 bytes per lane, interleaved into dwords (perm-like), dword LDS writes (ds_write2 candidates).
    Requires L even: a row pair is either wholly inside L or wholly outside."""
    r = _find_load(s, "((half_t*)V_t)[")
    if r is None:
        print("[patch P] no V_t store in K loop, skipped", flush=True)
        return s
    c = CTX
    assert c["L"] % 2 == 0 and c["bN"] % 2 == 0
    cpr = c["D"] // 8
    pairs = c["bN"] // 2
    steps = -(-pairs * cpr // c["th"])
    hy = "(int)blockIdx.y"
    wr = ""
    for i in range(4):
        wr += f"""
          *(uint*)(((half_t*)V_t) + {_swz(f"d0_v + {2 * i}", "kk_v", c["D"], c["bN"])}) = (a_v.{"xyzw"[i]} & 0xffffu) | (b_v.{"xyzw"[i]} << 16);
          *(uint*)(((half_t*)V_t) + {_swz(f"d0_v + {2 * i + 1}", "kk_v", c["D"], c["bN"])}) = (a_v.{"xyzw"[i]} >> 16) | (b_v.{"xyzw"[i]} & 0xffff0000u);"""
    code = f"""
    #pragma unroll
    for (int s_v = 0; s_v < {steps}; ++s_v) {{
      int c_v = s_v * {c["th"]} + ((int)threadIdx.x);
      if (c_v < {pairs * cpr}) {{
        int kk_v = (c_v / {cpr}) * 2;
        int d0_v = (c_v & {cpr - 1}) * 8;
        if (((k * {c["bN"]}) + kk_v) < {c["L"]}) {{
          uint4 a_v = *(uint4*)(V + ((((((k * {c["bN"]}) + kk_v) * {c["Hkv"]}) + {hy}) * {c["D"]}) + d0_v));
          uint4 b_v = *(uint4*)(V + ((((((k * {c["bN"]}) + kk_v + 1) * {c["Hkv"]}) + {hy}) * {c["D"]}) + d0_v));{wr}
        }}
      }}
    }}
"""
    print(f"[patch P] CK-style V staging, steps {steps}", flush=True)
    return s[:r[0]] + code + s[r[1]:]


def _first_write(s, k_loop, a):
    """Position of the first store to LDS array a in the K loop."""
    i = k_loop
    for ln in s[k_loop:].split("\n"):
        if ("half_t*)" + a) in ln.split(" = ")[0] and " = " in ln:
            return i + ln.index("half_t*)" + a)
        i += len(ln) + 1
    raise ValueError("no store to " + a)


def _rowidx(r):
    """Offset of (b, row r, hk, d0_v) in a (B, L, Hkv, D) tensor. blockIdx.y = b * Hkv + hk."""
    c = CTX
    by = "blockIdx.x" if c.get("swap") else "blockIdx.y"
    if c.get("B", 1) == 1:
        return f"(((({r}) * {c['Hkv']}) + ((int){by})) * {c['D']}) + d0_v"
    return f"(((((((int){by}) / {c['Hkv']}) * {c['L']}) + ({r})) * {c['Hkv']}) + (((int){by}) % {c['Hkv']})) * {c['D']} + d0_v"


def _stmt_or_block(s, k_loop, i):
    """Span of the load holding position i: the enclosing '#pragma unroll' block, or the statement itself when the copy has a single step and no loop."""
    p = s.rfind("#pragma unroll", k_loop, i)
    if p >= 0:
        e = _block_end(s, p)
        if e > i:
            return p, e
    a = s.rfind("\n", 0, i) + 1
    return a, s.index("\n", i) + 1


def _zload(s, four=False):
    """Z: one load block per K step. Each lane loads 2 rows of K and 2 rows of V, 16 bytes each, under one bound check, so the loads can issue together;
    K goes to LDS with ds_write_b128-sized writes, V either the same (V_s) or interleaved to dwords (V_t). Needs Y (the barrier between K and V goes)."""
    c = CTX
    assert c["L"] % 2 == 0 and c["bN"] % 2 == 0
    # K load block
    k_loop = s.index("for (int k = 0;")
    p, e = _stmt_or_block(s, k_loop, _first_write(s, k_loop, "K_s"))
    s = s[:p] + "/*Z*/" + s[e:]
    needle = "((half_t*)V_t)[" if "((half_t*)V_t)[" in s[k_loop:] else None
    if needle:
        r = _find_load(s, needle)
    else:
        r = _stmt_or_block(s, k_loop, _first_write(s, k_loop, "V_s"))
    cpr = c["D"] // 8
    n_rows = 4 if (four and needle) else 2     # rows per lane
    assert c["bN"] % n_rows == 0
    pairs = c["bN"] // n_rows
    steps = -(-pairs * cpr // c["th"])
    wr = ""
    for w in range(n_rows):
        wr += f"\n          *(uint4*)(((half_t*)K_s) + {_swz(f'kk_v + {w}', 'd0_v', c['bN'], c['D'])}) = k{w}_v;"
    if needle and n_rows == 2:
        for i in range(4):
            wr += f"""
          *(uint*)(((half_t*)V_t) + {_swz(f"d0_v + {2 * i}", "kk_v", c["D"], c["bN"])}) = (v0_v.{"xyzw"[i]} & 0xffffu) | (v1_v.{"xyzw"[i]} << 16);
          *(uint*)(((half_t*)V_t) + {_swz(f"d0_v + {2 * i + 1}", "kk_v", c["D"], c["bN"])}) = (v0_v.{"xyzw"[i]} >> 16) | (v1_v.{"xyzw"[i]} & 0xffff0000u);"""
    elif needle:
        # 4 rows per lane: 8 values of 4 halves each, written as 8-byte stores
        for i in range(4):
            x = "xyzw"[i]
            wr += f"""
          *(uint2*)(((half_t*)V_t) + {_swz(f"d0_v + {2 * i}", "kk_v", c["D"], c["bN"])}) = make_uint2((v0_v.{x} & 0xffffu) | (v1_v.{x} << 16), (v2_v.{x} & 0xffffu) | (v3_v.{x} << 16));
          *(uint2*)(((half_t*)V_t) + {_swz(f"d0_v + {2 * i + 1}", "kk_v", c["D"], c["bN"])}) = make_uint2((v0_v.{x} >> 16) | (v1_v.{x} & 0xffff0000u), (v2_v.{x} >> 16) | (v3_v.{x} & 0xffff0000u));"""
    else:
        for w in range(n_rows):
            wr += f"\n          *(uint4*)(((half_t*)V_s) + {_swz(f'kk_v + {w}', 'd0_v', c['bN'], c['D'])}) = v{w}_v;"
    names = [(f"k{w}_v", "K", w) for w in range(n_rows)] + [(f"v{w}_v", "V", w) for w in range(n_rows)]
    CTX["_z"] = dict(names=names, n_rows=n_rows)
    row = lambda A, w: f"*(uint4*)({A} + {_rowidx(f'(k * {c["bN"]}) + kk_v + {w}')})"  # noqa: E731
    loads = "\n".join(f"          uint4 {n} = {row(A, w)};" for n, A, w in names)
    code = f"""
    #pragma unroll
    for (int s_v = 0; s_v < {steps}; ++s_v) {{
      int c_v = s_v * {c["th"]} + ((int)threadIdx.x);
      if (c_v < {pairs * cpr}) {{
        int kk_v = (c_v / {cpr}) * {n_rows};
        int d0_v = (c_v & {cpr - 1}) * 8;
        if (((k * {c["bN"]}) + kk_v) < {c["L"]}) {{
{loads}{wr}
        }}
      }}
    }}
"""
    print(f"[patch Z] joint K+V load, steps {steps}, rows {n_rows}, V {'t' if needle else 's'}", flush=True)
    s = s[:r[0]] + code + s[r[1]:]
    left = [ln for ln in s[k_loop:_block_end(s, k_loop)].split("\n") if "half_t*)" in ln and any(("half_t*)" + a) in ln.split(" = ")[0] for a in ("K_s", "V_s", "V_t")) and " = " in ln and "_v" not in ln]
    if left:
        raise RuntimeError(f"[patch Z] {len(left)} original stores left")
    return s


def _oload(s, what, where="V"):
    """O (K+V) or Q (V only): the next K step's rows are loaded to registers just before the V GEMM of this step, so the loads overlap it.
    Stores to LDS happen at the top of the next step, before the barrier. Registers stay live across the back edge. Run after Z (needs its block)."""
    c = CTX
    k_loop = s.index("for (int k = 0;")
    hdr = s[k_loop:s.index("{", k_loop)]
    n_kv = re.match(r"for \(int k = 0; k < (.*); \+\+k\) ", hdr).group(1)
    p = s.index("#pragma unroll\n    for (int s_v = 0;", k_loop)
    e = _block_end(s, p)
    block = s[p:e]
    steps = int(re.search(r"s_v < (\d+);", block).group(1))
    if steps != 1:
        raise RuntimeError(f"[patch O] needs one load step per thread, have {steps}")
    slots = int(re.search(r"if \(c_v < (\d+)\)", block).group(1))
    cpr = c["D"] // 8
    n_rows = CTX["_z"]["n_rows"]
    stores = "\n".join(ln for ln in block.split("\n") if " = " in ln and any(a in ln.split(" = ")[0] for a in ("K_s", "V_s", "V_t")))
    row = lambda A, w, kx: f"*(uint4*)({A} + {_rowidx(f'(({kx}) * {c["bN"]}) + kk_v + {w}')})"  # noqa: E731
    names = CTX["_z"]["names"]
    staged = names if what == "O" else [n for n in names if n[1] == "V"]
    loads = lambda kx: "\n".join(f"          {n} = {row(A, w, kx)};" for n, A, w in staged)  # noqa: E731
    direct = "" if what == "O" else "".join(f"\n          uint4 {n} = {row(A, w, 'k')};" for n, A, w in names if n[1] == "K")
    decl = f"""
  int c_v = (int)threadIdx.x;
  int kk_v = (c_v / {cpr}) * {n_rows};
  int d0_v = (c_v & {cpr - 1}) * 8;
""" + "".join(f"  uint4 {n} = make_uint4(0, 0, 0, 0);\n" for n, _, _ in staged) + f"""
  if ((0 < ({n_kv})) && (c_v < {slots}) && (kk_v < {c["L"]})) {{
{loads("0")}
  }}
  """
    store_block = f"""
    if ((c_v < {slots}) && (((k * {c["bN"]}) + kk_v) < {c["L"]})) {{{direct}
{stores}
    }}
"""
    s2 = s[:p] + store_block + s[e:]
    if where == "V":      # before the V GEMM
        i = s2.index("half_t B_local_1[", k_loop)
        j = s2.rfind("{", 0, i)
    elif where == "G":    # right after the stores, so the loads overlap both GEMMs
        i = s2.index("half_t B_local[", k_loop)
        j = s2.rfind("{", 0, i)
    else:                 # H: after the K GEMM and its mask, before the softmax
        j = s2.index("m_prev[0] = m_i[0];", k_loop)
    nxt = f"""
    if (((k + 1) < ({n_kv})) && (c_v < {slots}) && ((((k + 1) * {c["bN"]}) + kk_v) < {c["L"]})) {{
{loads("k + 1")}
    }}
    """
    s2 = s2[:j] + nxt + s2[j:]
    print(f"[patch {what}] next-step load before V GEMM, slots {slots}", flush=True)
    return s2[:k_loop] + decl + s2[k_loop:]


def _barriers(s):
    """Y: W(K) W(V) | barrier | R(K) R(V) | barrier. All barriers of the K step are replaced by these two (generated placement varies by tile)."""
    k = s.index("for (int k = 0;")
    e = _block_end(s, k)
    body = s[k:e]
    n = body.count("__syncthreads();\n")
    body = body.replace("__syncthreads();\n", "")
    i = body.index("half_t B_local[")
    j = body.rfind("{", 0, i)
    body = body[:j] + "__syncthreads();\n    " + body[j:]
    body = body[:-1] + "  __syncthreads();\n  }"
    print(f"[patch Y] {n} barriers replaced by 2", flush=True)
    return s[:k] + body + s[e:]


def _stamps(s):
    """S: per-wave stamps (realtime 25 MHz, s_memtime) around the whole kernel, written into this block's own Q rows after the output, which no other block reads.
    One call only on a copy of Q. Slot: flat row r0 + wave of this block, first 16 values."""
    i = s.index("main_kernel(")
    b = s.index("{", i)
    e = _block_end(s, b)
    c = CTX
    rows = c.get("rows") or c["P"] * c["G"]
    NB = -(-c["L"] * c["G"] // rows)
    bx, by = ("(int)blockIdx.y", "(int)blockIdx.x") if c.get("swap") else ("(int)blockIdx.x", "(int)blockIdx.y")
    r0 = f"(({NB} - 1 - ({bx})) * {rows})" if c.get("rev") else f"(({bx}) * {rows})"
    fr = f"({r0} + (((int)threadIdx.x) >> 6))"
    pos = f"({fr} / {c['G']})"
    if c["B"] != 1:
        raise RuntimeError("stamps: B = 1 only")
    idx = f"((({pos} * {c['Hkv'] * c['G']}) + (({by}) * {c['G']}) + ({fr} % {c['G']})) * {c['D']})"
    end = f"""
  if ((((int)threadIdx.x) & 63) == 0 && {pos} < {c["L"]}) {{
    unsigned long t_c1 = __builtin_amdgcn_s_memtime();
    unsigned long t_r1 = __builtin_amdgcn_s_memrealtime();
    unsigned long* t_p = (unsigned long*)(Q + {idx});
    t_p[0] = t_r0; t_p[1] = t_r1; t_p[2] = t_c0; t_p[3] = t_c1;
  }}
"""
    start = "\n  unsigned long t_r0 = __builtin_amdgcn_s_memrealtime();\n  unsigned long t_c0 = __builtin_amdgcn_s_memtime();\n"
    print("[patch S] stamps in own Q rows", flush=True)
    return s[:b + 1] + start + s[b + 1:e - 1] + end + s[e - 1:]


def lds_events(s):
    """Ordered LDS accesses and barriers of one K step of the generated source: list of (kind, array)."""
    k = s.index("for (int k = 0;")
    body = s[k:_block_end(s, k)]
    ev = []
    for ln in body.split("\n"):
        if "__syncthreads" in ln:
            ev.append(("B", ""))
            continue
        for a in ("K_s", "V_s", "V_t"):
            if "half_t*)" + a in ln:
                lhs = ln.split(" = ")[0]
                ev.append(("W" if ("half_t*)" + a) in lhs else "R", a))
    return ev


def race_free(s):
    """Every write->read and read->write of one LDS array, cyclically over the K step, must cross a barrier. Threads write disjoint elements."""
    ev = lds_events(s)
    n = len(ev)
    bad = []
    for i, (t, a) in enumerate(ev):
        if t not in "WR":
            continue
        for d in range(1, n + 1):
            t2, a2 = ev[(i + d) % n]
            if t2 == "B":
                break
            if a2 == a and t2 != t:
                bad.append(f"{t}({a})->{t2}({a2})")
                break
    return (not bad), sorted(set(bad)), sum(1 for e in ev if e[0] == "B")


def apply(s, fl):
    if "S" in fl:
        s = _stamps(s)
    if "P" in fl:
        s = _vload2(s)
    if any(x in fl for x in "ZOQGH"):
        s = _zload(s, "4" in fl)
    if "O" in fl:
        s = _oload(s, "O")
    if "G" in fl:
        s = _oload(s, "O", "G")
    if "H" in fl:
        s = _oload(s, "O", "H")
    if "Q" in fl:
        s = _oload(s, "Q")
    if "Y" in fl:
        s = _barriers(s)
    if "K" in fl:
        s = _kuload(s, "K_s")
    if "X" in fl:
        s = _vload(s) if "((half_t*)V_t)[" in s else _kuload(s, "V_s")
    if "A" in fl:
        s, n = re.subn(r"__syncthreads\(\);\n(\s*m_i_clear\[[^\]]*\] = tl::AllReduce<tl::MaxOp, 64,)", r"\1", s)
        s, m = re.subn(r"(half_t B_local_1\[\d+\];\n\s*)__syncthreads\(\);\n", r"\1", s)
        print(f"[patch A] reduce barrier {n}, gemm2 barrier {m}", flush=True)
    if "C" in fl:
        # p_cast with v_cvt_pkrtz, 2 instructions per 4 values instead of 4 conversions and 2 packs. Rounds toward zero, as CK does.
        pat = r"float4 (v__\d+) = \*\(float4\*\)\(acc_s \+ \((i_\d+) \* 4\)\);\n\s*\*\(\(half_t\*\)\(&\(\(\(half2\*\)\(&\((__\d+)\.x\)\)\)->x\)\)\) = [^\n]*\n[^\n]*\n[^\n]*\n[^\n]*\n"
        s, n = re.subn(pat, lambda m: f"float4 {m.group(1)} = *(float4*)(acc_s + ({m.group(2)} * 4));\n      {m.group(3)}.x = __builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({m.group(1)}.x, {m.group(1)}.y));\n      {m.group(3)}.y = __builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({m.group(1)}.z, {m.group(1)}.w));\n", s)
        print(f"[patch C] pkrtz cast blocks {n}", flush=True)
    if "X" in fl and "((half_t*)V_t)[" not in s and False:
        pass
    if "E" in fl:
        # exp: scale folded with log2(e), v_fma + v_exp instead of v_mul + v_fma + v_exp on every S element
        k = s.index("for (int k = 0;")
        hit = 0
        for m in re.finditer(r"#pragma unroll", s[k:]):
            p = k + m.start()
            e = _block_end(s, p)
            blk = s[p:e]
            if "__expf(" in blk and "acc_s" in blk and "(m_i[0] * " in blk:
                sc = re.search(r"\(m_i\[0\] \* ([0-9.e+-]+f)\)", blk).group(1)
                nb = blk.replace(sc, f"({sc} * 1.442695041f)").replace("__expf(", "__builtin_amdgcn_exp2f(")
                s = s[:p] + nb + s[e:]
                hit += 1
                break
        print(f"[patch E] exp blocks {hit}", flush=True)
    if "B" in fl:
        m = re.search(r"row_sum\[0\] = tl::AllReduce<tl::SumOp, (\d+), (\d+), 0>::run\(row_sum\[0\], [^;]*;\n", s)
        if not m:
            print("[patch B] no row_sum[0] AllReduce, skipped", flush=True)
        else:
            th, sc = m.group(1), m.group(2)
            s = s[:m.start()] + s[m.end():]
            k = s.index("for (int k = 0;")
            e = _block_end(s, k)
            s = s[:e] + f"\n  l_i[0] = tl::AllReduce<tl::SumOp, {th}, {sc}, 0>::run(l_i[0], (float*)nullptr);\n" + s[e:]
            print(f"[patch B] moved row sum reduce ({th},{sc}) after K loop", flush=True)
    if "D" in fl:
        i = s.find("l_i[0] = (l_i[0] * sf[0]);\n")
        if i < 0:
            print("[patch D] pattern missing, skipped", flush=True)
        else:
            p = s.index("#pragma unroll", i)
            if "acc_o" in s[p:_block_end(s, p)]:
                e = _block_end(s, p)
                s = s[:p] + "if (sf[0] != 1.0f) {\n" + s[p:e] + "\n}\n" + s[e:]
                print("[patch D] gated acc_o rescale", flush=True)
            else:
                print("[patch D] next block is not acc_o, skipped", flush=True)
    return s


def install():
    fl = os.environ.get("N5_PATCH", "")
    if not fl:
        return
    from tilelang.contrib import hipcc
    orig = hipcc.compile_hip

    def comp(code, *a, **k):
        out = apply(code, fl)
        LAST["src"] = out
        ok, bad, nb = race_free(out)
        print(f"[patch check] race_free={ok} barriers={nb} {bad}", flush=True)
        if not ok and any(x in fl for x in "YPZOQGH"):
            raise RuntimeError(f"race check failed: {bad}")
        return orig(out, *a, **k)
    hipcc.compile_hip = comp
