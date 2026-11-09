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


def apply(s, fl):
    if "K" in fl:
        s = _kuload(s, "K_s")
    if "X" in fl:
        s = _vload(s) if "((half_t*)V_t)[" in s else _kuload(s, "V_s")
    if "A" in fl:
        s, n = re.subn(r"__syncthreads\(\);\n(\s*m_i_clear\[[^\]]*\] = tl::AllReduce<tl::MaxOp, 64,)", r"\1", s)
        s, m = re.subn(r"(half_t B_local_1\[\d+\];\n\s*)__syncthreads\(\);\n", r"\1", s)
        print(f"[patch A] reduce barrier {n}, gemm2 barrier {m}", flush=True)
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
    fl = os.environ.get("E3_PATCH", "")
    if not fl:
        return
    from tilelang.contrib import hipcc
    orig = hipcc.compile_hip

    def comp(code, *a, **k):
        return orig(apply(code, fl), *a, **k)
    hipcc.compile_hip = comp
