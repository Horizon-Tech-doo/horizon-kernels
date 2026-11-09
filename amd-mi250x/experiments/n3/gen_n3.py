#!/usr/bin/env python3
"""N3: patch TileLang's cached paged-decode kernels. usage: gen_n3.py <out.cu> <arms.txt> B L S TH BM ARM[,ARM..]
ARM tokens joined by '+': base, ns (no bounds checks), b (one load round), nb (only post-load + end-of-step barriers),
w1 (no math on wave>=1, one post-math barrier), bt1 (next-page table entry loaded with the K/V round), nob (no barriers at all, 1 wave only),
ld (load only, no LDS), ldl (load + LDS write + barrier), ck (clock stamps, extra Tr arg)."""
import glob, json, re, sys
R = "/scratch/project_XXXXXXXXX/tilelang/cache/0.1.15/linux-x86_64/kernels/"
exp = "/scratch/project_XXXXXXXXX/tilelang/exp/N3/gen/"

def find_base(S, TH, BM, MAXP=512):
    H0 = S * 256 * 6
    pps = -(-MAXP // S)
    hits = []
    for f in glob.glob(R + "*/device_kernel.cu"):
        s = open(f).read()
        if f"__launch_bounds__({TH}) main_kernel(int* __restrict__ BT" not in s: continue
        if "blk < 16384" not in s: continue
        if f"half_t q[{TH//2 if TH<=128 else 64}]" not in s and TH != 64: continue
        if f"((int)blockIdx.y) * {H0})" in s and f"blockIdx.x) * {pps}) + {pps}), ((n" in s: hits.append(f)
    if not hits: raise SystemExit(f"no base for S={S} TH={TH}")
    return open(sorted(hits)[0]).read(), sorted(hits)[0].split("/")[-2][:12]

LOOP = re.compile(r"#pragma unroll\n\s*for \(int (i_\d+) = 0; \1 < (\d+); \+\+\1\) \{\n\s*half_t broadcast_var_\d+ = half_t\(0\.000000e\+00f\);\n\s*uint4 (condval_\d+);\n\s*if \(\(\(blk < \d+\) && \(0 <= blk\)\)\) \{\n\s*\3 = \*\(uint4\*\)\((K|V) \+ (.*?)\);\n\s*\} else \{.*?\}\n\s*\*\(uint4\*\)\(\(\(half_t\*\)(K_s|V_s)\) \+ (.*?)\) = \3;\n\s*\}\n\s*__syncthreads\(\);\n", re.S)

def sub_i(e, v, n): return re.sub(r"\b%s\b" % v, str(n), e)

def make(src, name, toks, dyn):
    s = src[src.index('extern "C"'):]
    s = re.sub(r"__launch_bounds__\((\d+)\)\s+main_kernel\(", lambda m: f"__launch_bounds__({m.group(1)}) {name}(", s)
    TH = int(re.search(r"__launch_bounds__\((\d+)\)", s).group(1))
    if "base" in toks:
        return s, TH
    ms = list(LOOP.finditer(s))
    assert len(ms) == 2, len(ms)
    start, end = ms[0].start(), ms[1].end()
    info = []
    for m in ms:
        v, n, _, arr, ga, _, la = m.group(1), int(m.group(2)), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)
        info.append((arr, ga, la, v, n))
    if "ld" in toks or "ldl" in toks: toks = toks | {"ns", "b"}
    if {"b", "bt1"} & toks: toks = toks | {"ns"}
    # table lookup
    mbt = re.search(r"int condval_1;\n\s*if \((.*?)\) \{\n\s*condval_1 = BT\[(.*?)\];\n\s*\} else \{\n\s*condval_1 = 0;\n\s*\}\n\s*int blk = condval_1;\n", s, re.S)
    if mbt is None:
        mbt = re.search(r"int blk = BT\[()(.*?)\];\n", s, re.S)
    bt_expr = mbt.group(2)
    # loads
    def lds_writes(reg="r"):
        w = []
        for arr, ga, la, v, n in info:
            for i in range(n):
                w.append(f"*(uint4*)(((half_t*)buf_{arr}) + {sub_i(la, v, i)}) = {arr}_{i};")
        return "\n    ".join(w)
    loads = []
    for arr, ga, la, v, n in info:
        for i in range(n):
            loads.append(f"uint4 {arr}_{i} = *(uint4*)({arr} + {sub_i(ga, v, i)});")
    loads = "\n    ".join(loads)
    wr = lds_writes().replace("buf_K", "K_s").replace("buf_V", "V_s")
    if "ns" in toks:
        s = s[:mbt.start()] + "int blk = BT[" + bt_expr + "];\n" + s[mbt.end():]
        ms = list(LOOP.finditer(s))
        start, end = ms[0].start(), ms[1].end()
        if "b" in toks:
            newload = loads + "\n    " + wr + "\n    __syncthreads();\n"
        else:
            parts = []
            for arr, ga, la, v, n in info:
                for i in range(n):
                    parts.append(f"{{ uint4 x_ = *(uint4*)({arr} + {sub_i(ga, v, i)}); *(uint4*)(((half_t*){arr}_s) + {sub_i(la, v, i)}) = x_; }}")
                parts.append("__syncthreads();")
            newload = "\n    ".join(parts) + "\n"
        s = s[:start] + newload + s[end:]
    if "bt1" in toks:
        # blk of this step comes from the register filled in the previous round; first one read before the loop
        forhead = re.search(r"\n\s*for \(int t = 0; t < (.*?); \+\+t\) \{\n", s)
        pre = "\n  int blk_next_ = BT[" + sub_i(bt_expr, "t", 0) + "];"
        s = s[:forhead.start()] + pre + s[forhead.start():]
        s = s.replace("int blk = BT[" + bt_expr + "];\n", "int blk = blk_next_;\n    blk_next_ = BT[(" + re.sub(r"\bt\b", "(t + 1)", bt_expr) + ")];\n", 1) if False else s
        s = s.replace("int blk = BT[" + bt_expr + "];\n", "int blk = blk_next_;\n", 1)
        # load the next entry inside the K/V round (its result is used only on the next iteration)
        s = s.replace(loads, "int nxt_ = BT[" + re.sub(r"\bt\b", "(t + 1)", bt_expr) + "];\n    " + loads, 1)
        s = s.replace(wr + "\n", wr + "\n    blk_next_ = nxt_;\n", 1)
    if "ld" in toks or "ldl" in toks:
        i0 = s.index(loads) + len(loads)
        i1 = s.index("  if (((((((int)threadIdx.x) >> 6) * 8)") if "((int)threadIdx.x) >> 6) * 8) +" in s else None
        tail = re.search(r"\n  \}\n  if \(", s[i0:])
        keep = "\n    " + " ".join(f"xa_ ^= {a}_{i}.x ^ {a}_{i}.y ^ {a}_{i}.z ^ {a}_{i}.w;" for a, _, _, _, n in info for i in range(n))
        if "ldl" in toks:
            keep = "\n    " + wr + "\n    __syncthreads();\n" + keep
        body = s[:i0] + keep + "\n  }\n  if (threadIdx.x == 0) { Lse[(((int)blockIdx.y) * SS_) + ((int)blockIdx.x)] = (float)xa_; }\n}\n"
        forhead = re.search(r"\n\s*for \(int t = 0; t < ", body)
        body = body[:forhead.start()] + "\n  unsigned xa_ = 0;" + body[forhead.start():]
        s = body
        # drop setup we no longer need
    else:
        # barrier surgery after the load section
        mb1 = "    __syncthreads();\n    m_i_clear[0] = tl::AllReduce"
        mb2 = "      __syncthreads();\n      for (int j = 0;"
        nb = "nb" in toks or "w1" in toks or "nob" in toks
        if nb:
            assert mb1 in s and mb2 in s, name
            s = s.replace(mb1, "    m_i_clear[0] = tl::AllReduce", 1).replace(mb2, "      for (int j = 0;", 1)
        end_barrier = (("b" in toks) or ("w1" in toks)) and "nob" not in toks
        math0 = re.search(r"\n\s*#pragma unroll\n\s*for \(int i_\d+ = 0; i_\d+ < \d+; \+\+i_\d+\) \{\n\s*float condval_4;", s)
        assert math0, name
        i0 = math0.start(); i1 = s.find("\n  }\n  if (", i0)
        assert i1 > 0, name
        math = s[i0:i1]
        if "w1" in toks:
            math = "\n    if (((int)threadIdx.x) < 64) {" + math + "\n    }"
        if end_barrier:
            math += "\n    __syncthreads();"
        s = s[:i0] + math + s[i1:]
    if "vf" in toks:   # TIMING ONLY, wrong answer: V read from LDS as 16 contiguous 8-byte lane reads instead of 64 2-byte reads
        i0 = s.index("half_t B_local_1[64];"); i1 = s.index("for (int j", i0); i2 = s.index("for (int j_1", i1)
        s = s[:i1] + "for (int j = 0; j < 16; ++j) { *(uint2*)(B_local_1 + j*4) = *(uint2*)(((half_t*)V_s) + j*256 + (((int)threadIdx.x)&63)*4); }\n      " + s[i2:]
    if "rc" in toks:   # one division per row, multiply per element
        s, n1 = re.subn(r"\((v__\d+\.[xyzw])/(v__\d+\.[xyzw])\)", r"(\1*\2)", s)
        s, n2 = re.subn(r"make_float4\(l_i\[0\], l_i\[0\], l_i\[0\], l_i\[0\]\)", "make_float4(rc_, rc_, rc_, rc_)", s)
        assert n1 == 4 and n2 == 1, (n1, n2)
        i = s.rfind("\n  if (", 0, s.index("rc_, rc_"))
        s = s[:i] + "\n  float rc_ = 1.0f / l_i[0];" + s[i:]
    if "nob" in toks:
        s = s.replace("__syncthreads();", "")
    if "hnd" in toks:   # timing-only / data rearranged: [block, head, key, dim] instead of [block, key, head, dim]
        s = re.sub(r"\(\(\(\(int64_t\)blk\) \* \(int64_t\)16384\) \+ \(\(\(int64_t\)(i_\d+|\d+)\) \* \(int64_t\)(\d+)\)\)", lambda m: f"(((int64_t)blk) * (int64_t)16384) + (((int64_t){m.group(1)}) * (int64_t){int(m.group(2))//4})", s)
        s = s.replace("(int64_t)5) * (int64_t)1024)", "(int64_t)5) * (int64_t)256)").replace("(int64_t)4) * (int64_t)1024)", "(int64_t)4) * (int64_t)256)").replace("(int64_t)6) * (int64_t)1024)", "(int64_t)6) * (int64_t)256)")
        s = s.replace("(int64_t)3) * (int64_t)256)", "(int64_t)3) * (int64_t)4096)")
    if "swap" in toks:
        s = s.replace("blockIdx.x", "blockIdx.@").replace("blockIdx.y", "blockIdx.x").replace("blockIdx.@", "blockIdx.y")
    if "ck" in toks:
        s = s.replace("half_t* __restrict__ V) {", "half_t* __restrict__ V, unsigned long long* __restrict__ Tr) {\n  unsigned long long r0_=__builtin_amdgcn_s_memrealtime(), m0_=__builtin_amdgcn_s_memtime(); unsigned hw_=(unsigned)__builtin_amdgcn_s_getreg(63492);", 1)
        last = s.rstrip().rfind("}")
        s = s[:last] + "\n  if (((int)threadIdx.x)==0) { unsigned long long r1_=__builtin_amdgcn_s_memrealtime(), m1_=__builtin_amdgcn_s_memtime(); int bk_=(((int)blockIdx.y)*SS_+((int)blockIdx.x))*5; Tr[bk_]=r0_; Tr[bk_+1]=r1_; Tr[bk_+2]=m0_; Tr[bk_+3]=m1_; Tr[bk_+4]=hw_; }\n}\n"
    return s, TH

def k2src(S):
    src = open(glob.glob(R + "7153f106*/device_kernel.cu")[0]).read()
    src = src[src.index('extern "C"'):]
    s = src.replace("main_kernel(", f"k2_{S}(", 1)
    s = re.sub(r"\* 13\)", f"* {S})", s); s = re.sub(r"< 13;", f"< {S};", s); s = s.replace("* 3328)", f"* {S*256})")
    return s

def main():
    out, B, L, plan = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), json.loads(sys.argv[4])
    hdr = None; bodies = []; arms = []; seen_k2 = set()
    for a in plan:
        toks = set(a["toks"].split("+")); S, TH, BM = a["S"], a["TH"], a["BM"]
        src, h = find_base(S, TH, BM, L // 16) if "gen" not in a else (open(a["gen"]).read(), "gen")
        if hdr is None: hdr = src[:src.index('extern "C"')]
        nm = a["name"]
        b, th = make(src, nm, toks, a["dyn"])
        b = b.replace("SS_", str(S))
        bodies.append(b)
        arms.append({"name": nm, "S": S, "th": th, "dyn": a["dyn"], "toks": a["toks"], "base": h, "ck": "ck" in toks, "swap": "swap" in toks})
        if S not in seen_k2:
            seen_k2.add(S); bodies.append(k2src(S))
    open(out, "w").write(hdr + "\n".join(bodies))
    json.dump(arms, open(out + ".json", "w"), indent=0)
    print(json.dumps(arms))
main()
