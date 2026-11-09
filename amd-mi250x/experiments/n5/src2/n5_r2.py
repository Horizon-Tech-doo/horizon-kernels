"""N5 round 2, landed module only (gqa_fa V2). Lever 1: nops in the K-step load block, inserted after register allocation. Lever 2 companion: landed MHA arms.
Arms of a case run in one process with CK and SDPA. Forms:
  jit  gqa_flash_fast as landed
  ctl  landed source, hipcc -S, assembled unchanged (must match jit)
  n0   ctl with s_nop 0 after each of the first 3 of the 4 K-step loads (existing gap nops replaced)
  n3   same with s_nop 3
Usage: n5_r2.py --compile [--re R]             precompile on CPU, one cache per stack
       n5_r2.py --tag r1 --check --ck [--re R] [--inputs randn0] [--rep 50]
Never used: source-level __asm__ nops (C1: forces vmcnt waits)."""
import argparse, json, os, re, subprocess, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--compile", action="store_true"); ap.add_argument("--re", default="."); ap.add_argument("--tag", default="c"); ap.add_argument("--inputs", default="randn0")
ap.add_argument("--check", action="store_true"); ap.add_argument("--ck", action="store_true"); ap.add_argument("--rep", type=int, default=50)
a = ap.parse_args()
if a.compile:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Same explicit target on the login node and in the job, so a precompiled kernel is a cache hit in the job.
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import torch
import torch.nn.functional as F
import tilelang
from tilelang.contrib import hipcc
from tilelang.contrib.rocm import find_hipcc, find_rocm_path
from tilelang.contrib.hip_resource_info import filter_and_record, hipcc_remark_flag
from bench_common import bench
from n5lib import CASES, inputs, ref

OUT = os.environ.get("N5_R2_ASM", "/tmp/n5_r2_asm")
_orig = hipcc.compile_hip
NOP = {}          # {"name": arm name, "dose": None | 0 | 3}: set for ctl, n0, n3 only
COMPILES = [0]


def _is_ins(l):
    """Real instruction line: not a directive, label or comment."""
    l = l.strip()
    return bool(l) and not l.startswith((".", ";", "/")) and not l.endswith(":") and not re.match(r"^\S+:", l)


def load_block(lines):
    """Indices of the 4 global_load_dwordx4 of the K step: inside the loop (after the first 'Loop Header' label), at most 8 other instructions between loads, no label or branch in between."""
    start = next(i for i, l in enumerate(lines) if "Loop Header" in l or "in Loop:" in l)
    groups, cur, since = [], [], 0
    for i in range(start, len(lines)):
        l = lines[i].strip()
        if l.startswith("global_load_dwordx4"):
            cur.append(i); since = 0
        elif cur and (re.match(r"^\.?\w+:", l) or l.startswith(("s_cbranch", "s_branch"))):
            groups.append(cur); cur = []
        elif cur and _is_ins(l):
            since += 1
            if since > 8:
                groups.append(cur); cur = []
    if cur:
        groups.append(cur)
    g4 = [g for g in groups if len(g) == 4]
    if len(g4) != 1:
        raise RuntimeError(f"expected one group of 4 K-step loads, found groups {[len(g) for g in groups]}")
    return g4[0]


def pattern(lines, g):
    s = ""
    for i in range(g[0], g[-1] + 1):
        l = lines[i].strip()
        if not _is_ins(l):
            continue
        s += "L" if l.startswith("global_load") else ("n" + l.split()[1]) if l.startswith("s_nop") else "."
    return s


def edit(lines, dose):
    g = load_block(lines)
    before = pattern(lines, g)
    if dose is None:
        return lines, before, before
    drop = {i for i in range(g[0], g[-1]) if lines[i].strip().startswith("s_nop") and lines[i].split()[1] == "0"}   # a longer compiler nop is a hazard wait: keep it
    out = []
    for i, l in enumerate(lines):
        if i in drop:
            continue
        out.append(l)
        if i in g[:3]:
            out.append(f"\ts_nop {dose}\n")
    return out, before, pattern(out, load_block(out))


def asm_hook(code, target_format="hsaco", arch=None, options=None, path_target=None, verbose=False):
    COMPILES[0] += 1
    if not NOP:
        return _orig(code, target_format=target_format, arch=arch, options=options, path_target=path_target, verbose=verbose)
    os.makedirs(OUT, exist_ok=True)
    b = f"{OUT}/{NOP['name']}"
    open(b + ".cc", "w").write(code)
    cmd = [find_hipcc(), "-O3", "-S", "-gline-tables-only", f"--offload-arch={arch}", "--genco"] + list(options or []) + [hipcc_remark_flag(), "-o", b + ".raw.s", b + ".cc"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    filter_and_record(p.stdout)
    if p.returncode:
        raise RuntimeError(p.stdout[-2000:])
    lines, before, after = edit(open(b + ".raw.s").readlines(), NOP["dose"])
    open(b + ".s", "w").writelines(lines)
    binp = os.path.join(find_rocm_path(), "llvm", "bin")
    subprocess.check_call(f"{binp}/clang -x assembler --target=amdgcn-amd-amdhsa -mcpu=gfx90a -c {b}.s -o {b}.o && {binp}/ld.lld -shared {b}.o -o {b}.elf && "
                          f"{binp}/clang-offload-bundler -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx90a -input=/dev/null -input={b}.elf -output={b}.hsaco", shell=True)
    print(json.dumps(dict(asm=NOP["name"], before=before, after=after)), flush=True)
    return bytearray(open(b + ".hsaco", "rb").read())


hipcc.compile_hip = asm_hook          # below gqa_fa's hook: source edit first, then this
import gqa_fa
gqa_fa._install_hook()

FAST_FLAG = "-DGQA_FAST_N5_V2"
ARMS = []


def add(case, P, bN, th, rows, edge, rev, recip, swap, forms):
    for f in forms:
        fl = ("E" if edge else "") + ("R" if rev else "") + ("D" if recip else "") + ("W" if swap else "")
        ARMS.append(dict(name=f"{case}_{'r' + str(rows) if rows else 'P' + str(P)}_b{bN}_t{th}_{fl}_{f}", case=case, P=P, bN=bN, th=th, rows=rows, edge=edge, rev=rev, recip=recip, swap=swap, form=f))


for L in (4096, 8192, 16384):
    add(f"d128_{L}", 0, 32, 512, 128, True, True, True, True, ("jit", "ctl", "n0", "n3"))
    add(f"d128_{L}", 32, 96, 768, 0, True, True, True, True, ("jit", "ctl", "n3"))
add("d256_8192", 32, 32, 768, 0, True, True, True, True, ("jit", "ctl", "n3"))
# Lever 2 through the landed module (edge on, as the module requires)
add("mha_4096", 160, 32, 640, 0, True, False, False, False, ("jit",))
add("mha_4096", 160, 32, 640, 0, True, False, True, False, ("jit",))
add("mha_4096", 160, 32, 640, 0, True, False, True, True, ("jit",))
add("mha_4096", 160, 80, 640, 0, True, False, False, False, ("jit",))
pat = re.compile(a.re)
ARMS = [c for c in ARMS if pat.search(c["name"])]


def build(c):
    s = CASES[c["case"]]
    B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
    pos = (B, L, Hkv, G, D, causal, c["P"], c["bN"], c["th"])
    if c["form"] == "jit":
        return gqa_fa.gqa_flash_fast(*pos, 2, c["edge"], c["rev"], c["recip"], c["swap"], c["rows"])
    dose = {"ctl": None, "n0": 0, "n3": 3}[c["form"]]
    gqa_fa._FAST.clear(); gqa_fa._FAST.update(L=L, Hkv=Hkv, B=B, D=D, bN=c["bN"], threads=c["th"], swap=c["swap"])
    NOP.update(name=c["name"], dose=dose)
    try:
        f = gqa_fa.make(*pos, True, 2, c["edge"], c["rev"], c["recip"], c["swap"], c["rows"])
        return tilelang.compile(f, out_idx=[3], compile_flags=[FAST_FLAG, "-DN5_R2_" + c["form"]])
    finally:
        gqa_fa._FAST.clear(); NOP.clear()


if a.compile:
    for c in ARMS:
        t0 = time.time(); n0 = COMPILES[0]; r = dict(name=c["name"])
        try:
            build(c)
        except Exception as e:  # noqa: BLE001
            r["err"] = (type(e).__name__ + ": " + str(e))[-400:]
        r.update(compiles=COMPILES[0] - n0, t=round(time.time() - t0, 1))
        print(json.dumps(r), flush=True)
    sys.exit(0)

print(json.dumps(dict(info=1, torch=torch.__version__, hip=getattr(torch.version, "hip", None), tag=a.tag, n=len(ARMS), module_flag=FAST_FLAG)), flush=True)
rot = sum(map(ord, a.tag))
for case in sorted({c["case"] for c in ARMS}):
    s = CASES[case]
    B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
    flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
    rep = a.rep if L <= 8192 else max(20, a.rep // 4)
    for inp in a.inputs.split(","):
        Q, K, V = inputs(inp, s)
        R = ref(Q, K, V, causal, G) if a.check and L <= 8192 else None
        qt, kt, vt = Q.transpose(1, 2), K.transpose(1, 2).repeat_interleave(G, 1), V.transpose(1, 2).repeat_interleave(G, 1)
        base = dict(case=case, input=inp, tag=a.tag, stack="image")
        So = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2)
        ms = bench(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal), warmup=5, rep=max(10, rep // 2))
        print(json.dumps(dict(base, name=case + "_sdpa", impl="sdpa", ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))), flush=True)
        if a.ck:
            from flash_attn import flash_attn_func
            ms = bench(lambda: flash_attn_func(Q, K, V, causal=causal), warmup=10, rep=rep)
            print(json.dumps(dict(base, name=case + "_ck", impl="ck", ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))), flush=True)
        cs = [c for c in ARMS if c["case"] == case]
        rr = rot % max(1, len(cs))
        ref_out = {}
        for c in cs[rr:] + cs[:rr]:
            r = dict(base, name=c["name"], impl="tl", form=c["form"], P=c["P"], bN=c["bN"], th=c["th"], rows=c["rows"])
            n0 = COMPILES[0]; t0 = time.time()
            try:
                k = build(c)
                r["compiles"] = COMPILES[0] - n0
                O = k(Q, K, V)
                if a.check:
                    if R is not None:
                        r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 7)
                    r["rel_vs_sdpa"] = round(((O.float() - So.float()).norm() / So.float().norm()).item(), 7)
                    nrep = 30 if L <= 8192 else 10
                    r["repeat_equal"] = f"{sum(int(torch.equal(k(Q, K, V), O)) for _ in range(nrep))}/{nrep}"
                    key = (c["P"], c["bN"], c["th"], c["rows"], c["recip"], c["swap"])     # nops must not change a single bit
                    if key in ref_out:
                        r["bitwise_same_as_first_form"] = bool(torch.equal(ref_out[key], O))
                    else:
                        ref_out[key] = O.clone()
                del O
                ms = bench(lambda: k(Q, K, V), warmup=10, rep=rep)
                r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))
            except Exception as e:  # noqa: BLE001
                r["err"] = (type(e).__name__ + ": " + str(e))[-300:]
            r["wall_s"] = round(time.time() - t0, 1)
            print(json.dumps(r), flush=True)
        del Q, K, V, R, So, qt, kt, vt, ref_out
        torch.cuda.empty_cache()
