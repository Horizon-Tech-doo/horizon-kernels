"""Compile a TileLang-generated source with small text variants to find what removes scratch. Diagnosis only, login node, no GPU."""
import json, os, re, subprocess, sys, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import tilelang, tlfa
from tilelang.contrib.rocm import find_hipcc
from tilelang.env import TILELANG_TEMPLATE_PATH
case, P, bN, th, opts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rows = int(sys.argv[6]) if len(sys.argv) > 6 else 0
kind, L = case.split("_"); L = int(L)
Hkv, G, D, causal = (8, 1, 128, False) if kind == "mha" else (4, 6, int(kind[1:]), True)
src = tlfa.fwd(1, L, Hkv, G, D, causal, P, bN, th, opts, 2, True, True, True, True, rows).get_kernel_source()
src = re.sub(r'^#line.*\n', '', src, flags=re.M)
def comp(name, s, keep=False):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "k.cc"); open(f, "w").write(s)
        o = os.path.join(d, "k.s")
        r = subprocess.run([find_hipcc(), "-O3", "-S", "--cuda-device-only", "--offload-arch=gfx90a", "-std=c++17", "-I" + TILELANG_TEMPLATE_PATH, "-Rpass-analysis=kernel-resource-usage", "-o", o, f], capture_output=True, text=True)
        t = r.stdout + r.stderr
        res = {k: re.search(r"; " + k + r": (\d+)", open(o).read()).group(1) for k in ("NumVgprs", "NumAgprs", "ScratchSize", "Occupancy")} if os.path.exists(o) else {}
        asm = open(o).read() if os.path.exists(o) else ""
        print(name, res, "rc", r.returncode, "scratch ops", len(re.findall(r"buffer_(?:load|store)_dword.*s\[0:3\]", asm)), "\n" + ("" if r.returncode == 0 else t[-600:]))
        if keep: open(os.path.join(os.environ["E"], f"c2_{name}.s"), "w").write(asm)
comp("asis", src, True)
s = re.sub(r"half_t ([KV]_r)\[16\];", r"uint4 \1[2];", src)
s = re.sub(r"\(uint4\*\)\(([KV]_r) \+ \((\w+) \* 8\)\)", r"(\1 + \2)", s)
comp("uint4_arr", s)
print(re.findall(r".*K_r.*", s)[:3])
# ext vectors
s = re.sub(r"half_t ([KV]_r)\[16\];", r"typedef unsigned __attribute__((ext_vector_type(4))) u4v; u4v \1[2];", src, count=1)
s = re.sub(r"half_t ([KV]_r)\[16\];", r"u4v \1[2];", s)
s = re.sub(r"\*\(uint4\*\)\(([KV]_r) \+ \((\w+) \* 8\)\) = \*\(uint4\*\)\((.*)\);", r"\1[\2] = *(u4v*)(\3);", s)
s = re.sub(r"= \*\(uint4\*\)\(([KV]_r) \+ \((\w+) \* 8\)\);", r"= __builtin_bit_cast(uint4, \1[\2]);", s)
comp("extvec", s, True)
print(re.findall(r".*K_r.*", s)[:4])
open(os.path.join(os.environ["E"], "c2_extvec.cc"), "w").write(s)
s = "typedef unsigned __attribute__((ext_vector_type(4))) u4v;\n" + src
s = s.replace("#include <hip/hip_runtime.h>", "#include <hip/hip_runtime.h>\n", 1)
s = re.sub(r"\*\(uint4\*\)\(([KV]_r) \+ \((\w+) \* 8\)\) = \*\(uint4\*\)\((.*)\);", r"*(u4v*)(\1 + (\2 * 8)) = *(u4v*)(\3);", s)
s = re.sub(r"= \*\(uint4\*\)\(([KV]_r) \+ \((\w+) \* 8\)\);", r"= __builtin_bit_cast(uint4, *(u4v*)(\1 + (\2 * 8)));", s)
comp("accessor_half_arr", s)
