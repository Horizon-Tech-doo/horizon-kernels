"""dasm.py: gfx90a disassembly of decode kernels (k1 at 512 pages per sequence, k2 combine) from the C2P caches, opcode histogram and load runs, wheel vs fork. CPU only."""
import glob, os, re, struct, subprocess, sys, collections, json
OUT = "/scratch/project_XXXXXXXXX/tilelang/exp/C2P/asm3"; os.makedirs(OUT, exist_ok=True)
def co(dd, name):
    data = open(dd + "/executable.so", "rb").read()
    i = data.find(b"__CLANG_OFFLOAD_BUNDLE__")
    while i >= 0:
        c = struct.unpack_from("<Q", data, i + 24)[0]; p = i + 32
        for _ in range(c):
            off, sz, tl = struct.unpack_from("<QQQ", data, p); p += 24
            tr = data[p:p + tl].decode(errors="replace"); p += tl
            if "gfx90a" in tr and sz > 0:
                t = f"{OUT}/{name}.co"; open(t, "wb").write(data[i + off:i + off + sz])
                dis = subprocess.run(["/opt/rocm/llvm/bin/llvm-objdump", "-d", t], capture_output=True, text=True).stdout
                os.remove(t); open(f"{OUT}/{name}.dis", "w").write(dis); return dis
        i = data.find(b"__CLANG_OFFLOAD_BUNDLE__", i + 24)
def hist(dis):
    ops = [l.split()[0] for l in dis.splitlines() if l.startswith("\t") or l.startswith(" ")]
    return collections.Counter(ops)
for root, tree in (("Iw", "wheel"), ("I", "fork")):
    for p in glob.glob(f"/scratch/project_XXXXXXXXX/tilelang/exp/C2P/cache/{root}/**/device_kernel.cu", recursive=True):
        s = open(p).read(); d = os.path.dirname(p)
        k1 = "__launch_bounds__(128)" in s and ">> 2) * 512" in s and "half_t* __restrict__ K" in s
        k2 = "__launch_bounds__(256)" in s and "Lse" in s and "Op" in s and "half_t* __restrict__ O" in s and "half_t* __restrict__ K" not in s
        if not (k1 or k2): continue
        exp_ = "acc" if " exp(" in s or "::exp(" in s else "fast"
        nm = f"{tree}_{'k1' if k1 else 'k2'}_{exp_}_{os.path.basename(d)[:6]}"
        dis = co(d, nm); h = hist(dis)
        keys = [k for k in h if re.search(r"load|store|ds_|waitcnt|barrier|flat|mfma|scratch", k)]
        print(nm, "total", sum(h.values()), {k: h[k] for k in sorted(keys)}, json.load(open(d + "/resource_usage.json"))["main_kernel"]["extra"]["VGPRs"])
