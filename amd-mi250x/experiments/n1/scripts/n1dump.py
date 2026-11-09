"""Extract gfx90a code objects of gate/up kernels at T=4096 from a kernel cache; save .co + disassembly + launch geometry."""
import glob, json, os, re, struct, subprocess, sys
cache, tag, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
sig = "main_kernel(int* __restrict__ Bexp, half_t* __restrict__ Inter, int* __restrict__ Sid, half_t* __restrict__ W13, half_t* __restrict__ X)"
for d in glob.glob(cache + "/**/device_kernel.cu", recursive=True):
    src = open(d).read()
    if sig not in src: continue
    dd = os.path.dirname(d)
    host = open(dd + "/host_kernel.cu").read()
    v = [int(x) for x in re.findall(r"v_int64\) = \(\(int64_t\)(\d+)\)", host)]
    if not (len(v) >= 6): continue
    grid = v[-6:]
    if grid[1] != 764: continue   # T=4096, bM=64
    data = open(dd + "/executable.so", "rb").read()
    i = data.find(b"__CLANG_OFFLOAD_BUNDLE__")
    while i >= 0:
        c = struct.unpack_from("<Q", data, i + 24)[0]; p = i + 32
        for _ in range(c):
            off, sz, tl = struct.unpack_from("<QQQ", data, p); p += 24
            tr = data[p:p + tl].decode(errors="replace"); p += tl
            if "gfx90a" in tr and sz > 0:
                co = data[i + off:i + off + sz]
                tmp = f"{out}/tmp{os.getpid()}.co"; open(tmp, "wb").write(co)
                dis = subprocess.run(["/opt/rocm/llvm/bin/llvm-objdump", "-d", tmp], capture_output=True, text=True).stdout
                mf = dis.count("v_mfma")
                vars_ = sorted(set(re.findall(r"void\* (\w+) =", src)))
                has_real = "real" in vars_
                mask = len(re.findall(r"if \(", src))
                name = f"{tag}_{'last' if has_real else 'zero'}_if{mask}_m{mf}"
                os.rename(tmp, f"{out}/{name}.co")
                open(f"{out}/{name}.dis", "w").write(dis)
                open(f"{out}/{name}.json", "w").write(json.dumps({"grid": grid, "src_sha": __import__('hashlib').sha256(src.encode()).hexdigest()[:12], "mfma": mf, "load": dis.count("global_load"), "bar": dis.count("s_barrier"), "res": json.load(open(dd + "/resource_usage.json"))}))
                print(name, grid, mf, flush=True)
        i = data.find(b"__CLANG_OFFLOAD_BUNDLE__", i + 24)
