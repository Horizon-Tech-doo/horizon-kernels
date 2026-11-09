"""Clang 18 tile A at T = 16 with unroll 20000: the self-test built it first, so the set s dump saw a cache hit. Take it from the cache: gate/up, 512 MFMA, 320 dwordx4 loads."""
import glob, os, struct, subprocess, sys
cache, out = sys.argv[1:3]
for f in glob.glob(cache + "/**/device_kernel.cu", recursive=True):
    if "half_t* __restrict__ W13" not in open(f).read(): continue
    data = open(os.path.dirname(f) + "/executable.so", "rb").read()
    i = data.find(b"__CLANG_OFFLOAD_BUNDLE__")
    while i >= 0:
        c = struct.unpack_from("<Q", data, i + 24)[0]; p = i + 32
        for _ in range(c):
            off, sz, tl = struct.unpack_from("<QQQ", data, p); p += 24
            tr = data[p:p + tl].decode(errors="replace"); p += tl
            if "gfx90a" in tr and sz > 0:
                tmp = out + "/tmpc.co"; open(tmp, "wb").write(data[i + off:i + off + sz])
                d = subprocess.run(["/opt/rocm/llvm/bin/llvm-objdump", "-d", tmp], capture_output=True, text=True).stdout
                os.remove(tmp)
                if d.count("v_mfma") == 512 and d.count("global_load_dwordx4") == 320:
                    open(out + "/T_A_u20k_c18.dis", "w").write(d); print("found", os.path.dirname(f)[-20:])
        i = data.find(b"__CLANG_OFFLOAD_BUNDLE__", i + 24)
