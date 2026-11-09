"""CPU. Instruction streams (llvm-objdump, gfx90a code object) of pairs of code objects. Usage: icmp.py A B [A B ...]"""
import hashlib, json, os, re, struct, subprocess, sys
OD = "/opt/rocm/llvm/bin/llvm-objdump"
def co(p):
    d = open(p, "rb").read(); i = d.find(b"__CLANG_OFFLOAD_BUNDLE__")
    if i < 0:
        return d
    c = struct.unpack_from("<Q", d, i + 24)[0]; q = i + 32
    for _ in range(c):
        off, sz, tl = struct.unpack_from("<QQQ", d, q); q += 24; tr = d[q:q + tl].decode(errors="replace"); q += tl
        if "gfx90a" in tr and sz > 0:
            return d[i + off:i + off + sz]
def dis(p):
    t = f"/tmp/jport_{os.getpid()}.co"; open(t, "wb").write(co(p)); o = subprocess.run([OD, "-d", "--no-show-raw-insn", t], capture_output=True, text=True).stdout; os.remove(t)
    return [x for x in (re.sub(r"\s*//.*", "", l).strip() for l in o.splitlines() if l.startswith((" ", "\t"))) if x]
def meta(p):
    t = f"/tmp/jport_{os.getpid()}.co"; open(t, "wb").write(co(p)); o = subprocess.run(["/opt/rocm/llvm/bin/llvm-readelf", "--notes", t], capture_output=True, text=True).stdout; os.remove(t)
    g = lambda k: (re.search(rf"\.{k}:\s+(\d+)", o) or [None, None])[1]  # noqa: E731
    return dict(vgpr=g("vgpr_count"), lds=g("group_segment_fixed_size"))
args = sys.argv[1:]
for x, y in zip(args[::2], args[1::2]):
    a, b = dis(x), dis(y)
    print(json.dumps(dict(a=os.path.basename(x), b=os.path.basename(y), same_insn=a == b, n=[len(a), len(b)], insn_md5=[hashlib.md5("\n".join(z).encode()).hexdigest()[:12] for z in (a, b)], meta=[meta(x), meta(y)])))
