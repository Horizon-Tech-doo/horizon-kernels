"""CPU. Instruction stream of bytes3/<n>_frozen against bytes/<n>."""
import glob, hashlib, json, os, re, struct, subprocess
OD = "/opt/rocm/llvm/bin/llvm-objdump"
def co(p):
    d = open(p, "rb").read(); i = d.find(b"__CLANG_OFFLOAD_BUNDLE__"); c = struct.unpack_from("<Q", d, i + 24)[0]; q = i + 32
    for _ in range(c):
        off, sz, tl = struct.unpack_from("<QQQ", d, q); q += 24; tr = d[q:q + tl].decode(errors="replace"); q += tl
        if "gfx90a" in tr and sz > 0:
            return d[i + off:i + off + sz]
def dis(p):
    t = f"/tmp/d128c_{os.getpid()}.co"; open(t, "wb").write(co(p)); o = subprocess.run([OD, "-d", "--no-show-raw-insn", t], capture_output=True, text=True).stdout; os.remove(t)
    return [x for x in (re.sub(r"\s*//.*", "", l).strip() for l in o.splitlines() if l.startswith((" ", "\t"))) if x]
for p in sorted(glob.glob("bytes3/*_frozen.hsaco")):
    n = os.path.basename(p)[:-len("_frozen.hsaco")]; a = dis(p); b = dis(f"bytes/{n}.hsaco")
    print(json.dumps(dict(name=n, same_insn=a == b, n=[len(a), len(b)], insn_md5=[hashlib.md5("\n".join(x).encode()).hexdigest()[:12] for x in (a, b)])))
