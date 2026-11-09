"""CPU. For each kernel in the r2 cache: pull the amdgcn ELF out of executable.so, disassemble, print the K-step load pattern and a hash of the instruction stream. jit and ctl of one tile should have the same hash."""
import glob, hashlib, json, os, re, struct, subprocess, sys
BIN = sys.argv[2]; tmp = sys.argv[3]; os.makedirs(tmp, exist_ok=True)
for kd in sorted(glob.glob(sys.argv[1] + "/**/executable.so", recursive=True)):
    so = open(kd, "rb").read(); elf = None; i = 0
    while True:
        i = so.find(b"\x7fELF", i + 1)
        if i < 0: break
        if struct.unpack("<H", so[i + 18:i + 20])[0] == 0xE0:
            shoff, = struct.unpack("<Q", so[i + 40:i + 48]); shes, shn = struct.unpack("<HH", so[i + 58:i + 62])
            elf = so[i:i + shoff + shes * shn]; break
    d = os.path.dirname(kd); key = os.path.basename(d)[:10]
    src = open(d + "/device_kernel.cu").read()
    m = re.search(r"Tensor|tl_[a-z]+", src)
    if elf is None:
        print(json.dumps(dict(key=key, err="no amdgcn ELF"))); continue
    p = f"{tmp}/{key}.elf"; open(p, "wb").write(elf)
    dis = subprocess.run([BIN + "/llvm-objdump", "-d", "--mcpu=gfx90a", p], capture_output=True, text=True).stdout.splitlines()
    ins = [re.sub(r"\s*//.*", "", l).strip() for l in dis if "//" in l and not l.strip().startswith("//")]
    while ins and ins[-1].startswith("s_code_end"): ins.pop()
    last = max((i for i, l in enumerate(ins) if l.startswith("s_endpgm")), default=len(ins)); ins = ins[:last + 1]
    h = hashlib.md5("\n".join(ins).encode()).hexdigest()[:10]
    idx = [i for i, l in enumerate(ins) if l.startswith("global_load_dwordx4")]
    groups = []; cur = []
    for i in idx:
        if cur and i - cur[-1] > 5:
            groups.append(cur); cur = []
        cur.append(i)
    groups.append(cur)
    pats = ["".join("L" if ins[j].startswith("global_load") else "n3" if ins[j] == "s_nop 3" else "n" if ins[j].startswith("s_nop") else "." for j in range(g[0], g[-1] + 1)) for g in groups]
    rs = json.load(open(d + "/resource_usage.json")); e = list(rs.values())[0]["extra"] if rs else {}
    print(json.dumps(dict(key=key, hash=h, n_ins=len(ins), load_groups=pats, V=e.get("VGPRs"), scr=e.get("ScratchSize [bytes/lane]"), c_v="c_v" in src)), flush=True)
    os.remove(p)
