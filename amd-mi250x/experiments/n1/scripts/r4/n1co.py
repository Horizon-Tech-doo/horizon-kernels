"""Build one FusedMoE under this process's compiler and save all five gfx90a code objects plus launch geometry.
Usage: n1co.py OUT T TAG. Needs a TILELANG_CACHE_DIR used by this process only, so each kernel appears once there.
TAG: ship (tiles_for default of this compiler), small, mid, large, each optionally +uN for gate/up unroll threshold N."""
import glob, json, os, re, shutil, struct, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
out, Tn, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
import moe_fused as M
base, _, u = tag.partition("+u")
cfg = {"ship": None, "small": dict(M._SMALL), "mid": dict(M._MID), "large": dict(M._LARGE)}[base]
if cfg is not None:
    cfg["gu_unroll"] = int(u) if u else None
t0 = time.time()
m = M.FusedMoE(Tn, cfg=cfg)
dt = time.time() - t0
d = f"{out}/T{Tn}_{tag}"; os.makedirs(d, exist_ok=True)
meta = dict(T=Tn, N=m.N, P=m.P, NB=m.NB, E=m.E, H=m.H, I=m.I, cfg=m.cfg, unroll=m.unroll, clang=M.clang_major(), build_s=round(dt, 1), kernels={})
for f in glob.glob(os.environ["TILELANG_CACHE_DIR"] + "/**/device_kernel.cu", recursive=True):
    src = open(f).read(); dd = os.path.dirname(f)
    args = re.findall(r"\*\s*__restrict__\s+(\w+)", re.search(r"main_kernel\((.*?)\)\s*\{", src).group(1))
    k = {("Bexp", "Cur", "Ids", "Sid"): "ka", ("Cur", "Ids", "Sid"): "kb", ("Bexp", "Inter", "Sid", "W13", "X"): "k1",
         ("Bexp", "Inter", "Sid", "W2", "Wt", "Y"): "k2", ("O", "Y"): "ks"}.get(tuple(args))
    if k is None: continue
    if k in meta["kernels"]: raise SystemExit(f"two builds of {k}: use a fresh cache dir per process")
    lit = [int(x) for x in re.findall(r"v_int64\) = \(\(int64_t\)(\d+)\)", open(dd + "/host_kernel.cu").read())][1:]   # first is the device type
    if k in ("k1", "k2"):
        assert len(lit) == 6 and lit[3:5] == [1, 1], lit
        launch = [lit[0], lit[1], lit[2], lit[5]]
    else:
        assert len(lit) in (4, 5) and lit[2:4] == [1, 1], lit
        launch = [lit[0], 1, lit[1], lit[4] if len(lit) == 5 else 0]
    data = open(dd + "/executable.so", "rb").read()
    i = data.find(b"__CLANG_OFFLOAD_BUNDLE__"); saved = False
    while i >= 0 and not saved:
        c = struct.unpack_from("<Q", data, i + 24)[0]; p = i + 32
        for _ in range(c):
            off, sz, tl = struct.unpack_from("<QQQ", data, p); p += 24
            tr = data[p:p + tl].decode(errors="replace"); p += tl
            if "gfx90a" in tr and sz > 0:
                open(f"{d}/{k}.co", "wb").write(data[i + off:i + off + sz]); saved = True
        i = data.find(b"__CLANG_OFFLOAD_BUNDLE__", i + 24)
    assert saved, k
    meta["kernels"][k] = dict(args=args, launch=launch, res=json.load(open(dd + "/resource_usage.json"))["main_kernel"]["extra"])
assert len(meta["kernels"]) == 5, list(meta["kernels"])
json.dump(meta, open(f"{d}/meta.json", "w"))
print(json.dumps({"kind": "co", "T": Tn, "tag": tag, "clang": meta["clang"], "build_s": meta["build_s"], "cfg": m.cfg, "unroll": m.unroll, "launch": {k: v["launch"] for k, v in meta["kernels"].items()}, "occ": {k: v["res"].get("Occupancy [waves/SIMD]") for k, v in meta["kernels"].items()}}), flush=True)
