"""Resource usage of compiled kernels at T = 4096 (764 blocks launched at bM = 64), clang 18 cache: tile and occupancy."""
import glob, json, os, re
seen = {}
for d in glob.glob("cache/**/device_kernel.cu", recursive=True):
    src = open(d).read()
    dd = os.path.dirname(d)
    host = open(dd + "/host_kernel.cu").read()
    v = [int(x) for x in re.findall(r"v_int64\) = \(\(int64_t\)(\d+)\)", host)]
    if len(v) < 6 or v[-5] != 764: continue
    kind = "down" if "W2" in src.split("main_kernel(")[1][:200] else "gate_up" if "W13" in src.split("main_kernel(")[1][:200] else None
    if not kind: continue
    r = json.load(open(dd + "/resource_usage.json"))["main_kernel"]["extra"]
    k = (kind, v[-6], v[-4], v[-1], r["VGPRs"], r["AGPRs"], r["Occupancy [waves/SIMD]"])
    seen[k] = 1
print("kernel cols threads dyn_LDS VGPRs AGPRs occupancy")
for k in sorted(seen): print(*k)
