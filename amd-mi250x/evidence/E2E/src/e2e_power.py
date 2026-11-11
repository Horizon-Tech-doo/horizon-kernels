"""E2E power sampler, runs on the compute node host (Python 3.6). python3 e2e_power.py <out.csv> <meta.json> [hz]
Each row: wall time, the four Cray pm_counters accelerator energy counters (one per MI250X card = 2 GCDs) with the counter's own timestamp,
node / cpu / memory energy counters, and for each GCD of this job the amdgpu hwmon package power (reported on the primary die of a card only),
the sclk reading (freq1_input: the driver's current target level, not a measured frequency) and gpu_busy_percent.
No device identifiers are written: GCDs are g0, g1 in the job's own order; meta.json only says whether they are the two dies of one card."""
import glob, json, os, subprocess, sys, time
out, meta = sys.argv[1], sys.argv[2]
hz = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
PM = "/sys/cray/pm_counters/"
def rd(p):
    try:
        with open(p) as f: return f.read().split()
    except (OSError, IOError):
        return []
bdfs = []
try:
    j = json.loads(subprocess.check_output(["rocm-smi", "--showbus", "--json"]).decode())
    bdfs = [j[k]["PCI Bus"].lower() for k in sorted(j) if k.startswith("card")]
except Exception as e:
    print("rocm-smi failed", repr(e), flush=True)
dev = ["/sys/bus/pci/devices/" + b for b in bdfs]
def hw(d, name):
    g = glob.glob(d + "/hwmon/hwmon*/" + name)
    return g[0] if g else None
pw = [hw(d, "power1_average") for d in dev]
fq = [hw(d, "freq1_input") for d in dev]
buses = [int(b.split(":")[1], 16) for b in bdfs]
same = len(buses) == 2 and abs(buses[0] - buses[1]) == 5 and sum(p is not None for p in pw) == 1
json.dump({"gcds": len(bdfs), "gcds_with_power_reading": sum(p is not None for p in pw), "two_dies_of_one_card": bool(same), "hz": hz,
           "pm_version": rd(PM + "version"), "pm_raw_scan_hz": rd(PM + "raw_scan_hz"), "accel_power_cap_W": [rd(PM + "accel%d_power_cap" % i)[:1] for i in range(4)],
           "hwmon_power_cap_uW": [rd(hw(d, "power1_cap") or "/x")[:1] for d in dev]}, open(meta, "w"))
cols = ["t"] + [c for i in range(4) for c in ("a%d_J" % i, "a%d_us" % i, "a%d_W" % i)] + ["node_J", "node_us", "node_W", "cpu_J", "mem_J"]
cols += [c for i in range(len(dev)) for c in ("g%d_uW" % i, "g%d_sclk_Hz" % i, "g%d_busy" % i)]
f = open(out, "w"); f.write(",".join(cols) + "\n")
nxt = time.time()
while True:
    r = ["%.3f" % time.time()]
    for i in range(4):
        e = rd(PM + "accel%d_energy" % i); p = rd(PM + "accel%d_power" % i)
        r += [e[0] if e else "", e[2] if len(e) > 2 else "", p[0] if p else ""]
    e = rd(PM + "energy"); p = rd(PM + "power")
    r += [e[0] if e else "", e[2] if len(e) > 2 else "", p[0] if p else ""]
    for n in ("cpu_energy", "memory_energy"):
        e = rd(PM + n); r.append(e[0] if e else "")
    for i, d in enumerate(dev):
        for p in (pw[i], fq[i], d + "/gpu_busy_percent"):
            v = rd(p) if p else []
            r.append(v[0] if v else "")
    f.write(",".join(r) + "\n"); f.flush()
    nxt += 1.0 / hz
    s = nxt - time.time()
    if s > 0: time.sleep(s)
    else: nxt = time.time()
