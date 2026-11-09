# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# For GPU events matching a pattern, find the CPU op (innermost enclosing cpu_op on the launching thread) via the
# runtime-event correlation id. usage: who_launch.py TRACE.json.gz PATTERN
import collections, gzip, json, re, sys
ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
pat = re.compile(sys.argv[2])
gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and pat.search(e["name"])]
rt = {e["args"]["correlation"]: e for e in ev if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
ops = [e for e in ev if e.get("cat") == "cpu_op"]
by_tid = collections.defaultdict(list)
for o in ops:
    by_tid[o["tid"]].append(o)
for t in by_tid:
    by_tid[t].sort(key=lambda o: o["ts"])
cnt = collections.Counter(); dur = collections.Counter()
for g in gpu:
    r = rt.get(g.get("args", {}).get("correlation"))
    name = "?"
    if r:
        cands = [o for o in by_tid.get(r["tid"], []) if o["ts"] <= r["ts"] <= o["ts"] + o.get("dur", 0)]
        if cands:
            cands.sort(key=lambda o: o.get("dur", 0))
            name = " < ".join(c["name"][:60] for c in cands[:3])
    cnt[name] += 1; dur[name] += g["dur"]
for n, d in dur.most_common(15):
    print(f"{d / 1e3:9.1f} ms {cnt[n]:6d}  {n}")
