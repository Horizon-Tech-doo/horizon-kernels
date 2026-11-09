# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Split TileLang "main_kernel" time per step by launch geometry (norm: 640 threads; skinny: 128 threads, grid.x in
# {96, 48, 24}) and list the Inductor norm kernels, for a trace summarised with k5_trace.py's step definition.
import collections, gzip, json, statistics, sys
ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
ann = sorted([e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute_context_0(0)")], key=lambda e: e["ts"])
name = collections.Counter(e["name"] for e in ann).most_common(1)[0][0]; ann = [e for e in ann if e["name"] == name]
k = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
per = [b["ts"] - a["ts"] for a, b in zip(ann, ann[1:])]; med = statistics.median(per)
steps = [(a, b) for a, b in zip(ann, ann[1:]) if b["ts"] - a["ts"] < 1.5 * med][2:-1]
S = len(steps); acc = collections.Counter(); cnt = collections.Counter()
for a, b in steps:
    for e in k:
        if a["ts"] <= e["ts"] < b["ts"]:
            n = e["name"]; ar = e.get("args", {}); g, bl = ar.get("grid"), ar.get("block")
            if n == "main_kernel":
                if bl and bl[0] == 640: key = "K5b norm (TileLang, 640 thr)"
                elif bl and bl[0] == 128 and g and g[0] in (96, 48, 24) and g[2] == 1: key = "K5a skinny (TileLang)"
                else: key = "regfrag + draft head (TileLang)"
            elif "rms_norm" in n: key = "Inductor rms_norm"
            elif "16x16_128x2" in n and g == [8, 1, 1]: key = "cuBLAS in_proj_ba (M=4)"
            elif "32x32_64x1" in n and g and g[0] == 4 and g[2] == 27: key = "cuBLAS in_proj_ba (M=32)"
            elif "splitKreduce" in n and g == [3, 2, 1]: key = "cuBLAS in_proj_ba split-K reduce"
            else: continue
            acc[key] += e["dur"]; cnt[key] += 1
for key in sorted(acc): print(f"{key:36s} {acc[key]/S:8.1f} us/step  {cnt[key]/S:6.1f} calls/step  {acc[key]/cnt[key]:6.2f} us/call")
