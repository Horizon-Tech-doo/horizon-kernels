"""Tables from raw/ep-*.jsonl. Per job: per (world, T, arm) medians over rounds. Efficiency = tok/s at W / (W * tok/s at W=1), same arm, same job."""
import json, sys, statistics as st, collections as C
rows, checks = [], []
for f in sys.argv[1:]:
    job = f.split("-")[-1].split(".")[0]
    for l in open(f):
        if not l.startswith("{"): continue
        o = json.loads(l); o["job"] = job
        if "arm" in o and "round" in o: rows.append(o)
        if "check" in o: checks.append(o)
g = C.defaultdict(list)
for o in rows: g[o["job"], o["tokens_per_rank"], o["arm"], o["world"]].append(o)
for job in sorted({k[0] for k in g}):
    for T in sorted({k[1] for k in g if k[0] == job}, reverse=True):
        print(f"\n## job {job} T={T}\n| arm | W | disp | comp | comb | step | share | M tok/s | eff |\n|---|---|---|---|---|---|---|---|---|")
        for arm in ("old", "land", "skip"):
            base = None
            for W in (1, 8, 16, 32):
                r = g.get((job, T, arm, W))
                if not r: continue
                m = lambda k: st.median(x[k] for x in r)
                d, c, b, s, tp = m("dispatch_ms"), m("compute_ms"), m("combine_ms"), m("step_ms"), m("tokens_per_s_total")
                base = base or tp
                print(f"| {arm} | {W} | {d:.2f} | {c:.2f} | {b:.2f} | {s:.2f} | {(d+b)/(d+c+b):.0%} | {tp/1e6:.3f} | {tp/(W*base):.0%} |" + (f" range {min(x['tokens_per_s_total'] for x in r)/1e6:.3f}-{max(x['tokens_per_s_total'] for x in r)/1e6:.3f} drop {max(x['dropped_max'] for x in r)}"))
print("\nchecks worst:", max(c["rel_y_vs_fp32"] for c in checks), max(c["rel_out_vs_fp32"] for c in checks), "round trip", sum(c["round_trip_diff"] for c in checks), "nonfinite", sum(c["nonfinite"] for c in checks), "n", len(checks))
pos = C.defaultdict(list)
for o in rows: pos[o["job"], o["world"], o["tokens_per_rank"], o["arm"], o["pos"]].append(o["step_ms"])
