"""lbtab.py run.jsonl...: launch-bound arms, ours/SDPA per process (start SDPA and end SDPA), and gain against the same tile's no-bound arm in the same process."""
import json, sys, collections
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.startswith("{")]
    print("==", f)
    ref = collections.defaultdict(dict); t = collections.defaultdict(dict); bad = []
    for r in rows:
        if r.get("impl") in ("sdpa", "sdpa_end"): ref[(r["tag"], r["case"])][r["impl"]] = r["tflops"]
        if r.get("impl") == "tl":
            if "err" in r: bad.append((r["name"], r["err"][-200:])); continue
            t[(r["case"], r["name"].split(r["case"] + "_")[1])][r["tag"]] = r
    for k, v in ref.items(): print("  ref", k, v)
    for (c, n), d in sorted(t.items()):
        base = n.split("_J")[0] + "_JDRF" if "_J" in n else None
        s = []
        for tag in sorted(d):
            r = d[tag]; R = ref[(tag, c)]
            g = ""
            if base and base != n and base in {x[1] for x in t if x[0] == c}:
                g = f" gain {r['tflops'] / t[(c, base)][tag]['tflops'] - 1:+.1%}"
            s.append(f"{tag} {r['tflops']:.2f} /SD {r['tflops'] / R['sdpa']:.3f}-{r['tflops'] / R['sdpa_end']:.3f}{g} err {r.get('rel_vs_fp64')} {r.get('rel_vs_sdpa')} rep {r.get('repeat_equal')}")
        print(c, n, " | ".join(s))
    for b in bad: print("ERR", b)
