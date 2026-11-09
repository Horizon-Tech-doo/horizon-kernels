"""Summarize E3 jsonl. Usage: python3 summ.py <dir-with-out-jobids...>; prints name n median_ms min..max TF_median TF_range err."""
import glob, json, statistics as st, sys
rows = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + "/*.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if "name" in r:
                rows.setdefault(r["name"], []).append(r)
for n, rs in sorted(rows.items()):
    ok = [r for r in rs if "ms" in r]
    if not ok:
        print(f"{n:38s} ERR {rs[0].get('err','')[:80]}")
        continue
    ms = [r["ms"] for r in ok]
    tf = [r["tflops"] for r in ok]
    e = [r["rel_vs_fp64"] for r in rs if "rel_vs_fp64" in r]
    print(f"{n:38s} n={len(ok)} ms={st.median(ms):8.3f} [{min(ms):8.3f},{max(ms):8.3f}] TF={st.median(tf):7.2f} [{min(tf):6.2f},{max(tf):6.2f}] err={e[0] if e else ''}")
