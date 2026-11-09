"""mdtab.py run.jsonl...: ours / vLLM, time, lower is better. Per process (tree, cycle), median of round medians; ours absolute, vLLM absolute."""
import json, sys, collections, statistics as S
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.startswith("{")]
    print("==", f)
    t = collections.defaultdict(list); st = collections.defaultdict(list); err = collections.defaultdict(list)
    for r in rows:
        k = (r.get("wl"), r.get("T", r.get("B")), r.get("route", r.get("L")))
        if r.get("kind") == "time":
            t[k + (r["arm_tree"], r["tag"], r["arm"])].append(r["tot_med"])
            if "stages" in r: st[k + (r["arm_tree"], r["tag"])].append(r["stages"])
        if r.get("kind") == "err":
            err[(r.get("T", r.get("B")), r.get("route", r.get("L")), r["arm"])].append((r["arm_tree"], r["tag"], r.get("rel", r.get("rel_vs_fp64")), r["nonfinite"], r["sha"]))
    keys = sorted({k[:3] for k in t}, key=str)
    for k in keys:
        out = []
        for tree in ("wheel", "fork", "forkx", "forkxP", "forkxO", "forkp3"):
            for tag in ("c1", "c2"):
                o = t.get(k + (tree, tag, "ours")); v = t.get(k + (tree, tag, "vllm")) or t.get(k + (tree, tag, "vllm_tri"))
                if not o: continue
                s = st.get(k + (tree, tag))
                ss = " k1 %.3f k2 %.3f" % (S.median(x[1] for x in s), S.median(x[2] for x in s)) if s else ""
                out.append(f"{tree}/{tag} {S.median(o):.3f}/{S.median(v):.3f}={S.median(o) / S.median(v):.3f}{ss}")
        print(k, " | ".join(out))
    print("errors (arm: worst rel, nonfinite, distinct output hashes per tree)")
    for k, v in sorted(err.items(), key=str):
        rels = [x[2] for x in v if x[2] is not None]
        hs = collections.defaultdict(set)
        for x in v: hs[x[0]].add(x[4])
        print(" ", k, "worst", max(rels) if rels else None, "nonfinite", sum(x[3] for x in v), {a: sorted(b) for a, b in hs.items()})
