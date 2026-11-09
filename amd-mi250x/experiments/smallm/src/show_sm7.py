# Per-draw median ms of each arm, ratio = hb ms / ours ms on the paired hb arm (same A/B memory as stated), per job. argv: sm7.jsonl ...
import json, sys, statistics as st, collections as c
PAIRS = {"16": {"rng": "hb_own", "land": "hb_padb", "st": "hb_st"}, "64": {"land128": "hb_own", "rng96": "hb_own", "pad96": "hb_padb"}}
for f in sys.argv[1:]:
    ms = c.defaultdict(list); rel = []; bey = []
    for l in open(f):
        if not l.startswith("{"): continue
        r = json.loads(l)
        if "impl" in r: ms[r["tag"], r["draw"], r["impl"]].append(r)
        if "rel" in r: rel.append(r); bey.append(r["beyond_M_max"])
    print(f)
    for tag, pp in PAIRS.items():
        draws = sorted({k[1] for k in ms if k[0] == tag})
        for ours, hb in pp.items():
            rs = []; allr = []
            for d in draws:
                o = [x["ms"] for x in ms[tag, d, ours]]; h = [x["ms"] for x in ms[tag, d, hb]]
                rs.append(st.median(h) / st.median(o))
                by_round_h = {x["round"]: x["ms"] for x in ms[tag, d, hb]}
                allr += [by_round_h[x["round"]] / x["ms"] for x in ms[tag, d, ours]]
            tf_o = st.median(x["TF"] for d in draws for x in ms[tag, d, ours]); tf_h = st.median(x["TF"] for d in draws for x in ms[tag, d, hb])
            print(f"  {tag} {ours:8s}/{hb:8s} draw medians " + " ".join(f"{x:.3f}" for x in rs) + f" | median {st.median(rs):.3f} | single reads {min(allr):.3f}-{max(allr):.3f} n={len(allr)} | TF ours {tf_o} hb {tf_h}")
    for r in rel:
        print("  rel", r["tag"], r["draw"], r["rel"], "beyond_M", r["beyond_M_max"])
