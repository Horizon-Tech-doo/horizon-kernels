import json, sys, collections
for f in sys.argv[1:]:
    print("==", f)
    rows = [json.loads(l) for l in open(f) if l.startswith("{")]
    tab = collections.defaultdict(lambda: collections.defaultdict(list))
    errs = collections.defaultdict(dict)
    eq = {}
    for r in rows:
        if "ratio" in r:
            for n, v in r["ratio"].items(): tab[tuple(r["shape"]) + tuple(r["tile"])][n].append(v)
            tab[tuple(r["shape"]) + tuple(r["tile"])]["hbTF"].append(r["tf"]["hb"])
        if "rel_err" in r:
            errs[tuple(r["shape"])][r["arm"]] = (round(r["rel_err"], 7), r["pass_fired"])
        if "check" in r:
            eq[tuple(r["shape"])] = r["equal"]
    for k, v in tab.items():
        print(k, " ".join(f"{n} {sorted(x)[len(x)//2]:.3f} [{min(x):.3f},{max(x):.3f}]" for n, x in v.items() if n != "hbTF"), "hbTF %.1f" % (sorted(v["hbTF"])[len(v["hbTF"])//2]), "bits==hand", eq.get(k[:3]), errs[k[:3]])
