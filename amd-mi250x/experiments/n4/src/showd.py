import json,sys,glob,statistics as S,os,collections
d=sys.argv[1]
for f in sorted(glob.glob(d+"/*.jsonl")):
    rs=[json.loads(l) for l in open(f) if l.startswith("{")]
    print("==",os.path.basename(f),"rows",len(rs))
    for r in rs:
        if r.get("kind")=="check" and (r.get("bit_same_as_first") is False or r.get("nonfinite") or "rel_tl_vs_ck" in r): print(" ",r)
    if not rs: continue
    g=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rs:
        if r.get("kind")!="time": continue
        key=r.get("case") or r.get("route")
        g[key][r["impl"],r["arm"]].append(r.get("tflops",r.get("med")))
    for key,a in g.items():
        print(" ",key)
        for (impl,arm),v in sorted(a.items()):
            print("   ",impl,arm,"med",round(S.median(v),3),"range",min(v),max(v))
        base={impl:S.median(a[(impl,arm)]) for (impl,arm) in a if arm in("k0v0","x0w0")}
        for (impl,arm),v in sorted(a.items()):
            print("    rel",impl,arm,round(S.median(v)/base[impl],4))
        other="ck" if any(i=="ck" for i,_ in a) else ("vllm" if any(i=="vllm" for i,_ in a) else None)
        if other:
            ours="tl" if other=="ck" else "ours"
            for (impl,arm),v in sorted(a.items()):
                if impl==ours and (other,arm) in a: print("    ours/vendor same-arm",arm,round(S.median(v)/S.median(a[(other,arm)]),4))
                if impl==ours: print("    ours/vendor-base",arm,round(S.median(v)/base[other],4))
