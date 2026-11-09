import json,sys,glob,os,statistics as S,re,collections
d=sys.argv[1]
print(open(d+"/time.txt").read().split("\n")[2] if os.path.exists(d+"/time.txt") else "")
for m in ("placed","unplaced"):
    f=f"{d}/best_{m}.out"
    g=collections.defaultdict(lambda: collections.defaultdict(list)); err={}
    for l in open(f):
        if not l.startswith("{"): continue
        r=json.loads(l)
        if "arm" in r:
            g[r["M"],r["N"],r["K"]][r["arm"]].append(r["TF"]); err[r["M"],r["N"],r["K"]]=r.get("fp32_rel")
    print("best",m)
    for k,a in g.items():
        h,o=a["hipblaslt_torch"],a["best"]
        print(" ",k,"hb",h,"best",o,"ratio",[round(x/y,3) for x,y in zip(o,h)],"rel",round(err[k],6))
for sn in "4k 8k 16k 6k 12k 17k 32k 4kk 5632".split():
    row=[sn]
    for m in ("placed","unplaced"):
        rs=[json.loads(l) for l in open(f"{d}/tile_{sn}_{m}.jsonl") if l.startswith("{")]
        if not rs: row.append((m,"EMPTY")); continue
        h=[r["TF"] for r in rs[1:] if r.get("impl")=="hipblaslt"]; o=[r["TF"] for r in rs[1:] if r.get("impl")=="rule"]
        row.append((m,rs[0].get("rule"),"off",rs[0].get("a_minus_b_mod256"),"err",rs[0].get("rel_err"),rs[0].get("rel_err_hb"),"hb",round(S.median(h),2),"tile",round(S.median(o),2),"r",[round(a/b,3) for a,b in zip(o,h)]))
    print(*row)
