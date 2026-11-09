# N4 round 3, cap with offset 0 and 128. Per shape: ratio ours over hb, per process per read. argv: dir
import json,sys,glob,re,statistics as st,collections as cc
O=sys.argv[1]
for sn in "16k 6k 8k 12k 4k 17k 32k 4kk".split():
    R=cc.defaultdict(list); fnv=cc.defaultdict(set); P={}; hbv=cc.defaultdict(list)
    for f in sorted(glob.glob(f"{O}/cap_{sn}_d*.jsonl")):
        d=re.search(r"_d(\d)",f).group(1); r=[json.loads(l) for l in open(f) if l.startswith("{")]
        off={x["step"]:x["boff"] for x in r if "boff" in x}; hb={x["step"]:x["TF"] for x in r if x.get("impl")=="hb"}
        for x in r:
            if "check" in x: fnv[x["check"][4:8]].add(x["fnv"][:8])
            if x.get("impl")=="hb": hbv[off[x["step"]]].append(x["TF"])
            elif "impl" in x:
                n="c8" if "_c8_" in x["impl"] else "o128"; R[(n,off[x["step"]])].append(x["TF"]/hb[x["step"]]); P.setdefault((n,off[x["step"]],d),[]).append(round(x["TF"]/hb[x["step"]],3))
    print(sn," ".join(f"hb@{k} {st.median(v):.2f}" for k,v in sorted(hbv.items())),"fnv",dict(fnv))
    for k,v in sorted(R.items()): print(f"   {k[0]:5s} off {k[1]:4d} med {st.median(v):.3f} [{min(v):.3f}-{max(v):.3f}] n{len(v)} below1 {sum(x<1 for x in v)}")
    print("   o128@128 per card:"," ".join(f"d{d}:"+"/".join(str(x) for x in P.get(("o128",128,str(d)),[])) for d in range(8)))
