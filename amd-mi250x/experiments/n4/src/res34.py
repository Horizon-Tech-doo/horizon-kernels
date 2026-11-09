# N4 round 3, job 34: per shape, allocation and arm, TF and ratio over hb against the B shift. Ratio: ours over hb, same step, throughput, higher is better.
import json,sys,glob,statistics as st,collections as cc
O=sys.argv[1]
for f in sorted(glob.glob(f"{O}/sh_*.jsonl")):
    r=[json.loads(l) for l in open(f) if l.startswith("{")]
    off={x["step"]:x["boff"] for x in r if "boff" in x}; hb={x["step"]:x["TF"] for x in r if x.get("impl")=="hb"}
    print("==",f.split("/")[-1],[ (x["check"][4:8],x["fnv"][:8]) for x in r if "check" in x][:2])
    T=cc.defaultdict(lambda:cc.defaultdict(list)); H=cc.defaultdict(list)
    for x in r:
        if "impl" in x and x["q"]!="X":
            H[off[x["step"]]].append(hb[x["step"]])
            if x["impl"]!="hb": T["o128" if "o128" in x["impl"] else "c8"][off[x["step"]]].append(round(x["TF"]/hb[x["step"]],3))
    print("  B shift:",*[f"{d:>18d}" for d in sorted(H)][:0])
    for d in sorted(H):
        print(f"  {d:8d} hb {st.median(H[d]):7.2f}  c8 {' '.join('%.3f'%v for v in T['c8'][d])}   o128 {' '.join('%.3f'%v for v in T['o128'][d])}")
