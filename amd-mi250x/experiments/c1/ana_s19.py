# C1 s19: per process TF per arm, by mode. Ratio convention: ours over direct hipBLASLt, throughput, higher is better, same round. argv: jsonl files.
import json,sys,statistics as st,collections as cc
for f in sys.argv[1:]:
    proc=None; R=cc.defaultdict(lambda:cc.defaultdict(list)); cs=cc.defaultdict(set); P=[]; addr=[]
    for l in open(f):
        if not l.startswith("{"):
            if "HIP" in l or "rc" in l: print("!",l.strip()[:150])
            continue
        x=json.loads(l)
        if "process" in x: proc=x["process"]; P.append(proc); continue
        if "A" in x and "shape" in x: addr.append((proc,x["shape"],x["A"],x["B"],x["C"])); continue
        if x.get("check")=="cs": cs[x["label"].split("_")[0]+"@"+x["shape"]].add(x["cs"]); continue
        if "label" in x and "TF" in x: R[proc][(x["shape"],x["label"])].append(x["TF"])
    print("==",f)
    tab=cc.defaultdict(lambda:cc.defaultdict(list))
    for p,d in R.items():
        m=p.replace("big.","").split(".")[0]
        for (sh,lab),v in d.items(): tab[(sh,lab)][m].append(st.median(v))
    for (sh,lab),t in sorted(tab.items()):
        print(f"{sh:18s} {lab:24s} "+" | ".join(f"{m} n{len(v)} med {st.median(v):7.2f} min {min(v):7.2f} max {max(v):7.2f}" for m,v in sorted(t.items())))
    print("checksums per arm, shape:",{k:len(v) for k,v in cs.items() if len(v)>1})
    print("processes",len(P),"addr examples",addr[:3])
