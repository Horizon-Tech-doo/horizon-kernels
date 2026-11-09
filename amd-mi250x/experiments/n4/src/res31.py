# N4 round 3, step 3: per shape, ratio ours over hb per process (2 rounds), medians, UMC, agfx, TF/GHz. argv: dir
import json,sys,gzip,glob,re,statistics as st,collections as cc
O=sys.argv[1]
m={int(l.split()[1]):l.split()[5] for l in open(f"{O}/map.txt") if l.startswith("rocr")}
ms=[]
for l in gzip.open(f"{O}/metrics.jsonl.gz","rt"):
    l=l.strip()
    if l.startswith("{") and l.endswith("}"):
        try: ms.append(json.loads(l))
        except Exception: pass
ms.sort(key=lambda q:q["t"])
for sn in "4k 12k 17k 32k 5632 4kk 16k 6k".split():
    print("==",sn); R=cc.defaultdict(list); F=cc.defaultdict(set)
    for f in sorted(glob.glob(f"{O}/all_d*_{sn}.jsonl")):
        d=int(re.search(r"_d(\d+)_",f).group(1)); rows=[json.loads(x) for x in open(f) if x.startswith("{")]
        a=[x for x in rows if "impl" in x]; hb={x["round"]:x["TF"] for x in a if x["impl"]=="hb"}
        line=[]
        for x in a:
            s=[q[m[d]] for q in ms if x["t0"]+0.5<=q["t"]<=x["t1"] and m[d] in q]
            g=st.median([z["agfx"] for z in s]) if s else 0; u=st.mean([z["umc"] for z in s]) if s else 0; pw=max([z["pw"] for z in s]+[0])
            n="c8" if "_c8_" in x["impl"] else "o128" if "o128" in x["impl"] else "hb"
            R[n].append((x["TF"]/hb[x["round"]],x["TF"],x["TF"]/g*1000 if g else 0,u,g,pw))
            if n!="hb": line.append(f"{n} r{x['round']} {x['TF']/hb[x['round']]:.3f} {x['TF']/g*1000 if g else 0:.1f}")
        print(f"  d{d} hb {hb[0]:.1f}/{hb[1]:.1f} {' '.join(line)}")
        for x in rows:
            if "check" in x: F["c8" if "_c8_" in x["check"] else "o128"].add(x["fnv"])
    for k,v in R.items():
        print(f" {k:5s} ratio med {st.median(z[0] for z in v):.3f} [{min(z[0] for z in v):.3f}-{max(z[0] for z in v):.3f}] n{len(v)} TF {st.median(z[1] for z in v):.2f} TF/GHz {st.median(z[2] for z in v):.1f} umc {st.mean(z[3] for z in v):.1f} agfx {st.median(z[4] for z in v):.0f} maxpw {max(z[5] for z in v)}")
    print(" fnv",{k:sorted(v) for k,v in F.items()})
