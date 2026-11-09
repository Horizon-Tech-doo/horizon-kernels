# N4 round 3: per process and read: TF, ratio to hb of the same round/realloc, agfx, TF/GHz, UMC. argv: dir glob-of-jsonl key(round|realloc) [window_skip]
import json,sys,gzip,glob,re,statistics as st
O,pat,key=sys.argv[1:4]; skip=float(sys.argv[4]) if len(sys.argv)>4 else 0.5
m={int(l.split()[1]):l.split()[5] for l in open(f"{O}/map.txt") if l.startswith("rocr")}
ms=[]
for l in gzip.open(f"{O}/metrics.jsonl.gz","rt"):
    l=l.strip()
    if l.startswith("{") and l.endswith("}"):
        try: ms.append(json.loads(l))
        except Exception: pass
for f in sorted(glob.glob(f"{O}/{pat}")):
    d=int(re.search(r"_d(\d+)",f).group(1)); rows=[json.loads(x) for x in open(f) if x.startswith("{")]
    a=[x for x in rows if "impl" in x]; hb={x[key]:x["TF"] for x in a if x["impl"]=="hb"}
    print("==",f.split("/")[-1],"card",m[d])
    for x in a:
        s=[q[m[d]] for q in ms if x["t0"]+skip<=q["t"]<=x["t1"] and m[d] in q]
        g=st.median([z["agfx"] for z in s]) if s else 0; u=st.mean([z["umc"] for z in s]) if s else 0; p=max([z["pw"] for z in s]) if s else 0
        print(f"  {x['impl']:24s} {key[0]}{x[key]} TF {x['TF']:7.2f} ratio {x['TF']/hb[x[key]]:.3f} agfx {g:5.0f} TF/GHz {x['TF']/g*1000 if g else 0:6.1f} umc {u:4.1f}")
    for x in rows:
        if "check" in x and "fnv" in x: print("  fnv",x["check"],x["fnv"],x.get("rel"))
