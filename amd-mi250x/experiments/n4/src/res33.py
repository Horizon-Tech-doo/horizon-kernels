# N4 round 3, job 33. Ratio ours over hb, per step. argv: dir
import json,sys,glob,statistics as st
O=sys.argv[1]
r=[json.loads(l) for l in open(f"{O}/shift_d0.jsonl") if l.startswith("{")]
hb={x["step"]:x["TF"] for x in r if x.get("impl")=="hb"}
off={x["step"]:(x["aoff"],x["boff"]) for x in r if "aoff" in x}
line=[];
for x in r:
    if "o128" in x.get("impl",""):
        if x["q"]=="X": print("\n".join(line)); line=[]; print("X",round(x["TF"]/hb[x["step"]],3),"hb",hb[x["step"]]); cur=None
        else:
            if x["q"]!=cur: line.append(x["q"]+": "); cur=x["q"]
            line[-1]+=f"{x['TF']/hb[x['step']]:.3f} "
print("\n".join(line)); print([x for x in r if "check" in x][:2])
for m in ("plain","contig"):
    for f in sorted(glob.glob(f"{O}/mode_{m}_*.jsonl")):
        r=[json.loads(l) for l in open(f) if l.startswith("{")]; hb={x["step"]:x["TF"] for x in r if x.get("impl")=="hb"}
        print(f.split("/")[-1],"o128"," ".join(f"{x['TF']/hb[x['step']]:.3f}" for x in r if "o128" in x.get("impl","")),"| c8"," ".join(f"{x['TF']/hb[x['step']]:.3f}" for x in r if "_c8_" in x.get("impl","")),"| hb med %.2f"%st.median(hb.values()))
