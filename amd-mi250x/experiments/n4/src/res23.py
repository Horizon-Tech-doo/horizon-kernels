# Step 3 table. For each condition, shape, device, kernel arm: TF, ratio hand/hb by round, delivered clock, target clock, package power of the OAM, twin activity, throttle bits, from gpu_metrics in the arm's window.
import json,sys,gzip,glob,re,os,statistics as st
O=sys.argv[1]
rows=[l.split() for l in open(f"{O}/map.txt") if l.startswith("rocr")]
m={int(r[1]):(r[5],int(r[7])) for r in rows}   # rocr -> (card, numa)
print("map rocr->card,numa",m)
ms=[json.loads(l) for l in gzip.open(f"{O}/metrics.jsonl.gz","rt") if l.strip().startswith("{") and l.strip().endswith("}")]
def twin(card):
    c=[v[0] for v in m.values() if v[1]==[x for x in m.values() if x[0]==card][0][1] and v[0]!=card]; return c[0] if c else None
def win(card,t0,t1):
    tw=twin(card); s=[x for x in ms if t0+0.5<=x["t"]<=t1 and card in x]
    if not s: return {}
    q=lambda k,c=card: round(st.median([x[c][k] for x in s if c in x and k in x[c]]),0) if any(c in x and k in x[c] for x in s) else None
    pw=[x[c].get("pw",0) for x in s for c in (card,tw) if c and c in x]; 
    pwm=[max(x[card].get("pw",0),x[tw].get("pw",0) if tw and tw in x else 0) for x in s]
    return dict(agfx=q("agfx"),cgfx=q("cgfx"),act=q("act"),umc=q("umc"),twin_act=q("act",tw) if tw else None,pkgW=round(st.median(pwm)),pkgWmax=max(pwm),thr=max(x[card].get("thr",0) for x in s),ithr=max(x[card].get("ithr",0) for x in s),n=len(s))
for tag in ("idle","twin","all"):
  for sn in ("16k","8k","6k"):
    print("==",tag,sn)
    rat=[]
    for f in sorted(glob.glob(f"{O}/{tag}_d*_{sn}.jsonl")):
        d=int(re.search(r"_d(\d+)_",f).group(1)); r=[json.loads(l) for l in open(f) if l.startswith("{")]
        a=[x for x in r if "impl" in x]; 
        if not a: print("d",d,"NO DATA",open(f).read()[:200]); continue
        by={}
        for x in a: by.setdefault(x["impl"],{})[x["round"]]=x
        hk=[k for k in by if k!="hb"][0]
        for rn in sorted(by["hb"]):
            hb,hd=by["hb"][rn],by[hk].get(rn)
            if not hd: continue
            R=hd["TF"]/hb["TF"]; rat.append(R)
            print(f"d{d} {m[d][0]} r{rn} hb {hb['TF']:.2f} hand {hd['TF']:.2f} ratio {R:.3f} | hb {win(m[d][0],hb['t0'],hb['t1'])} | hand {win(m[d][0],hd['t0'],hd['t1'])}")
    if rat: print("ratio range",round(min(rat),3),round(st.median(rat),3),round(max(rat),3),"n",len(rat))
