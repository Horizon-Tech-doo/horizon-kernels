# Medians per shape: hand through the real script against hb through torch, and the C++ harness pair on the same job.
import json,sys,glob,statistics as st,os
O=sys.argv[1]
print("shape  auto_hb auto_hand auto_ratio | cpp_hb cpp_hand cpp_ratio | auto/cpp rel_rule rel_hb")
for sn in "4k 8k 16k 17k 32k 6k 12k 5632 4kk".split():
    a=[json.loads(l) for l in open(f"{O}/auto_{sn}.jsonl") if l.startswith("{")]
    c=[json.loads(l) for l in open(f"{O}/hb_{sn}_randn.jsonl") if l.startswith("{")]
    def m(rows,n): 
        v=[r["TF"] for r in rows if r.get("impl")==n]; return st.median(v) if v else float("nan")
    h=[r for r in a if "rel_err" in r]
    ah,ar=m(a,"hipblaslt"),m(a,"rule"); ch=m(c,"hb"); cn=[r["impl"] for r in c if r.get("impl","hb")!="hb"]; cr=m(c,cn[0]) if cn else float("nan")
    ck=[r for r in c if "check" in r]
    # paired within round
    print(sn,ah,ar,round(ar/ah,3),"|",ch,cr,round(cr/ch,3),"|",round((ar/ah)/(cr/ch),4),h[0].get("rel_err") if h else None,h[0].get("rel_err_hb") if h else None, ck[0]["rel"] if ck else None, ck[0].get("fnv") if ck else None)
