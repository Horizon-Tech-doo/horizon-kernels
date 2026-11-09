import json,glob,statistics as st,sys
d=sys.argv[1]
SH=["4k","8k","32k","17k","16k"]
print("shape direct_randn_1 direct_randn_2 direct_fixed torch_at torch_out gap_ms(torch_at-direct) direct_fixed/randn ours_rr25632w2_fixed/randn")
def med(f,a):
    v=[json.loads(l)["TF"] for l in open(f) if l.startswith("{") and "\"impl\"" in l and json.loads(l)["impl"]==a] if glob.glob(f) else []
    return st.median(v) if v else None
for s in SH:
    r1=med(f"{d}/hb_{s}_randn1.jsonl","hb");r2=med(f"{d}/hb_{s}_randn2.jsonl","hb");fx=med(f"{d}/hb_{s}_fixed.jsonl","hb")
    t=[med(f"{d}/torch_{s}_{i}.jsonl","torch_at") for i in (1,2)];o=[med(f"{d}/torch_{s}_{i}.jsonl","torch_out") for i in (1,2)]
    t=[x for x in t if x];o=[x for x in o if x]
    M,N,K={"4k":(4096,)*3,"8k":(8192,)*3,"32k":(32768,1024,2048),"17k":(4096,17408,5120),"16k":(16384,)*3}[s]
    fl=2*M*N*K*1e-9
    dd=(r1+r2)/2
    print(s,r1,r2,fx,round(st.mean(t),2),round(st.mean(o),2),round(fl/st.mean(t)-fl/dd,3),round(fx/dd,4), round(med(f"{d}/hb_{s}_fixed.jsonl",f"rr25632w2_{s}")/((med(f"{d}/hb_{s}_randn1.jsonl",f"rr25632w2_{s}")+med(f"{d}/hb_{s}_randn2.jsonl",f"rr25632w2_{s}"))/2),4))
