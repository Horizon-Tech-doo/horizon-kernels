import json,statistics as st,collections
jobs=[23504733,23504734]
all_=collections.defaultdict(lambda: collections.defaultdict(list))
bad=[];nchk=0;compiles=set();worst=0
for j in jobs:
    R=[json.loads(l) for l in open(f"raw/{j}/run.jsonl")]
    R=[r for r in R if "case" in r]
    for r in R:
        if "err" in r: bad.append((j,r["name"],r["err"][-100:])); continue
        k=(r["case"],r["input"],r["name"].split("_",2)[-1] if r.get("impl")=="tl" else r["impl"])
        all_[j][k].append(r)
        if r.get("impl")=="tl":
            nchk+=1
            nrep=10 if "16384" in r["case"] else 30
            if r.get("repeat_equal")!=f"{nrep}/{nrep}": bad.append((j,r["name"],r["tag"],r.get("repeat_equal")))
            if r.get("bitwise_same_as_first_form") is False: bad.append((j,r["name"],"bitwise"))
            x=r.get("rel_vs_fp64") or 0; worst=max(worst,x)
            if x>=2e-3: bad.append((j,r["name"],x))
            if r.get("compiles"): compiles.add((j,r["name"],r["tag"]))
    print(j,"rows",len(R),"bitwise True",sum(r.get("bitwise_same_as_first_form") is True for r in R),"False",sum(r.get("bitwise_same_as_first_form") is False for r in R))
print("arms checked",nchk,"worst rel_vs_fp64",worst,"bad",bad,"compiles in job",compiles)
keys=sorted({k for j in jobs for k in all_[j]},key=str)
def med(j,k,f="tflops"):
    return st.median(x[f] for x in all_[j][k])
for k in keys:
    s=[]
    for j in jobs:
        tf=[x["tflops"] for x in all_[j][k]]
        ck=med(j,(k[0],k[1],"ck")); sd=med(j,(k[0],k[1],"sdpa"))
        s.append(f"{min(tf):6.2f}-{max(tf):6.2f} /CK {min(tf)/ck:.3f}-{max(tf)/ck:.3f} /SD {st.median(tf)/sd:.3f}")
    print("%-10s %-6s %-30s %s | %s"%(k[0],k[1],k[2],s[0],s[1]))
print("paired ratio to base arm, same round, [min med max]")
for j in jobs:
    for k in keys:
        if not k[2].endswith(("_n0","_n3","_ctl","land_fullD","land_fullDW","ED_jit","EDW_jit","b80_t640_land_full","b80_t640_E_jit","b32_t640_E_jit")): continue
        b=k[2].rsplit("_",1)[0]+"_jit"
        if k[2].endswith(("land_fullD","land_fullDW","b80_t640_land_full")): b="P160_b32_t640_land_full"
        if k[2] in("P160_b32_t640_ED_jit","P160_b32_t640_EDW_jit","P160_b80_t640_E_jit"): b="P160_b32_t640_E_jit"
        if k[2]=="P160_b32_t640_E_jit": b="P160_b32_t640_land_full"
        rs=[]
        for x in all_[j][k]:
            y=[z for z in all_[j][(k[0],k[1],b)] if z["tag"]==x["tag"]]
            if y: rs.append(x["tflops"]/y[0]["tflops"])
        print(j,k[0],k[1],k[2],"vs",b,"%.4f %.4f %.4f"%(min(rs),st.median(rs),max(rs)))
