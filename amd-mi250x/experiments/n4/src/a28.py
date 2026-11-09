import json,sys,glob
for f in sorted(glob.glob(sys.argv[1]+"/auto_*.jsonl")):
    r=[json.loads(l) for l in open(f) if l.startswith("{")]
    a=[x for x in r if "impl" in x]; hb={x["round"]:x["TF"] for x in a if x["impl"]=="hipblaslt"}
    h=r[0] if r else {}
    print(f.split("/")[-1],{k:h.get(k) for k in("rule","hand_tile","rel_err","rel_err_hb")},"ratio",[round(x["TF"]/hb[x["round"]],3) for x in a if x["impl"]=="rule"],"hb",list(hb.values()))
