import json,sys,glob,re
for f in sorted(glob.glob(sys.argv[1]+"/s1_d*.jsonl")):
    d=re.search(r"_d(\d)",f).group(1); rows=[json.loads(l) for l in open(f) if l.startswith("{")]
    al={x["realloc"]:x for x in rows if "dummy_mb" in x}
    hb={x["realloc"]:x["TF"] for x in rows if x.get("impl")=="hb"}
    for x in rows:
        if x.get("impl")=="hand":
            r=x["realloc"]; A=int(al[r]["A"],16);B=int(al[r]["B"],16);C=int(x["C"],16); CH=[y["C"] for y in rows if y.get("impl")=="hb" and y["realloc"]==r][0]
            print(f"d{d} r{r} pos{x['pos']} ratio {x['TF']/hb[r]:.3f} A-B {(A-B)/2**20:9.1f}M A-C {(A-C)/2**20:9.1f}M B-C {(B-C)/2**20:9.1f}M Clow {C&0xffffffff:08x} C_hb {CH[-8:]} Alow {A&0xffffffff:08x} Blow {B&0xffffffff:08x}")
