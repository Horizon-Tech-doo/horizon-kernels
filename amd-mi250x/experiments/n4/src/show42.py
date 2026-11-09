import json,sys,glob,statistics as S,os
d=sys.argv[1]
for f in sorted(glob.glob(d+"/*.jsonl")):
    rs=[json.loads(l) for l in open(f) if l.startswith("{")]
    if not rs: print(os.path.basename(f),"EMPTY"); continue
    h=rs[0]; hb=[r["TF"] for r in rs[1:] if r.get("impl")=="hipblaslt"]; o=[r["TF"] for r in rs[1:] if r.get("impl")=="rule"]
    rt=[a/b for a,b in zip(o,hb)]
    print(os.path.basename(f),h.get("rule"),h.get("hand_tile"),"off",h.get("a_minus_b_mod256"),"err",h.get("rel_err"),h.get("rel_err_hb"),"hb",round(S.median(hb),2) if hb else None,"ours",round(S.median(o),2) if o else None,"ratio",round(S.median(o)/S.median(hb),3) if hb and o else None,[round(x,3) for x in rt])
