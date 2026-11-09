#!/bin/bash
# usage (inside container): build.sh <tag> B L '<plan json>' ; writes gen/<tag>.{cu,co,s,tsv}
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; R=/scratch/project_XXXXXXXXX/tilelang
cd $E; tag=$1
TPL=$R/venv/lib/python3.12/site-packages/tilelang/src
python3 gen_n3.py gen/$tag.cu $2 $3 "$(cat gen/$tag.plan)" > gen/$tag.arms || exit 1
$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -c -x hip gen/$tag.cu -o gen/$tag.co 2>&1 | grep -E "error|fatal" | head
$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -S -x hip gen/$tag.cu -o gen/$tag.s 2>&1 | grep -E "error|fatal" | head -5
python3 - gen/$tag <<'PY'
import json,re,sys
t=sys.argv[1]; arms=json.load(open(t+".cu.json"))
with open(t+".tsv","w") as f:
    for a in arms: f.write(f"{a['name']} {a['S']} {a['th']} {a['dyn']} {1 if a['ck'] else 0} {'ld' if 'ld' in a['toks'] else 'full'} {1 if a['swap'] else 0}\n")
s=open(t+".s").read()
for fn in re.split(r"\n(?=\s*\.globl)",s):
    m=re.search(r"\.globl\s+(\S+)",fn)
    if not m or m.group(1).endswith(".kd"): continue
    n=m.group(1)
    if n.startswith("k2_"): continue
    i=re.search(r"; NumVgprs: (\d+)",fn); a=re.search(r"; NumAgprs: (\d+)",fn); o=re.search(r"; Occupancy: (\d+)",fn); sc=re.search(r"; ScratchSize: (\d+)",fn)
    lines=fn.split("\n")
    g=sum(1 for l in lines if re.search(r"\bglobal_load_dwordx4",l)); w=sum(1 for l in lines if re.search(r"s_waitcnt.*vmcnt",l)); v=sum(1 for l in lines if l.strip().startswith("v_")); mf=sum(1 for l in lines if "mfma" in l); bar=sum(1 for l in lines if "s_barrier" in l)
    # loads between consecutive vmcnt waits, in order
    grp=[];c=0
    for l in lines:
        if "global_load_dwordx4" in l: c+=1
        elif "vmcnt" in l and "s_waitcnt" in l:
            if c: grp.append(c)
            c=0
    print(n,"v",i and i.group(1),"a",a and a.group(1),"occ",o and o.group(1),"scr",sc and sc.group(1),"x4",g,"vmwait",w,"valu",v,"mfma",mf,"barriers",bar,"groups",grp)
PY
