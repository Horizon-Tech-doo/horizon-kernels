#!/bin/bash
# N4 round 5 job C. $1 = fwd or rev. hbr, hb direct, same process, B offsets 0 and 128. Hand-scheduled asm.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/r5c_$J; mkdir -p $O; date -u > $O/time.txt; echo "order=$1" >> $O/time.txt; md5sum gp_r5.py gen_r5c.py hbr > $O/md5.txt
export HBR_DELTAS=0,128; ORDV=$1
args(){ python - "$@" <<'PY'
import json,sys
d=json.load(open("r5/dync.json")); pre,sn=sys.argv[1],sys.argv[2]; order=sys.argv[3]; arms=sys.argv[4:]
if order=="rev": arms=arms[::-1]
o=[]
for a in arms:
    if a=="hb": o.append("hb"); continue
    n=f"{pre}_{a}_{sn}"; v=d[n]; o.append(f"{n},r5/{n}.co,{v['bM']},{v['bN']},256,{v['dyn']}")
print(" ".join(o))
PY
}
for sp in "3k 3072 3072 3072 1000 100" "4k 4096 4096 4096 400 10" "6k 6144 6144 6144 120 3" "8k 8192 8192 8192 50 2" "12k 12288 12288 12288 16 1" "16k 16384 16384 16384 7 1"; do set -- $sp
  ./hbr $2 $3 $4 plain bbbbbb $5 $6 $(args r5c $1 $ORDV hb c8 o128 s0) > $O/s0_$1.jsonl 2>&1; date -u >> $O/time.txt
done
for sp in "8x6 8192 6144 4096 150 3 a16" "6x8 6144 8192 4096 150 3 b16"; do set -- $sp
  ./hbr $2 $3 $4 plain bbbbbb $5 $6 $(args r5c $1 $ORDV hb c8 o128 $7) > $O/nc_$1.jsonl 2>&1; date -u >> $O/time.txt
done
for sp in "6k1k 6144 6144 1024 700 50" "3k256 3072 3072 256 5000 200" "6k256 6144 6144 256 2500 100"; do set -- $sp
  ./hbr $2 $3 $4 plain bbbbbb $5 $6 $(args r5c $1 $ORDV hb c8 o128 a16 b16) > $O/lowa_$1.jsonl 2>&1; date -u >> $O/time.txt
done
for sp in "3k 3072 3072 3072 1000 100" "6k 6144 6144 6144 120 3"; do set -- $sp
  ./hbr $2 $3 $4 plain bbbbbb $5 $6 $(args r5t $1 $ORDV hb a16 a0b32 a16c4 a16c16 a16r8 a16f15 a16f40) > $O/tune_$1.jsonl 2>&1; date -u >> $O/time.txt
done
