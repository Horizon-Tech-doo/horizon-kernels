#!/bin/bash
# N4 round 3, step 2: untuned 128x256 race at the cap. 8 GCDs, 3 s per arm, 2 rotated rounds. Arm order is the listed order, rotated one position per round by hbn.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/${JN:-j30}_$J; mkdir -p $O; date -u > $O/time.txt
python map23.py > $O/map.txt; python pollm.py $O/metrics.jsonl 0.1 & PP=$!
arm(){ python -c "import json;d=json.load(open('h11/dyn.json'))['$1'];print('$1,h11/$1.co,%d,%d,256,%d'%(d['bM'],d['bN'],d['dyn']))"; }
for sp in "16k 16384 16384 16384 37 2" "6k 6144 6144 6144 700 3"; do set -- $sp; sn=$1;M=$2;N=$3;K=$4;it=$5;w=$6
  a="hb"; for t in ${ARMS:-o128 p16 a48b16 a96b16 fr15 fr40 c4 c16 r4}; do a="$a $(arm h11_${t}_$sn)"; done
  pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d ./hbn $M $N $K randn 2 $it $w 0 $a > $O/all_d${d}_$sn.jsonl 2>&1 & pids+=($!); done
  wait "${pids[@]}"
done
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
