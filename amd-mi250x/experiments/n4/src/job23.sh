#!/bin/bash
# N4 round 2, step 3: standing when GCDs are busy. Same card alone, with its twin running the same process, then all 8. Sustained 6 s per arm, gpu_metrics on.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j23_$J; mkdir -p $O; date -u > $O/time.txt; echo "rocr=$ROCR_VISIBLE_DEVICES" >> $O/time.txt
python map23.py > $O/map.txt; read T W < <(tail -1 $O/map.txt); echo "T=$T W=$W" >> $O/time.txt
python pollm.py $O/metrics.jsonl 0.1 & PP=$!
EXTRA="$1"
arm(){ # tag dev sn M N K it warm
  local n=h9_25664c8d1p32_$3 x=""
  [ -f "h10/h10_${EXTRA}_$3.co" ] && x="h10_${EXTRA}_$3,h10/h10_${EXTRA}_$3.co,${EXTRA2:-256,128},256,${EXTRADYN:-61440}"
  ROCR_VISIBLE_DEVICES=$2 ./hbn $4 $5 $6 randn 2 $7 $8 0 hb $n,h9/$n.co,256,128,256,61440 $x > $O/$1_d$2_$3.jsonl 2>&1
}
cond(){ tag=$1; shift
  for sp in "16k 16384 16384 16384 75 2" "8k 8192 8192 8192 600 3" "6k 6144 6144 6144 1400 3"; do set -- $sp "$@"; sn=$1;M=$2;N=$3;K=$4;it=$5;w=$6; shift 6
    pids=(); date +%s.%N >> $O/phase_$tag.txt; echo "$sn" >> $O/phase_$tag.txt
    for d in "$@"; do arm $tag $d $sn $M $N $K $it $w & pids+=($!); done
    wait "${pids[@]}"
  done
}
cond idle $T
cond twin $T $W
cond all 0 1 2 3 4 5 6 7
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
