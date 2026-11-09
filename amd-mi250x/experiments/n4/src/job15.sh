#!/bin/bash
# N7: does giving up CUs raise the clock enough to win? One process per mask, same arms. Plus the idle baseline of the sampler.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j15_$J; mkdir -p $O
python pollm.py $O/metrics.jsonl 0.05 & PP=$!
/bin/sleep 5
for mask in "" "0:0-99" "0:0-87" "0:0-54"; do tag=${mask//[:\-]/_}; [ -z "$tag" ] && tag=all
  for sp in "8k 8192 8192 8192 3 15 3" "16k 16384 16384 16384 3 4 1"; do set -- $sp; sn=$1
    A="hb h7_25664c16d1p32_$sn,h7/h7_25664c16d1p32_$sn.co,$(cat h7/h7_25664c16d1p32_$sn.geom)"
    HSA_CU_MASK=$mask ./hbn $2 $3 $4 randn $5 $6 $7 5000 $A > $O/mask${tag}_$sn.jsonl 2>&1
  done
done
kill $PP; date -u >> $O/time.txt
