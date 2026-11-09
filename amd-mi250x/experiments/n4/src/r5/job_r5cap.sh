#!/bin/bash
# N4 round 5 cap test. 8 GCDs, one hbr process per GCD, hb direct same process, hand-scheduled asm. B offsets 0 and 128. About 1 s per arm.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/r5cap_$J; mkdir -p $O; date -u > $O/time.txt; md5sum gp_r5.py gen_r5.py hbr > $O/md5.txt
python pollm.py $O/metrics.jsonl 0.1 & PP=$!
export HBR_DELTAS=0,128
for sp in "6k 6144 6144 6144 233 2" "3k 3072 3072 3072 1800 100" "12k 12288 12288 12288 30 1"; do set -- $sp; sn=$1
  a="hb r5_c8_$sn,r5/r5_c8_$sn.co,256,128,256,61440 r5_o128_$sn,r5/r5_o128_$sn.co,128,256,256,61440 r5_a16_$sn,r5/r5_a16_$sn.co,256,192,256,64512 r5_b16_$sn,r5/r5_b16_$sn.co,192,256,256,64512"
  pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d ./hbr $2 $3 $4 plain bbbb $5 $6 $a > $O/cap_${sn}_d$d.jsonl 2>&1 & pids+=($!); done
  wait "${pids[@]}"; date -u >> $O/time.txt
done
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl
