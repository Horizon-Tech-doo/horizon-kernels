#!/bin/bash
# N4 round 5 job A. $1 = fwd or rev (arm order). hbr, hipBLASLt direct, same process. Hand-scheduled asm arms from gp_r5.py. B offset 0 and 128 inside one allocation, arm start rotates per step.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/r5a_$J; mkdir -p $O; date -u > $O/time.txt; echo "order=$1" >> $O/time.txt; md5sum gp_r5.py gen_r5.py hbr hbn ../../scripts/gemm_prefetch.py > $O/md5.txt
export HBR_DELTAS=0,128
declare -A D=([c8]=61440 [o128]=61440 [a16]=64512 [a32b0]=65536 [a0]=57344 [b16]=64512)
declare -A BM=([c8]=256 [o128]=128 [a16]=256 [a32b0]=256 [a0]=256 [b16]=192) BN=([c8]=128 [o128]=256 [a16]=192 [a32b0]=192 [a0]=192 [b16]=256)
args(){ local sn=$1; shift; local o=""; for a in "$@"; do if [ $a = hb ]; then o="$o hb"; else n=r5_${a}_$sn; o="$o $n,r5/$n.co,${BM[$a]},${BN[$a]},256,${D[$a]}"; fi; done; echo $o; }
if [ "$1" = rev ]; then T="b16 a0 a32b0 a16 o128 c8 hb"; L="o128 c8 hb"; else T="hb c8 o128 a16 a32b0 a0 b16"; L="hb c8 o128"; fi
ROCR_VISIBLE_DEVICES=0 AMD_LOG_LEVEL=4 ./hbn 3072 3072 3072 randn 1 1 0 0 hb 2>&1 >/dev/null | grep -aE "ShaderName|grid=|group_seg" | grep -aE "Cijk|grid=|group_seg" | sed -E "s/^.*(ShaderName)/\1/; s/^.*(grid=)/\1/; s/kernel_obj.*//" | cut -c1-260 | uniq -c > $O/hblog_3k.txt
for sp in "3k 3072 3072 3072 1000 100" "6k 6144 6144 6144 120 3" "12k 12288 12288 12288 16 1"; do set -- $sp
  ./hbr $2 $3 $4 plain bbbbbb $5 $6 $(args $1 $T) > $O/tile_$1.jsonl 2>&1; date -u >> $O/time.txt
done
for sp in "lk192 4096 4096 192 5000 200" "lk512 4096 4096 512 3000 200" "lk1k 4096 4096 1024 2000 100" "lk2560 2560 2560 256 5000 200"; do set -- $sp
  ./hbr $2 $3 $4 plain bbbbbb $5 $6 $(args $1 $L) > $O/low_$1.jsonl 2>&1; date -u >> $O/time.txt
done
