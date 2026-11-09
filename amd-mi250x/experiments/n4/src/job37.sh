#!/bin/bash
# N4 round 3: one GCD. Corrects job 35's arms and compiled runs, where the shift list restarted at 0 after each X, so they only read offset 0. Here each X is followed by offset 0 and offset 128 in the same allocation. Then the placed rule through the real script.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j37_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $O/md5.txt
arm(){ python -c "import json;d=json.load(open('h11/dyn.json'))['$1'];print('$1,h11/$1.co,%d,%d,256,%d'%(d['bM'],d['bN'],d['dyn']))"; }
export HBR_DELTAS=0,128
for sp in "16k 16384 6 1" "6k 6144 60 2"; do set -- $sp; sn=$1
  ./hbr $2 $2 $2 plain XbbXbbXbb $3 $4 hb $(for t in o128 p16 a48b16 a96b16 fr15 fr40 c4 c16 r4 r8; do arm h11_${t}_$sn; done) > $O/arms_$sn.jsonl 2>&1
done
C=../C1/s12
./hbr 8192 8192 8192 plain XbbXbbXbb 30 2 hb $(arm h11_o128_8k) rr25632w2_8k,$C/rr25632w2.asis_8k.hsaco,256,128,256,24576 base_kernel,$C/b25632a2.asis_8k.hsaco,256,128,256,24576 > $O/comp_8k.jsonl 2>&1
./hbr 16384 16384 16384 plain XbbXbbXbb 6 1 hb $(arm h11_o128_16k) rr25632w2_16k,$C/rr25632w2.asis_16k.hsaco,256,128,256,24576 base_kernel,$C/b25632a2.asis_16k.hsaco,256,128,256,24576 > $O/comp_16k.jsonl 2>&1
date -u >> $O/time.txt
for sp in "8k 8192 8192 8192 3 10 3" "6k 6144 6144 6144 3 15 3" "17k 4096 17408 5120 3 10 3" "12k 12288 12288 12288 3 4 1" "16k 16384 16384 16384 3 3 1" "4k 4096 4096 4096 3 20 3" "5632 5632 3200 4096 3 15 3" "32k 32768 1024 2048 3 20 3" "4kk 16384 4096 4096 3 10 3"; do set -- $sp; sn=$1
  python ../../scripts/gemm_prefetch.py auto $2 $3 $4 $5 $6 $7 - placed > $O/auto_$sn.jsonl 2> $O/auto_$sn.err
done
date -u >> $O/time.txt
