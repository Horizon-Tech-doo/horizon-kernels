#!/bin/bash
# N4 round 3: one GCD. Part 1: inside each of 4 allocations of A and B at 16384^3, shift A, B, then both by 0..6 MiB with the data rewritten: does the 128x256 state depend on the address inside the allocation, on A-B, or only on the allocation? Part 2: interleaved processes, plain against contiguous, state per reallocation.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j33_$J; mkdir -p $O; date -u > $O/time.txt
arm(){ python -c "import json;d=json.load(open('h11/dyn.json'))['$1'];print('$1,h11/$1.co,%d,%d,256,%d'%(d['bM'],d['bN'],d['dyn']))"; }
H=$(arm h11_o128_16k); C8=$(arm h11_c8_16k); s=aaaaaaaaabbbbbbbbbccccccccc
./hbr 16384 16384 16384 plain X${s}X${s}X${s}X${s} 6 1 hb $H > $O/shift_d0.jsonl 2>&1
date -u >> $O/time.txt
for p in 0 1 2; do for m in plain contig; do ./hbr 16384 16384 16384 $m XXXXXXXXXX 6 1 hb $H $C8 > $O/mode_${m}_$p.jsonl 2>&1; done; done
date -u >> $O/time.txt
