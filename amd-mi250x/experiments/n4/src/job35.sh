#!/bin/bash
# N4 round 3: one GCD. Refine the A-B offset around 128 B; do the step-2 schedule arms and the compiled C1 kernels respond? Arms listed per run; order rotates by step. HBR_DELTAS lists B shifts.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j35_$J; mkdir -p $O; date -u > $O/time.txt
arm(){ python -c "import json;d=json.load(open('h11/dyn.json'))['$1'];print('$1,h11/$1.co,%d,%d,256,%d'%(d['bM'],d['bN'],d['dyn']))"; }
export HBR_DELTAS=0,16,32,48,96,112,128,144,160,192,224,240,256,384,640,896,1152,4224,8320,16512,65664,262272,1048704
s=bbbbbbbbbbbbbbbbbbbbbbb
for sp in "16k 16384 6 1" "6k 6144 60 2" "8k 8192 30 2"; do set -- $sp; sn=$1
  ./hbr $2 $2 $2 plain X${s}X${s} $3 $4 hb $(arm h11_c8_$sn) $(arm h11_o128_$sn) > $O/fine_$sn.jsonl 2>&1
done
date -u >> $O/time.txt
export HBR_DELTAS=0,128
for sp in "16k 16384 6 1" "6k 6144 60 2"; do set -- $sp; sn=$1
  ./hbr $2 $2 $2 plain XbXbXb $3 $4 hb $(for t in o128 p16 a48b16 a96b16 fr15 fr40 c4 c16 r4 r8; do arm h11_${t}_$sn; done) > $O/arms_$sn.jsonl 2>&1
done
C=../C1/s12
./hbr 8192 8192 8192 plain XbXbXb 30 2 hb $(arm h11_o128_8k) rr25632w2_8k,$C/rr25632w2.asis_8k.hsaco,256,128,256,24576 base_kernel,$C/b25632a2.asis_8k.hsaco,256,128,256,24576 > $O/comp_8k.jsonl 2>&1
./hbr 16384 16384 16384 plain XbXbXb 6 1 hb $(arm h11_o128_16k) rr25632w2_16k,$C/rr25632w2.asis_16k.hsaco,256,128,256,24576 base_kernel,$C/b25632a2.asis_16k.hsaco,256,128,256,24576 > $O/comp_16k.jsonl 2>&1
date -u >> $O/time.txt
