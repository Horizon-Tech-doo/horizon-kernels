#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j1_$J; mkdir -p $O; S=/scratch/project_XXXXXXXXX/tilelang/exp/E1/solo
date -u > $O/time.txt
bash meta.sh > $O/meta.txt 2>&1
python pollm.py $O/metrics.jsonl 0.02 & PP=$!
arms(){ sn=$1; echo hb b12864_$sn,$S/b12864_$sn.co,128,128,256,32768 b25632w2_$sn,$S/b25632w2_$sn.co,256,128,256,24576 rr12864w3_$sn,$S/rr12864w3_$sn.co,128,128,256,32768 rr25632w2_$sn,$S/rr25632w2_$sn.co,256,128,256,24576; }
# kernel names: direct and torch
for sp in "16k 16384 16384 16384" "8k 8192 8192 8192" "4k 4096 4096 4096"; do set -- $sp
  AMD_LOG_LEVEL=4 ./hbn $2 $3 $4 fixed 1 1 0 0 hb 2>&1 >/dev/null | grep -aE "ShaderName|grid=|group_seg" | grep -aE "Cijk|grid=|group_seg" | sed -E "s/^.*(ShaderName)/\1/; s/^.*(grid=)/\1/" | cut -c1-260 | uniq -c > $O/hblog_direct_$1.txt
done
cat > /tmp/n4_tlog_$J.py <<'P'
import torch,sys
M,N,K=map(int,sys.argv[1:4]); a=torch.randn(M,K,device='cuda',dtype=torch.float16); b=torch.randn(N,K,device='cuda',dtype=torch.float16); torch.backends.cuda.preferred_blas_library('cublaslt')
for _ in range(2): c=a@b.T
torch.cuda.synchronize()
P
AMD_LOG_LEVEL=4 python /tmp/n4_tlog_$J.py 16384 16384 16384 2>&1 >/dev/null | grep -aE "ShaderName|grid=|group_seg" | grep -aE "Cijk|grid=|group_seg" | sed -E "s/^.*(ShaderName)/\1/; s/^.*(grid=)/\1/" | cut -c1-260 | uniq -c > $O/hblog_torch_16k.txt; rm -f /tmp/n4_tlog_$J.py
date -u >> $O/time.txt
# same-job series: per shape, process A randn, process B fixed, process C randn, torch randn twice; probe processes
for sp in "4k 4096 4096 4096 5 40 5" "8k 8192 8192 8192 5 20 5" "32k 32768 1024 2048 5 40 5" "17k 4096 17408 5120 5 20 5" "16k 16384 16384 16384 3 6 2"; do set -- $sp; sn=$1; M=$2;N=$3;K=$4;R=$5;IT=$6;W=$7
  echo "SHAPE $sn $(date -u +%T)" >> $O/time.txt
  ./hbn $M $N $K randn $R $IT $W 0 $(arms $sn) > $O/hb_${sn}_randn1.jsonl 2>&1
  python tgap.py $M $N $K $R $IT $W > $O/torch_${sn}_1.jsonl 2>/dev/null
  ./hbn $M $N $K fixed $R $IT $W 0 $(arms $sn) > $O/hb_${sn}_fixed.jsonl 2>&1
  ./hbn $M $N $K randn $R $IT $W 0 $(arms $sn) > $O/hb_${sn}_randn2.jsonl 2>&1
  python tgap.py $M $N $K $R $IT $W > $O/torch_${sn}_2.jsonl 2>/dev/null
  ./hbn $M $N $K randn 3 $IT $W 5000 $(arms $sn) > $O/hb_${sn}_probe.jsonl 2>&1
done
kill $PP; date -u >> $O/time.txt
