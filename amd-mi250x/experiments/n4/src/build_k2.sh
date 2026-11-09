#!/bin/bash
# Inside the container. Hand and TileLang-generated K2 kernels, one TU each.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4 || exit 1
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
HC="$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -I."
cp ../K1/k1_common.h .
mkdir -p solo hand
python gen_k2.py > gen_k2.out 2>&1
one(){ n=$1; shift; $HC "$@" -DKNAME=$n -c -x hip k2_gemm.cu -o hand/$n.co 2>&1 | grep -E "error" | head -3; $HC "$@" -DKNAME=$n -S -x hip k2_gemm.cu -o hand/$n.s 2>&1 | grep -E " error" | head -1; }
for sp in "4k 4096 4096" "8k 8192 8192" "32k 2048 1024" "17k 5120 17408" "16k 16384 16384"; do set -- $sp; sn=$1; K=$2; N=$3
  c="-DKK=$K -DNN=$N"
  for spec in "h1_25664 1 64 256 128" "h1_25632 1 32 256 128" "h1_12825664 1 64 128 256" "h5_25664 5 64 256 128" "h5_25632 5 32 256 128" "h3_25664 3 64 256 128" "h4_25632 4 32 256 128" "h5_12825664 5 64 128 256" "h0_25664 0 64 256 128"; do set -- $spec
    one ${1}_$sn $c -DK2V=$2 -DK2CH=$3 -DBM=$4 -DBN=$5 &
    one ${1}w2_$sn $c -DK2V=$2 -DK2CH=$3 -DBM=$4 -DBN=$5 -DWPE=2 &
    while [ $(jobs -r | wc -l) -ge 8 ]; do sleep 0.2; done
  done
  # stagger, hipBLASLt's SU4 SUS256: 4 starts, 128 K apart
  one h1_25664s4_$sn $c -DK2V=1 -DK2CH=64 -DBM=256 -DBN=128 -DSTG=4 -DSTGS=2 &
  one h1_25664s4w2_$sn $c -DK2V=1 -DK2CH=64 -DBM=256 -DBN=128 -DSTG=4 -DSTGS=2 -DWPE=2 &
  one h1_25632s4w2_$sn $c -DK2V=1 -DK2CH=32 -DBM=256 -DBN=128 -DSTG=4 -DSTGS=4 -DWPE=2 &
  wait
done
for f in solo/*.cu; do b=${f%.cu}; ( $HC -c -x hip $f -o $b.co 2>&1 | grep -E "error"|head -2 ; $HC -S -x hip $f -o $b.s 2>&1 | grep -E " error"|head -1 ) & while [ $(jobs -r | wc -l) -ge 8 ]; do sleep 0.2; done; done; wait
cat ../E1/res.py > res.py
python res.py solo/*_8k.s hand/*_8k.s > res_8k.txt
ls hand/*.co | wc -l; ls solo/*.co | wc -l
