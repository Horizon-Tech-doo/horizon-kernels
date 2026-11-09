#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4; mkdir -p h9
R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
HC="$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -I."
for sp in "4k 4096 4096 4096" "8k 8192 8192 8192" "16k 16384 16384 16384" "17k 4096 17408 5120" "32k 32768 1024 2048" "6k 6144 6144 6144" "12k 12288 12288 12288" "5632 5632 3200 4096" "4kk 16384 4096 4096"; do set -- $sp
  n=h9_25664c8d1p32_$1; python gen_h7.py $n $2 $3 $4 256 128 64 h9/$n.cu sw=c8 rdiv=1 pad=32; $HC -c -x hip h9/$n.cu -o h9/$n.co 2>&1 | grep " error" | head -2 & 
done; wait; ls h9/*.co | wc -l
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
CUDA_VISIBLE_DEVICES= python - <<P
import gemm_prefetch as g
for s in [(4096,4096,4096),(8192,8192,8192),(16384,16384,16384),(4096,17408,5120),(32768,1024,2048),(6144,6144,6144),(12288,12288,12288),(5632,3200,4096),(16384,4096,4096)]:
    print(s,g.select_tile(*s)); g.gemm_auto(*s)
P
