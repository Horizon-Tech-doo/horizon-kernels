#!/bin/bash
# Compiles every h7 variant listed for every shape.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
HC="$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -I."
rm -f h7/*
for sp in "8k 8192 8192 8192" "4k 4096 4096 4096" "16k 16384 16384 16384" "17k 4096 17408 5120" "32k 32768 1024 2048"; do set -- $sp; sn=$1;M=$2;N=$3;K=$4
  while read -r tag BM BN CH dyn opts; do [ -z "$tag" ] && continue; n=${tag}_$sn
    case $tag in h8*) G=gen_h8.py;; *) G=gen_h7.py;; esac; python $G $n $M $N $K $BM $BN $CH h7/$n.cu $opts || continue
    echo "$BM,$BN,256,$dyn" > h7/$n.geom
    ( $HC -c -x hip h7/$n.cu -o h7/$n.co 2>&1 | grep -E " error|invalid|illegal" | head -3 | cut -c1-220; $HC -S -x hip h7/$n.cu -o h7/$n.s 2>&1 | grep -E " error" | head -1 | cut -c1-200 ) &
    while [ $(jobs -r | wc -l) -ge 8 ]; do sleep 0.2; done
  done < h7_list.txt
done; wait
python res.py h7/*_8k.s
ls h7/*.co | wc -l
