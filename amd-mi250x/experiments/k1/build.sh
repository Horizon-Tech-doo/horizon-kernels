#!/bin/bash
# Run inside the container (tl_run.sh). Builds one code object per GEMM kernel, one for the micro tests, and the host.
cd /scratch/project_XXXXXXXXX/tilelang/exp/K1 || exit 1
R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
HC="$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -I."
mkdir -p co s cu
python gen_ref.py || exit 1
one(){ # name extra flags
  n=$1; shift
  $HC "$@" -DKNAME=$n -c -x hip k1_gemm.cu -o co/$n.co 2>&1 | grep -E "error|illegal" | head -5
  $HC "$@" -DKNAME=$n -S -x hip k1_gemm.cu -o s/$n.s 2>&1 | grep -E "error" | head -2
}
export -f one
for V in 0 1 2 3 4 5 6; do for CH in 32 64; do
  [ $V = 6 ] && [ $CH = 64 ] && continue
  one g${V}_${CH} -DK1V=$V -DK1CH=$CH -DK1MATH=1 -DK1LOAD=1 &
  one g${V}_${CH}_l -DK1V=$V -DK1CH=$CH -DK1MATH=0 -DK1LOAD=1 &
  one g${V}_${CH}_m -DK1V=$V -DK1CH=$CH -DK1MATH=1 -DK1LOAD=0 &
  wait
done; done
for spec in "g1_32w3 1 32" "g4_32w3 4 32" "g0_64w3 0 64" "g1_64w3 1 64" "g3_64w3 3 64" "g5_64w3 5 64" "g4_64w3 4 64"; do set -- $spec; one $1 -DK1V=$2 -DK1CH=$3 -DK1MATH=1 -DK1LOAD=1 -DWPE=3 & done; wait
for f in cu/ref_*.cu; do n=$(basename $f .cu); ( $HC -c -x hip $f -o co/$n.co 2>&1 | grep -E "error" | head -3; $HC -S -x hip $f -o s/$n.s 2>&1 | grep -E " error" | head -2 ) & done; wait
$HC -c -x hip k1_micro.cu -o micro.co 2>&1 | grep -E "error|illegal" | head -20
$HC -S -x hip k1_micro.cu -o micro.s 2>&1 | grep -E " error" | head -5
hipcc -O2 -std=c++17 main.cpp -o k1 2>&1 | grep -E "error" | head
ls -la micro.co k1; ls co | wc -l
for f in s/*.s; do awk -v n=$(basename $f .s) '/; (NumVgprs|NumAgprs|TotalNumVgprs|ScratchSize|Occupancy):/{v=v" "$2$3} END{print n,v}' $f; done
