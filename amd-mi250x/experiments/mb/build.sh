#!/bin/bash
R=/scratch/project_XXXXXXXXX/tilelang
cd $R/analysis/mb
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
python gen.py $1 || exit 1
$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -c -x hip kernels.cu -o kernels.co 2>&1 | grep -E "error|warning: unk|ignored" | head
$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -S -x hip kernels.cu -o kernels.s 2>&1 | grep -E "error" | head -5
ls -la kernels.co kernels.s
hipcc -O2 -std=c++17 main.cpp -o mb 2>&1 | grep -E "error" | head
ls -la mb
awk '/\.amdgcn_target|^[a-zA-Z_0-9]+:|; (NumVgprs|NumAgprs|TotalNumVgprs|ScratchSize|Occupancy):/' kernels.s | grep -vE "^amdhsa|^  " | tr -d ";" | awk '/^[a-zA-Z_0-9]+:/{printf "\n%s",$0;next} /NumV|NumA|Total|Scratch|Occ/{printf " %s %s",$1,$2}' | sed 's/;.*//' 
echo
