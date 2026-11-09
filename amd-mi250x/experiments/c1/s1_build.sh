#!/bin/bash
# Step 1: compile JIT device sources with TileLang's exact hipcc flags (contrib/hipcc.py + rocm/backend.py),
# plus the second __launch_bounds__ argument. No fork build. Records Occupancy from the compiler's own remark.
cd /scratch/project_XXXXXXXXX/tilelang/exp/C1
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
HC=$(python -c "from tilelang.contrib.rocm import find_hipcc; print(find_hipcc())" 2>/dev/null)
echo "hipcc=$HC"
FL="-O3 -c -gline-tables-only --offload-arch=gfx90a --genco -std=c++17 -I$TPL -Rpass-analysis=kernel-resource-usage"
echo "flags=$FL"
mkdir -p s1 
one(){ # name src bounds
 n=$1; src=$2; bo=$3
 sed -E "s/__launch_bounds__\([0-9]+\)/$bo/" $src > s1/$n.cc
 $HC $FL -o s1/$n.hsaco s1/$n.cc > s1/$n.log 2>&1; rc=$?
 $HC ${FL/ -c / -S } -o s1/$n.s s1/$n.cc > /dev/null 2>&1
 printf "%s rc=%s bounds=%s " $n $rc "$bo"
 grep -E "VGPRs:|AGPRs:|ScratchSize|Occupancy|LDS Size" s1/$n.log | sed -E 's/.*remark: [^:]*:[0-9:]* *//;s/.*Function Name[^ ]*//' | awk -F: '{gsub(/ /,"",$1);gsub(/ /,"",$2);printf "%s=%s ",$1,$2}' ; echo
}
for T in 64 128 256 512 768; do
 one t${T}_none s1src/t$T.cu "__launch_bounds__($T)" &
 for N in 1 2 3 4 5 6 8 10 11; do one t${T}_a$N s1src/t$T.cu "__launch_bounds__($T, $N)" & done
 one t${T}_attr2 s1src/t$T.cu "__launch_bounds__($T) __attribute__((amdgpu_waves_per_eu(2)))" &
 wait
done
sort -V <<< "" ; 
echo "== equality: hand attribute form vs two-argument form at 256 (asm, excluding debug lines)"
diff <(grep -vE '^\s*(\.file|\.loc|;)' s1/t256_a2.s) <(grep -vE '^\s*(\.file|\.loc|;)' s1/t256_attr2.s) | wc -l
