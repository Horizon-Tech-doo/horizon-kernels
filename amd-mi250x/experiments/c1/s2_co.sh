#!/bin/bash
# Compile fork JIT sources with the JIT's exact flags to code objects for the bare harness (no python wrapper).
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
HC=$(python -c "from tilelang.contrib.rocm import find_hipcc; print(find_hipcc())" 2>/dev/null)
FL="-O3 -c -gline-tables-only --offload-arch=gfx90a --genco -std=c++17 -I$TPL"
for s in 4k 8k 32k 17k 16k; do $HC $FL -o s2co/$s.hsaco s2co/$s.cc || echo FAIL $s; done; ls -la s2co | head -12
# a9 limit check
for T in 256; do for N in 7 9; do sed -E "s/__launch_bounds__\([0-9]+\)/__launch_bounds__($T, $N)/" s1src/t$T.cu > s1/t${T}_a$N.cc; $HC $FL -Rpass-analysis=kernel-resource-usage -o s1/t${T}_a$N.hsaco s1/t${T}_a$N.cc > s1/t${T}_a$N.log 2>&1; done; done
python3 s1_table.py > s1_table.txt; grep -E "^256 a(7|8|9|10) " s1_table.txt
