#!/bin/bash
# Fresh compile of the final kernels: registers, occupancy, scratch, race check, generated source, assembly op counts. One empty cache per stack.
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
for st in L I; do
  rm -rf $E/cache/${st}OYBD_$PH
  for one in "d256_8192_P32_b32_t768_land_fullERDW" "d256_8192_P32_b48_t768_land_fullERW" "d128_4096_r128_b32_t512_land_fullERDW" "d128_4096_P32_b96_t768_land_fullERDW" "mha_4096_P160_b32_t640_land_full" "mha_4096_P192_b96_t768_land_full"; do
    N5_ENV="N5_DUMP=$E/dump"
    TL_$st OYBD "n5_compile.py $one\$" 2>&1 | grep -E '^\{|^\[patch' | sed "s/^/$st /"
  done
done
