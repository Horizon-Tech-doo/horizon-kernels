#!/bin/bash
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
source $S/flagsets2.sh
k=$1
N5_ENV="HIPCC_COMPILE_FLAGS_APPEND='${FS[$k]}'"
TL_I r3$k 'n5_r3.py --plan plans3/flags.json --compile' 2>&1 | grep -E '^\{|Error'
echo PRE_DONE
