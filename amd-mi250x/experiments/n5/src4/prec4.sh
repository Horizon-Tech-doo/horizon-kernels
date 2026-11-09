#!/bin/bash
# Login-node precompile. $1: I or L, $2: flavour (r4, r4ilp, ...), $3: flag set key or base, $4: plan. Core files off.
ulimit -c 0
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
source $S/flagsets4.sh
N5_ENV=""; [ "$3" != base ] && N5_ENV="HIPCC_COMPILE_FLAGS_APPEND='${FS[$3]}'"
if [ "$1" = I ]; then TL_I $2 "n5_r4.py --plan plans4/$4.json --compile"; else TL_L $2 "n5_r4.py --plan plans4/$4.json --compile"; fi 2>&1 | grep -E "^\{|Traceback|Error" | sed "s/^{/{\"stack\": \"$1\", \"flagset\": \"$3\", /"
echo "PRE_DONE $1 $2 $4"
