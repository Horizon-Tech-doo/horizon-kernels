#!/bin/bash
# $1 stack (I or L), $2 plan, $3 regex
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
N5_ENV=""
if [ "$1" = I ]; then TL_I r3 "n5_r3.py --plan plans3/$2.json --compile --re $3"; else TL_L r3 "n5_r3.py --plan plans3/$2.json --compile --re $3"; fi 2>&1 | grep -E "^\{|Traceback|Error"
echo PRE_DONE
