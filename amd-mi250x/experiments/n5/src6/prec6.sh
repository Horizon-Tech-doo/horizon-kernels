#!/bin/bash
# Login-node precompile, CPU only. Core files off.
ulimit -c 0
export FM_ENV=""
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src6/n5env6.sh
cd $S6; mkdir -p pre; rm -f pre/*
export C3_ARMS=$S6/plans/forms.json
for rx in '_b64_' 'r256.*_SDRF' 'r256.*_JDRF' 'r128_b32'; do (FM6_I "c6.py --compile --re '$rx'" 2>&1 | grep -E '^\{|Traceback|Error' > "pre/f_${rx//[^0-9A-Z]/}.out") & done; wait
echo forms >> pre/state
(TL_I r5 "n5_r5.py --plan $S6/plans/tile22.json --compile" 2>&1 | grep -E '^\{|Traceback|Error' > pre/tile22.out) &
(TL_L r5 "n5_r5.py --plan $S6/plans/l18.json --compile" 2>&1 | grep -E '^\{|Traceback|Error' > pre/l18.out) &
wait; echo ALLDONE >> pre/state
