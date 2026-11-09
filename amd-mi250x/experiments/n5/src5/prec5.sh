#!/bin/bash
# Login-node precompile, CPU only. Core files off.
ulimit -c 0
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src5/n5env5.sh
cd $S; mkdir -p pre
for L in 2816 3072 3584; do (TL_I r5 "n5_r5.py --plan plans5/grid$L.json --compile" 2>&1 | grep -E '^\{|Traceback' > pre/grid$L.out; echo done grid$L >> pre/state) & done; wait
(TL_I r5 "n5_r5.py --plan plans5/v22.json --compile" 2>&1 | grep -E '^\{|Traceback' > pre/v22.out) &
(TL_I r5 "n5_r5.py --plan plans5/batch.json --compile" 2>&1 | grep -E '^\{|Traceback' > pre/batch.out) &
(TL_L r5 "n5_r5.py --plan plans5/v18.json --compile" 2>&1 | grep -E '^\{|Traceback' > pre/v18.out) &
wait; echo done val >> pre/state
TL_L r5 "n5_r5.py --plan plans5/attr.json --compile" 2>&1 | grep -E '^\{|Traceback' > pre/attr.out
N5_ENV="HIPCC_COMPILE_FLAGS_APPEND='-mllvm -misched=gcn-max-ilp'" TL_L r5ilp "n5_r5.py --plan plans5/attr_ilp.json --compile" 2>&1 | grep -E '^\{|Traceback' > pre/attr_ilp.out
echo done attr >> pre/state
export C3_TILES=$S/plans5/sweep16.json C3_FORMS=SDRF,JDRF FM_ENV=""
for rx in '_b32_' '_b48_' '_b(64|80|96)_'; do (FM5_L "--compile --re '$rx'" 2>&1 | grep -E '^\{|Traceback' > "pre/sweep_${rx//[^0-9]/}.out") & done; wait
echo ALLDONE >> pre/state
