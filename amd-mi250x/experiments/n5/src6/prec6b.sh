ulimit -c 0
export FM_ENV=""
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src6/n5env6.sh
TL_I r5 "n5_r5.py --plan $S6/plans/tile22b.json --compile" 2>&1 | grep -E "^\{|Traceback|Error" > $S6/pre/tile22b.out
