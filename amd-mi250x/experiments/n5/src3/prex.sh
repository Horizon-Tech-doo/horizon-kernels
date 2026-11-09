#!/bin/bash
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
N5_ENV='N5_NOX=1'
TL_I r3x 'n5_r3.py --plan plans3/nox.json --compile' 2>&1 | grep -E '^\{|Traceback|Error'
N5_ENV=''
TL_I r3 'n5_r3.py --plan plans3/nox.json --compile' 2>&1 | grep -E '^\{|Traceback|Error'
echo PRE_DONE
