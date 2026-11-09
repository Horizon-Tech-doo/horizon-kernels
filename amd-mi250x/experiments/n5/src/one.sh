#!/bin/bash
# one.sh <stack L|I> <flavour> <regex>: compile one on login and dump
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
N5_ENV="N5_DUMP=$E/dump"
TL_$1 "$2" "n5_compile.py $3" 2>&1 | grep -E '^\{|^\[patch|Traceback|Error'
