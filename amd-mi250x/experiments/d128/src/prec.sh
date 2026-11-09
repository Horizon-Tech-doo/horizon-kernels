#!/bin/bash
# Login node, CPU only. Keeps hand and form bytes in $D/bytes, then a second independent compile of the same arms for the determinism check.
ulimit -c 0
source /scratch/project_XXXXXXXXX/tilelang/exp/D128/src/env.sh
cd $SRC; python3 plan.py; mkdir -p $D/bytes $D/bytes2 $D/pre $D/cache
md5sum $SRC/gqa_fa.py $SRC/tlfa.py $F/build/lib/libtilelang.so > $D/pre/md5.txt
( WH $D/cache/W "handcap.py $D/bytes" 2>&1 | grep -E '^\{|Traceback|Error' > $D/pre/hand.out ) &
for rx in '_(2816|3072)_.*JDRF|_(2816|3072)_.*SDRF' '_(3584|4096)_.*RF' '_8192_.*RF' '_16384_.*RF'; do
  n=$(echo "$rx" | tr -dc '0-9' | cut -c1-8)
  ( FM $D/cache/F "d128.py --compile $D/bytes --re '$rx'" 2>&1 | grep -E '^\{|Traceback|Error' > $D/pre/form_$n.out ) &
done
wait
# hand wrappers need the bytes first
FM $D/cache/F "d128.py --compile $D/bytes --re 'hand'" 2>&1 | grep -E '^\{|Traceback|Error' > $D/pre/wrap.out
# determinism: second compile in a fresh cache and temp dir
( FM $D/cache/F2 "d128.py --compile $D/bytes2 --re '_(2816|3072|3584|8192)_r128_b64_.*JDRF|_16384_r256_b32_.*SDRF'" 2>&1 | grep -E '^\{|Traceback|Error' > $D/pre/form2.out ) &
( WH $D/cache/W2 "handcap.py $D/bytes2 --re '_(3072|8192)_'" 2>&1 | grep -E '^\{|Traceback|Error' > $D/pre/hand2.out ) &
wait; echo ALLDONE > $D/pre/state
