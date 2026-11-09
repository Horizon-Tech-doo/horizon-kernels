#!/bin/bash
# Login node, CPU only. Re-apply the local series on clean upstream, compare with the built tree byte for byte.
R=/scratch/project_XXXXXXXXX/tilelang; M=$R/fork-main; T=$R/exp/C2P/r5; C=$T/clean
rm -rf $C; mkdir -p $C; git -C $M archive 6e8f599e2fa41f05bb812edd8d06985b04f4f18b | tar xf - -C $C
cd $C; for p in $T/patches/0*.patch; do echo "$(basename $p): $(patch -p1 -F0 --no-backup-if-mismatch < $p | grep -c -E 'fuzz|offset|FAILED') fuzz_offset_fail_lines"; done
n=0; d=0; for f in $(git -C $M status --short | awk '{print $2}'); do
  if [ -d $M/$f ]; then for g in $(cd $M; find $f -type f ! -name '*.pyc'); do n=$((n+1)); cmp -s $M/$g $C/$g || { d=$((d+1)); echo DIFF $g; }; done
  else n=$((n+1)); cmp -s $M/$f $C/$f || { d=$((d+1)); echo DIFF $f; }; fi; done
echo "compared=$n differ=$d"; ls -la --time-style=long-iso $M/build/lib/libtilelang.so | awk '{print $6,$7}'
find $M/src $M/tilelang -newer $M/build/lib/libtilelang.so -type f ! -name '*.pyc' | head
rm -rf $C; echo DONE
