#!/bin/bash
# Login node, CPU only: fill the kernel caches for md.py on the image stack.
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
cd $E; mkdir -p out/mdcomp
( export FM_ENV="C2P_ARM=fork"; source $S/cenv.sh; FM_I "src/md.py --compile" ) > out/mdcomp/fork.out 2>&1 &
( export FM_ENV="C2P_ARM=forkx C2P_FASTEXP=1"; source $S/cenv.sh; FM_I "src/md.py --compile" ) > out/mdcomp/forkx.out 2>&1 &
( export FM_ENV="C2P_ARM=wheel"; source $S/cenv.sh; WH_I "src/md.py --compile" ) > out/mdcomp/wheel.out 2>&1 &
wait; echo COMPDONE > out/mdcomp/done
