#!/bin/bash
# N4 round 3, allocation follow-up at 16384^3 on 8 GCDs. Phase a: which buffer carries the mode (N = nothing changes = persistence control). Phase b: all arms on the same allocations, 7 reallocations, is any schedule immune. Phase c: same at 6144^3.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/${JN:-j32}_$J; mkdir -p $O; date -u > $O/time.txt
python map23.py > $O/map.txt; python pollm.py $O/metrics.jsonl 0.1 & PP=$!
arm(){ python -c "import json;d=json.load(open('h11/dyn.json'))['$1'];print('$1,h11/$1.co,%d,%d,256,%d'%(d['bM'],d['bN'],d['dyn']))"; }
run(){ tag=$1; M=$2; shift 2; seq=$1; it=$2; w=$3; shift 3; pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d ./hbr $M $M $M plain $seq $it $w "$@" > $O/${tag}_d$d.jsonl 2>&1 & pids+=($!); done; wait "${pids[@]}"; }
run a16 16384 XNANBNCNXNANBNCNXNANBNCN 8 1 hb $(arm h11_o128_16k)
run b16 16384 XXXXXXX 8 1 hb $(for t in o128 p16 a48b16 a96b16 fr15 fr40 c4 c16 r4 r8 c8; do arm h11_${t}_16k; done)
run c6 6144 XXXXXXXX 110 3 hb $(arm h11_o128_6k) $(arm h11_c8_6k)
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
