#!/bin/bash
# Inside the container: -fsyntax-only on the CUDA files that use the moved attribute key, against the final tree, with the flags ninja uses for a ROCm source.
F=/scratch/project_XXXXXXXXX/tilelang/fork-main; B=$F/build; cd $B
CMD=$(ninja -t commands tilelang | grep -m1 "src/rocm/op/reduce.cc" | sed 's/ -MD//; s/ -MT [^ ]*//; s/ -MF [^ ]*//; s/ -o [^ ]*//; s/ -c / /; s# [^ ]*src/rocm/op/reduce.cc##')
for f in src/cuda/codegen/codegen_cuda.cc src/cuda/transform/auto_schedule/role_based_scheduler.cc src/cuda/transform/materialize_ws_schedule.cc; do
  $CMD -fsyntax-only -w $F/$f > /tmp/c2p_syn.txt 2>&1; echo "$f rc=$? errors=$(grep -c ' error:' /tmp/c2p_syn.txt)"; grep ' error:' /tmp/c2p_syn.txt | head -3 | cut -c1-200
done
rm -f /tmp/c2p_syn.txt
