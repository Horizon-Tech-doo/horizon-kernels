#!/bin/bash
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Body of the training context-length ladder (27-09-2026); run by run_trainctx.sh with vLLM stopped, under a hard
# 150-minute timeout. Order: GPU probe -> r1 smoke (1k + correctness) -> r2 smoke + r2 ladder -> r1 ladder.
set -uo pipefail
D=$HK_WORK/trainctx; P=$HK_VENV/bin/python
export LD_LIBRARY_PATH=$HK_DATA/cuda-compat PYTHONPATH=$HK_DATA/llmc-pkgs
export TRITON_CACHE_DIR=$D/triton MAX_JOBS=4 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
T0=$(date +%s)
LENGTHS=8192,16384,32768,65536,131072
mkdir -p $D/triton $D/tmp
echo "== $(date -u '+%d-%m-%Y %H:%M') UTC body start"

# ---- probe (SDPA backends at head_dim 256, causal_conv1d, FLA, Triton dequant); expandable segments if it works
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if ! timeout 900 $P $D/trainctx.py --probe $D/probe.json; then
  echo "probe failed with expandable_segments: retrying with the default allocator"
  export PYTORCH_CUDA_ALLOC_CONF=
  timeout 900 $P $D/trainctx.py --probe $D/probe.json || { echo "PROBE FAILED"; exit 5; }
fi
echo "allocator: PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-default}"

left() { echo $(( $1 - $(date +%s) )); }

# ---- r1 smoke (BF16-dequantized base): 1k steps + correctness vs the served model
timeout 1500 $P $D/trainctx.py --rep r1 --smoke --deadline $(( $(date +%s) + 1200 )) \
  || echo "r1 smoke rc=$?"

# ---- r2 (NVFP4-packed base): smoke + ladder; its ladder may use the time up to T0 + 95 min
DL2=$(( T0 + 95 * 60 ))
timeout $(( $(left $DL2) + 900 )) $P $D/trainctx.py --rep r2 --smoke --lengths $LENGTHS --deadline $DL2 \
  || echo "r2 rc=$?"

# ---- r1 ladder, until T0 + 140 min (the outer timeout is 150 min)
DL1=$(( T0 + 140 * 60 ))
if [ $(left $DL1) -gt 600 ]; then
  timeout $(( $(left $DL1) + 480 )) $P $D/trainctx.py --rep r1 --lengths $LENGTHS --deadline $DL1 \
    || echo "r1 ladder rc=$?"
else
  echo "r1 ladder skipped: not enough time left"
fi
echo "TRAINCTX_BODY_DONE $(date -u '+%d-%m-%Y %H:%M') UTC"
