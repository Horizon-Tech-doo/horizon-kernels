#!/bin/bash
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Body of the follow-up run (27-09-2026), run by run_trainctx2.sh with vLLM stopped, hard 45-minute timeout.
set -uo pipefail
D=$HK_WORK/trainctx; P=$HK_VENV/bin/python
export LD_LIBRARY_PATH=$HK_DATA/cuda-compat PYTHONPATH=$HK_DATA/llmc-pkgs
export TRITON_CACHE_DIR=$D/triton MAX_JOBS=4 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T0=$(date +%s)
echo "== $(date -u '+%d-%m-%Y %H:%M') UTC follow-up body start"
timeout 900 $P $D/trainctx.py --probe $D/probe2.json || { echo "PROBE FAILED"; exit 5; }
cp $D/probe2.json $D/probe.json
# (1) correctness on the GPU: exact vs served shard scales
for m in served exact; do
  timeout 600 $P $D/trainctx.py --check-logits $D/check_logits.jsonl --rep r2 --device cuda --shard-scales $m
done
# (2) offload fallbacks, exactly pinned
timeout 1100 $P $D/trainctx.py --rep r1 --lengths 65536 --settings offload,offload+mlp-chunk --offload-gb 24 \
  --shard-scales served --deadline $(( T0 + 25 * 60 )) --out $D/results2.jsonl || echo "r1 64k rc=$?"
timeout 1100 $P $D/trainctx.py --rep r2 --lengths 131072 --settings offload,offload+mlp-chunk --offload-gb 24 \
  --shard-scales served --deadline $(( T0 + 40 * 60 )) --out $D/results2.jsonl || echo "r2 128k rc=$?"
echo "TRAINCTX2_BODY_DONE $(date -u '+%d-%m-%Y %H:%M') UTC"
