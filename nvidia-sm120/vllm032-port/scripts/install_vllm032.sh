#!/bin/bash
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=~/.cache/uv-032
V=$HK_VENV
echo "start $(date -u +%FT%TZ)"
uv venv $V --python 3.12 || exit 1
uv pip install --python $V/bin/python --link-mode copy "vllm==0.32.0" "huggingface_hub[cli,hf_xet]" || { echo INSTALL_FAILED; exit 1; }
uv pip freeze --python $V/bin/python > ~/vllm032_freeze.txt
rm -rf $UV_CACHE_DIR
df -h $HK_DATA | tail -1; du -sh $V
echo "INSTALL_DONE $(date -u +%FT%TZ)"
