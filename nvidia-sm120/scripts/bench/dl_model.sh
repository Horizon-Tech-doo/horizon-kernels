# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
export PATH=$HOME/.local/bin:$PATH HF_HOME=$HK_DATA/huggingface
uvx --from "huggingface_hub[cli,hf_xet]" hf download Qwen/Qwen3.8-27B-FP8 && echo DL_DONE
