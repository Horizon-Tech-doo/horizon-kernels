# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Formats track (27-09-2026): shadow model folder for the checkpoint-mix arms (horizon_kernels/ckpt_mix.py mode 2).
#   make_mix_shadow.py BASE_DIR OUT_DIR gdn|attn[,..]
# Every file of BASE_DIR (default use: $HK_WORK/nvours/nv-hmtp = NVIDIA's checkpoint + our retrained MTP layer) is
# symlinked, except config.json and hf_quant_config.json, which are rewritten so that the chosen FP8 projection layers
# are declared NVFP4 (group 16); the load-time swap then feeds them our NVFP4 tensors. Nothing large is written.
import json, os, re, sys

base, out, kinds = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
PAT = {"gdn": r"\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$", "attn": r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"}
rx = re.compile("|".join(PAT[k] for k in kinds))
os.makedirs(out, exist_ok=True)
for f in os.listdir(base):
    if f in ("config.json", "hf_quant_config.json"):
        continue
    src = os.path.realpath(os.path.join(base, f))
    dst = os.path.join(out, f)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)
NV = {"quant_algo": "NVFP4", "group_size": 16}
c = json.load(open(os.path.join(base, "config.json")))
q = c["quantization_config"]
moved = [n for n, v in q["quantized_layers"].items() if rx.search(n) and v.get("quant_algo") == "FP8"]
for n in moved:
    q["quantized_layers"][n] = dict(NV)
g0, g1 = q["config_groups"]["group_0"], q["config_groups"]["group_1"]
assert g0["weights"]["num_bits"] == 8 and g1["weights"]["num_bits"] == 4
g0["targets"] = [t for t in g0["targets"] if t not in moved]
g1["targets"] = sorted(set(g1["targets"]) | set(moved))
json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=2)
h = json.load(open(os.path.join(base, "hf_quant_config.json")))
for n in moved:
    h["quantization"]["quantized_layers"][n] = dict(NV)
json.dump(h, open(os.path.join(out, "hf_quant_config.json"), "w"), indent=2)
print(f"{out}: {len(moved)} layers FP8 -> NVFP4 ({kinds})")
