# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json, os, shutil
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
import sys
if len(sys.argv) != 2:
    sys.exit("usage: finish_nvfp4.py <hf-model-id>  (the BF16 source checkpoint)")
REPO=sys.argv[1]
SRC="$HK_DATA/models/checkpoint-src"; OUT="$HK_DATA/models/Qwen3.8-27B-variant-NVFP4"; T="$HK_DATA/templates/qwen-fp8"
p=hf_hub_download(REPO,"model-extra-00001-of-00001.safetensors",local_dir=SRC)
with safe_open(p,"pt") as f: mtp={k:f.get_tensor(k) for k in f.keys() if k.startswith("mtp.")}
assert len(mtp)==15; save_file(mtp,f"{OUT}/model_mtp.safetensors",metadata={"format":"pt"}); os.remove(p)
with safe_open(f"{OUT}/model.safetensors","pt") as f: keys=list(f.keys())
wm={k:"model.safetensors" for k in keys}; wm.update({k:"model_mtp.safetensors" for k in mtp})
tot=sum(os.path.getsize(f"{OUT}/{x}") for x in ["model.safetensors","model_mtp.safetensors"])
json.dump({"metadata":{"total_size":tot},"weight_map":dict(sorted(wm.items()))},open(f"{OUT}/model.safetensors.index.json","w"),indent=2)
c=json.load(open(f"{OUT}/config.json")); qc=c["quantization_config"]
if "re:^mtp.*" not in qc["ignore"]: qc["ignore"].append("re:^mtp.*")
json.dump(c,open(f"{OUT}/config.json","w"),indent=2)
for x in ["generation_config.json","chat_template.jinja","tokenizer.json","tokenizer_config.json","vocab.json","preprocessor_config.json","video_preprocessor_config.json"]: shutil.copy(f"{T}/{x}",f"{OUT}/{x}")
for x in ["merges.txt","checkpoint_metadata.json"]: shutil.copy(f"{SRC}/{x}",f"{OUT}/{x}")
g=qc["config_groups"]["group_0"]
print("tensors",len(keys),"mtp",len(mtp),"size GB",tot/1e9,"format",qc.get("format"),"w",g["weights"]["num_bits"],g["weights"]["type"],g["weights"]["group_size"],"a",(g.get("input_activations") or {}).get("num_bits"),"ignore_n",len(qc["ignore"]))
