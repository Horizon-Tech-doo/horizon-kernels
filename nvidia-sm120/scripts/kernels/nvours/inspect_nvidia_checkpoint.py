# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json, re, collections, struct, os
M="$HK_DATA/models/nvidia-Qwen3.8-27B-NVFP4"
c=json.load(open(f"{M}/config.json"))
q=c.get("quantization_config",{})
print("config quant keys:", {k:(v if not isinstance(v,(dict,list)) else type(v).__name__) for k,v in q.items()})
for g,v in q.get("config_groups",{}).items():
    print(g, {k:v[k] for k in v if k!="targets"}, "ntargets", len(v.get("targets",[])))
    tg=collections.Counter(re.sub(r"layers\.\d+","layers.N",t) for t in v.get("targets",[]))
    print("   ", dict(tg))
print("ignore:", q.get("ignore", q.get("exclude_modules"))[:40] if q.get("ignore", q.get("exclude_modules")) else None)
h=json.load(open(f"{M}/hf_quant_config.json"))["quantization"]
print("hf_quant keys", [k for k in h if k!="quantized_layers"], {k:h[k] for k in h if k not in("quantized_layers",)})
ql=collections.Counter((re.sub(r"layers\.\d+","layers.N",k), json.dumps(v)) for k,v in h["quantized_layers"].items())
for k,v in sorted(ql.items()): print("  ", k, v)
idx=json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
# read headers
hdr={}
for f in sorted(set(idx.values())):
    with open(f"{M}/{f}","rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; d=json.loads(fh.read(n)); d.pop("__metadata__",None)
        for k,v in d.items(): hdr[k]=(v["dtype"],v["shape"])
pat=collections.OrderedDict()
for k,(dt,sh) in hdr.items():
    kk=re.sub(r"layers\.\d+","layers.N",k)
    if "visual" in kk: kk="visual.*"
    pat.setdefault(kk,set()).add((dt,tuple(sh)))
for k,v in pat.items():
    if k=="visual.*": continue
    print(k, sorted(v)[:3], "+%d"%(len(v)-3) if len(v)>3 else "")
print("visual tensors:", sum(1 for k in hdr if "visual" in k))
