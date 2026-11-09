# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json, struct, re
M="$HK_DATA/models/nvidia-Qwen3.8-27B-NVFP4"
idx=json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
cache={}
def hdr(f):
    if f not in cache:
        fh=open(f"{M}/{f}","rb"); n=struct.unpack("<Q",fh.read(8))[0]; cache[f]=(json.loads(fh.read(n)),8+n)
    return cache[f]
def scalar(name):
    f=idx[name]; h,off=hdr(f); e=h[name]; a,b=e["data_offsets"]
    with open(f"{M}/{f}","rb") as fh: fh.seek(off+a); return struct.unpack("<f",fh.read(4))[0]
tied=0; untied=[]
for L in range(64):
    p=f"model.language_model.layers.{L}.mlp."
    g=scalar(p+"gate_proj.weight_scale_2"); u=scalar(p+"up_proj.weight_scale_2")
    gi=scalar(p+"gate_proj.input_scale"); ui=scalar(p+"up_proj.input_scale")
    if g==u: tied+=1
    else: untied.append((L,g,u,max(g,u)/min(g,u)))
    if L in (0,1,3,31,63): print(L,"gate ws2",g,"up ws2",u,"in",gi,ui,"down ws2",scalar(p+"down_proj.weight_scale_2"))
print("gate/up weight_scale_2 tied in",tied,"of 64 layers")
for x in untied[:10]: print("  untied",x)
if untied: print("  ratio range", min(x[3] for x in untied), max(x[3] for x in untied))
print("lm_head ws2", scalar("lm_head.weight_scale_2"), "input", scalar("lm_head.input_scale"))
# FP8 qkv/z per-tensor scales tiedness (fused in_proj_qkvz in vLLM)
d=[]
for L in range(64):
    p=f"model.language_model.layers.{L}.linear_attn."
    if p+"in_proj_qkv.weight_scale" in idx:
        d.append((L,scalar(p+"in_proj_qkv.weight_scale"),scalar(p+"in_proj_z.weight_scale")))
print("GDN qkv vs z FP8 weight_scale equal in", sum(a==b for _,a,b in d), "of", len(d), d[:2])
d=[]
for L in range(64):
    p=f"model.language_model.layers.{L}.self_attn."
    if p+"q_proj.weight_scale" in idx:
        d.append((L,)+tuple(scalar(p+x+"_proj.weight_scale") for x in "qkv"))
print("attn q/k/v FP8 weight_scale all equal in", sum(a==b==c for _,a,b,c in d), "of", len(d), d[:2])
