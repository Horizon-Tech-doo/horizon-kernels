# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json, os, sys, time, shutil, threading, queue, torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from huggingface_hub import hf_hub_download
torch.set_num_threads(48)
import sys
if len(sys.argv) != 2:
    sys.exit("usage: convert_fp8.py <hf-model-id>  (the BF16 source checkpoint)")
REPO=sys.argv[1]
SRC="$HK_DATA/models/checkpoint-src"; OUT="$HK_DATA/models/Qwen3.8-27B-variant-FP8"
T="$HK_DATA/templates/qwen-fp8"
os.makedirs(OUT, exist_ok=True)
idx=json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
tmpl=json.load(open(f"{T}/model.safetensors.index.json"))["weight_map"]
QUANT={k[:-len("_scale_inv")] for k in tmpl if k.endswith(".weight_scale_inv")}
assert all(q in idx for q in QUANT), "name mismatch"
shards=sorted(set(idx.values()))
B=128; FMAX=448.0
def quant(w):
    o,i=w.shape; w=w.float()
    po,pi=(-o)%B,(-i)%B
    wp=torch.nn.functional.pad(w,(0,pi,0,po))
    blk=wp.view((o+po)//B,B,(i+pi)//B,B)
    amax=blk.abs().amax(dim=(1,3)).clamp(min=1e-12)
    s=(amax/FMAX).to(torch.bfloat16)
    sf=s.float()[:,None,:,None]
    q=(blk/sf).clamp(-FMAX,FMAX).to(torch.float8_e4m3fn)
    deq=(q.float()*sf).view(o+po,i+pi)[:o,:i]
    err=((deq-w).norm()/w.norm().clamp(min=1e-12)).item()
    q=q.view(o+po,i+pi)[:o,:i].contiguous()
    return q,s.contiguous(),err
dlq=queue.Queue(maxsize=2)
def dl():
    for sh in shards:
        if os.path.exists(f"{OUT}/{sh}"): dlq.put((sh,None)); continue
        p=None
        for a in range(8):
            try: p=hf_hub_download(REPO,sh,local_dir=SRC); break
            except Exception as e: print("retry",sh,e,flush=True); time.sleep(15)
        dlq.put((sh,p))
threading.Thread(target=dl,daemon=True).start()
wm={}; maxerr=0
for n in range(len(shards)):
    sh,p=dlq.get(); t0=time.time()
    if p is None and os.path.exists(f"{OUT}/{sh}"):
        with safe_open(f"{OUT}/{sh}","pt") as f:
            for k in f.keys(): wm[k]=sh
        print(f"[{n+1}/{len(shards)}] {sh} already done",flush=True); continue
    if p is None: print("FAILED download",sh,flush=True); sys.exit(1)
    sd=load_file(p); out={}; nq=0
    for k,v in sd.items():
        if k in QUANT:
            q,s,e=quant(v); out[k]=q; out[k+"_scale_inv"]=s; nq+=1; maxerr=max(maxerr,e)
        else: out[k]=v
    save_file(out,f"{OUT}/{sh}.tmp",metadata={"format":"pt"}); os.replace(f"{OUT}/{sh}.tmp",f"{OUT}/{sh}")
    for k in out: wm[k]=sh
    os.remove(p)
    print(f"[{n+1}/{len(shards)}] {sh} tensors={len(sd)} quantized={nq} maxrelerr={maxerr:.4f} {time.time()-t0:.0f}s",flush=True)
    del sd,out
total=sum(os.path.getsize(f"{OUT}/{sh}") for sh in shards)
missing=set(tmpl)-set(wm); extra=set(wm)-set(tmpl)
print("missing",len(missing),list(missing)[:5],"extra",len(extra),list(extra)[:5],flush=True)
json.dump({"metadata":{"total_size":total},"weight_map":dict(sorted(wm.items()))},open(f"{OUT}/model.safetensors.index.json","w"),indent=2)
for f in ["config.json","generation_config.json","chat_template.jinja","tokenizer.json","tokenizer_config.json","vocab.json","preprocessor_config.json","video_preprocessor_config.json"]:
    shutil.copy(f"{T}/{f}",f"{OUT}/{f}")
for f in ["merges.txt","checkpoint_metadata.json"]: shutil.copy(f"{SRC}/{f}",f"{OUT}/{f}")
print("DONE total_size",total/1e9,"GB maxrelerr",maxerr,flush=True)
