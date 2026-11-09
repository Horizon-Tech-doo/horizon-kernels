# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import os, json, time, shutil, threading, queue, subprocess, traceback, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
import sys
if len(sys.argv) != 2:
    sys.exit("usage: quant_nvfp4.py <hf-model-id>  (the BF16 source checkpoint)")
REPO=sys.argv[1]
SRC="$HK_DATA/models/checkpoint-src"; OUT="$HK_DATA/models/Qwen3.8-27B-variant-NVFP4"
T="$HK_DATA/templates/qwen-fp8"
def log(*a): print(time.strftime("%d-%m-%Y %H:%M:%S"),*a,flush=True)
try:
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration
    cfg=AutoConfig.from_pretrained(SRC)
    with torch.device("cuda"): model=Qwen3_5ForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
    model.eval(); sd_ref=model.state_dict(); log("model allocated", torch.cuda.memory_allocated()/1e9,"GB")
    idx=json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    shards=sorted(set(idx.values())); q=queue.Queue(maxsize=2)
    def dl():
        for sh in shards:
            p=None
            for a in range(8):
                try: p=hf_hub_download(REPO,sh,local_dir=SRC); break
                except Exception as e: log("retry",sh,e); time.sleep(15)
            q.put((sh,p))
    threading.Thread(target=dl,daemon=True).start()
    loaded=set(); mtp={}
    for n in range(len(shards)):
        sh,p=q.get(); sd=load_file(p)
        for k,v in sd.items():
            if k.startswith("mtp."): mtp[k]=v.clone()
            else: sd_ref[k].copy_(v.to("cuda")); loaded.add(k)
        os.remove(p); del sd; log(f"loaded {n+1}/{len(shards)} {sh}")
    missing=set(sd_ref)-loaded; assert not missing, list(missing)[:5]
    assert len(mtp)==15, len(mtp)
    tok=AutoTokenizer.from_pretrained(T)
    from datasets import load_dataset
    ds=load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]").shuffle(seed=42).select(range(256))
    def prep(ex):
        t=tok.apply_chat_template(ex["messages"], tokenize=False)
        return tok(t, max_length=2048, truncation=True, add_special_tokens=False)
    ds=ds.map(prep, remove_columns=ds.column_names)
    log("calib ready", len(ds))
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    ignore=["re:.*lm_head","re:visual.*","re:model.visual.*","re:.*embed_tokens$","re:.*linear_attn\\.in_proj_a$","re:.*linear_attn\\.in_proj_b$","re:^mtp.*"]
    recipe=QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=2048, num_calibration_samples=256, pipeline="basic")
    log("calibrated; saving")
    model.save_pretrained(OUT, save_compressed=True)
    save_file(mtp, f"{OUT}/model_mtp.safetensors", metadata={"format":"pt"})
    ip=f"{OUT}/model.safetensors.index.json"
    if os.path.exists(ip): ix=json.load(open(ip))
    else:
        from safetensors import safe_open
        with safe_open(f"{OUT}/model.safetensors","pt") as f: ix={"metadata":{},"weight_map":{k:"model.safetensors" for k in f.keys()}}
    for k in mtp: ix["weight_map"][k]="model_mtp.safetensors"
    ix["metadata"]["total_size"]=sum(os.path.getsize(f"{OUT}/{f}") for f in set(ix["weight_map"].values()))
    json.dump(ix,open(ip,"w"),indent=2)
    c=json.load(open(f"{OUT}/config.json")); qc=c["quantization_config"]
    if "re:^mtp.*" not in qc.get("ignore",[]): qc.setdefault("ignore",[]).append("re:^mtp.*")
    json.dump(c,open(f"{OUT}/config.json","w"),indent=2)
    for f in ["generation_config.json","chat_template.jinja","tokenizer.json","tokenizer_config.json","vocab.json","preprocessor_config.json","video_preprocessor_config.json"]:
        shutil.copy(f"{T}/{f}",f"{OUT}/{f}")
    for f in ["merges.txt","checkpoint_metadata.json"]: shutil.copy(f"{SRC}/{f}",f"{OUT}/{f}")
    log("DONE", ix["metadata"]["total_size"]/1e9, "GB", "ignore:", [i for i in qc["ignore"] if i.startswith("re:")] , "n_ignore", len(qc["ignore"]))
except Exception:
    log("FAILED"); traceback.print_exc()
finally:
    try: del model
    except: pass
    torch.cuda.empty_cache()
    log("run finished")  # (site-specific: restart your vLLM service here)
