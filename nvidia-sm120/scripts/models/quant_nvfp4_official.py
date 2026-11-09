# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import os, json, time, shutil, threading, queue, subprocess, traceback, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
REPO="Qwen/Qwen3.8-27B"
SRC="$HK_DATA/models/qwen-src"; OUT="$HK_DATA/models/Qwen3.8-27B-NVFP4"
KEEP="model-00018-of-00018.safetensors"   # holds MTP; llm-compressor copies it at save time
def log(*a): print(time.strftime("%d-%m-%Y %H:%M:%S"),*a,flush=True)
ok=False
try:
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration
    cfg=AutoConfig.from_pretrained(SRC)
    with torch.device("cuda"): model=Qwen3_5ForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
    model.eval(); model.name_or_path=SRC; model.config._name_or_path=SRC
    sd_ref=model.state_dict(); log("model allocated", round(torch.cuda.memory_allocated()/1e9,1),"GB")
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
    loaded=set()
    for n in range(len(shards)):
        sh,p=q.get(); sd=load_file(p)
        for k,v in sd.items():
            if not k.startswith("mtp."): sd_ref[k].copy_(v.to("cuda")); loaded.add(k)
        if sh!=KEEP: os.remove(p)
        del sd; log(f"loaded {n+1}/{len(shards)} {sh}")
    missing=set(sd_ref)-loaded; assert not missing, list(missing)[:5]
    tok=AutoTokenizer.from_pretrained(SRC)
    from datasets import load_dataset
    ds=load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:2000]").shuffle(seed=42).select(range(512))
    def prep(ex): return tok(tok.apply_chat_template(ex["messages"], tokenize=False), max_length=2048, truncation=True, add_special_tokens=False)
    ds=ds.map(prep, remove_columns=ds.column_names); log("calib ready", len(ds))
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    ignore=["re:.*lm_head","re:visual.*","re:model.visual.*","re:.*embed_tokens$","re:.*linear_attn\\.in_proj_a$","re:.*linear_attn\\.in_proj_b$","re:^mtp.*"]
    recipe=QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=2048, num_calibration_samples=512,
            pipeline="sequential", sequential_targets=["Qwen3_5DecoderLayer"], sequential_offload_device="cuda")
    log("calibrated; saving")
    model.save_pretrained(OUT, save_compressed=True)
    for f in ["generation_config.json","chat_template.jinja","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","preprocessor_config.json","video_preprocessor_config.json","LICENSE"]:
        if os.path.exists(f"{SRC}/{f}"): shutil.copy(f"{SRC}/{f}",f"{OUT}/{f}")
    ix=json.load(open(f"{OUT}/model.safetensors.index.json")); c=json.load(open(f"{OUT}/config.json"))
    nm=sum(k.startswith("mtp.") for k in ix["weight_map"])
    log("DONE mtp tensors:",nm,"size GB:",round(sum(os.path.getsize(f"{OUT}/{f}") for f in set(ix["weight_map"].values()))/1e9,2),"mtp ignored:","re:^mtp.*" in c["quantization_config"]["ignore"])
    ok=True
except Exception:
    log("FAILED"); traceback.print_exc()
finally:
    try: del model
    except: pass
    torch.cuda.empty_cache()
    log("run finished")  # (site-specific: restart your vLLM service here)
