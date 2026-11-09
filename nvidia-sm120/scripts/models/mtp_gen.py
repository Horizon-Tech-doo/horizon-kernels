# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Stage 1: self-distillation data - the served model answers a mixed prompt set (chat, maths, code; thinking on/off)
import json, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,"$HK_DATA/llmc-pkgs")
from datasets import load_dataset
OUT="$HK_DATA/models/mtp-data/gen.jsonl"; URL="http://localhost:8000/v1/chat/completions"
random.seed(0)
P=[]
uc=load_dataset("HuggingFaceH4/ultrachat_200k",split="train_sft[2000:6000]")   # 0-2000 used for NVFP4 calibration
P+=[m["messages"][0]["content"] for m in uc.select(range(2400))]
P+=[x["question"] for x in load_dataset("openai/gsm8k","main",split="train").shuffle(seed=1).select(range(700))]
P+=[x["problem"] for x in load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K",split="train").shuffle(seed=1).select(range(900))]
random.shuffle(P)
def gen(i_p):
    i,p=i_p; think = (i%4==0)
    body={"model":"qwen3.8","messages":[{"role":"user","content":p}],"max_tokens":1536,"temperature":0.7,"top_p":0.9,
          "chat_template_kwargs":{"enable_thinking":think}}
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900))
        m=r["choices"][0]["message"]
        return {"prompt":p,"think":think,"reasoning":m.get("reasoning") or m.get("reasoning_content") or "","content":m.get("content") or ""}
    except Exception as e: return None
t=time.time(); n=0
with open(OUT,"w") as f, ThreadPoolExecutor(48) as ex:
    for res in ex.map(gen, enumerate(P)):
        if res and res["content"]: f.write(json.dumps(res)+"\n"); n+=1
        if n%200==0: print(time.strftime("%d-%m-%Y %H:%M:%S"),n,"done",round(time.time()-t),"s",flush=True)
print("DONE",n,flush=True)
