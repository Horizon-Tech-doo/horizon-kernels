# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json, re, sys, time, urllib.request, random
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,"$HK_DATA/llmc-pkgs")
from datasets import load_dataset
TAG=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 200
URL="http://localhost:8000/v1/chat/completions"
def chat(msg, max_tokens=1536):
    b=json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":msg}],"max_tokens":max_tokens,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(URL,b,{"Content-Type":"application/json"}),timeout=600))
    return r["choices"][0]["message"]["content"] or ""
g=load_dataset("openai/gsm8k","main",split="test").select(range(N))
def gsm(ex):
    out=chat(ex["question"]+"\nSolve step by step, then give the final answer on the last line as: Answer: <number>")
    gold=ex["answer"].split("####")[-1].strip().replace(",","")
    m=re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)",out) or re.findall(r"(-?[\d,]*\.?\d+)",out)
    pred=m[-1].replace(",","").rstrip(".") if m else ""
    try: return float(pred)==float(gold)
    except: return False
mp=load_dataset("TIGER-Lab/MMLU-Pro",split="test").shuffle(seed=0).select(range(N))
L="ABCDEFGHIJ"
def mmlu(ex):
    opts="\n".join(f"{L[i]}. {o}" for i,o in enumerate(ex["options"]))
    q=ex["question"]
    out=chat(f"{q}\n\n{opts}\n\nThink briefly, then answer on the last line as: Answer: <letter>",1024)
    m=re.findall(r"Answer:\s*\(?([A-J])\b",out)
    return bool(m) and m[-1]==ex["answer"]
t=time.time()
with ThreadPoolExecutor(8) as ex:
    a=sum(ex.map(gsm,g)); b=sum(ex.map(mmlu,mp))
b2=json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":"Write a 400-word essay about the history of the printing press."}],"max_tokens":600,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
sp=[]
for i in range(3):
    s=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(URL,b2,{"Content-Type":"application/json"}))); sp.append(r["usage"]["completion_tokens"]/(time.time()-s))
res={"model":TAG,"gsm8k":a/N,"mmlu_pro":b/N,"n":N,"single_stream_tok_s":round(sum(sp)/3,1),"eval_min":round((time.time()-t)/60,1)}
print(json.dumps(res)); open("~/eval_results.jsonl","a").write(json.dumps(res)+"\n")
