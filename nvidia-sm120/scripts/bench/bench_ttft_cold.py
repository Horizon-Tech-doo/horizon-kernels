# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json,time,urllib.request,random
random.seed()
import random as R; R.seed()
words="azure rust tenant subscription pipeline kernel policy firewall cluster gateway storage queue".split()
doc=" ".join(R.choice(words) for _ in range(9000))
def ttft(tag):
    body={"model":"qwen3.8","messages":[{"role":"system","content":str(R.random())+" You are a helpful assistant. "*200},{"role":"user","content":tag+" Summarize in one sentence: "+doc}],"max_tokens":20,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}
    req=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.time();first=None;ptok=0
    for line in urllib.request.urlopen(req,timeout=600):
        line=line.decode().strip()
        if not line.startswith("data:") or line=="data: [DONE]": continue
        d=json.loads(line[5:])
        if d.get("usage"): ptok=d["usage"]["prompt_tokens"]
        if first is None and d.get("choices") and d["choices"][0]["delta"].get("content"): first=time.time()-t
    return ptok,first
p,f=ttft("run-%d"%random.randint(0,10**9)); print(f"long prompt cold: {p} tokens, first token after {f:.2f}s ({p/f:.0f} tok/s prefill)")
