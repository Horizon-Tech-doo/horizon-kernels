# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json,time,urllib.request,random
R=random.Random()
words="azure rust tenant subscription pipeline kernel policy firewall cluster gateway storage queue".split()
doc=str(R.random())+" Here is a log:\n"+" ".join(R.choice(words) for _ in range(120000))
def ask(msgs):
    body={"model":"qwen3.8","messages":msgs,"max_tokens":100,"stream":True,"chat_template_kwargs":{"enable_thinking":False},"stream_options":{"include_usage":True}}
    req=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.time();first=None;ptok=0;txt=""
    for line in urllib.request.urlopen(req,timeout=900):
        line=line.decode().strip()
        if not line.startswith("data:") or line=="data: [DONE]": continue
        d=json.loads(line[5:])
        if d.get("usage"): ptok=d["usage"]["prompt_tokens"]
        if d.get("choices") and d["choices"][0]["delta"].get("content"):
            txt+=d["choices"][0]["delta"]["content"]
            if first is None: first=time.time()-t
    return ptok,first,txt
m=[{"role":"user","content":doc+"\nHow many times does the word azure appear, roughly?"}]
p,f,a=ask(m); print(f"turn 1: {p} tokens, first token after {f:.1f}s",flush=True)
m+= [{"role":"assistant","content":a},{"role":"user","content":"And which word seems most common?"}]
p,f,a=ask(m); print(f"turn 2 (follow-up, same conversation): {p} tokens, first token after {f:.1f}s",flush=True)
