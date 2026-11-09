# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json,time,urllib.request,random,sys
R=random.Random()
words="azure rust tenant subscription pipeline kernel policy firewall cluster gateway storage queue".split()
def run(n_words):
    doc=" ".join(R.choice(words) for _ in range(n_words))
    body={"model":"qwen3.8","messages":[{"role":"user","content":str(R.random())+" Here is a log:\n"+doc+"\nWrite a 300 word essay about cloud security."}],"max_tokens":300,"stream":True,"chat_template_kwargs":{"enable_thinking":False},"stream_options":{"include_usage":True}}
    req=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.time();first=None;ptok=ctok=0
    for line in urllib.request.urlopen(req,timeout=900):
        line=line.decode().strip()
        if not line.startswith("data:") or line=="data: [DONE]": continue
        d=json.loads(line[5:])
        if d.get("usage"): ptok=d["usage"]["prompt_tokens"]; ctok=d["usage"]["completion_tokens"]
        if first is None and d.get("choices") and d["choices"][0]["delta"].get("content"): first=time.time()-t
    tot=time.time()-t
    print(f"{ptok:>7} prompt tokens: first token {first:6.1f}s ({ptok/first:6.0f} tok/s read), then {ctok/(tot-first):5.1f} tok/s output",flush=True)
for n in map(int,sys.argv[1:]): run(n)
