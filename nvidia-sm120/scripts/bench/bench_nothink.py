# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json,time,urllib.request,concurrent.futures as cf,sys
P="Write a PowerShell function that lists Active Directory users whose passwords expire in the next 14 days, with comments."
def one(_):
    t=time.time()
    req=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":P}],"max_tokens":1500,"chat_template_kwargs":{"enable_thinking":False}}).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req,timeout=600))
    return d["usage"]["completion_tokens"],time.time()-t,d["choices"][0]["message"].get("content") or ""
one(0)  # warmup
for n in [int(x) for x in sys.argv[1:]] or (1,2):
    t=time.time()
    with cf.ThreadPoolExecutor(n) as ex: r=list(ex.map(one,range(n)))
    w=time.time()-t; tok=sum(x[0] for x in r)
    print(f"{n} parallel: {tok} tokens in {w:.1f}s -> per request {r[0][0]/r[0][1]:.1f} tok/s, total {tok/w:.1f} tok/s")
print(r[0][2][:400])
