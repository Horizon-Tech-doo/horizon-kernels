# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json,time,sys,urllib.request
from concurrent.futures import ThreadPoolExecutor
URL="http://localhost:8000/v1/chat/completions"; TAG=sys.argv[1]
TOPICS=["bridges","printing","railways","radio","vaccines","maps","clocks","glass"]
def req(i):
    msg="Write a detailed 500-word essay about the history of "+TOPICS[i%8]+"."
    b=json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":msg}],"max_tokens":700,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    s=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(URL,b,{"Content-Type":"application/json"}))); return r["usage"]["completion_tokens"], time.time()-s
req(99)
one=[req(i) for i in range(3)]; single=sum(t for t,_ in one)/sum(d for _,d in one)
s=time.time()
with ThreadPoolExecutor(8) as ex: many=list(ex.map(req,range(8)))
agg=sum(t for t,_ in many)/(time.time()-s)
res={"model":TAG,"single_tok_s":round(single,1),"8users_total_tok_s":round(agg,1)}
print(json.dumps(res)); open("~/speed_results.jsonl","a").write(json.dumps(res)+"\n")
