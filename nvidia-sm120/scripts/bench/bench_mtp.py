# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json,time,sys,urllib.request
from concurrent.futures import ThreadPoolExecutor
URL="http://localhost:8000/v1/chat/completions"; TAG=sys.argv[1]
P={"prose":["Write a detailed 400-word essay about the history of railways.","Write a 400-word short story about a lighthouse keeper."],
   "maths":["A shop sells pens at 3 for 2.40 EUR and notebooks at 4.50 EUR each. Anna buys 12 pens and 5 notebooks and pays with a 50 EUR note. Solve step by step: how much change does she get?",
            "Solve step by step: a train leaves at 08:15 travelling 84 km/h; a second leaves the same station at 09:00 at 112 km/h on the same track. When and where does the second catch the first?",
            "Solve step by step: find all real x such that 2x^2 - 7x + 3 = 0, then compute the sum of their squares."],
   "code":["Write a Python function that parses an ISO-8601 date string and returns the weekday name, with type hints, docstring and 5 pytest tests.",
           "Write a TypeScript Express route that validates a JSON body with zod (name, email, age>=18) and returns 201 or 400 with errors."]}
def req(msg,mt=600):
    b=json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":msg}],"max_tokens":mt,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    s=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(URL,b,{"Content-Type":"application/json"}),timeout=600)); return r["usage"]["completion_tokens"], time.time()-s
req("hi",20)
res={"spec_tokens":TAG}
for k,ps in P.items():
    rs=[req(p) for p in ps]; res[k]=round(sum(t for t,_ in rs)/sum(d for _,d in rs),1)
allp=[p for ps in P.values() for p in ps]
s=time.time()
with ThreadPoolExecutor(8) as ex: many=list(ex.map(req,(allp*2)[:8]))
res["8users_total"]=round(sum(t for t,_ in many)/(time.time()-s),1)
print(json.dumps(res),flush=True); open("~/mtp_sweep.jsonl","a").write(json.dumps(res)+"\n")
