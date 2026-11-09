# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json,sys,urllib.request,time
U="http://localhost:11500"
def post(p,b=None): return urllib.request.urlopen(urllib.request.Request(U+p,json.dumps(b or {}).encode(),{"Content-Type":"application/json"}),timeout=600).read()
def chat(msg,n): 
    b={"model":"dev","messages":[{"role":"user","content":msg}],"max_tokens":n,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    s=time.time(); r=json.loads(post("/v1/chat/completions",b)); return r["usage"]["completion_tokens"],time.time()-s
chat("warm up",32)
MSG=sys.argv[1] if len(sys.argv)>1 else "Write a detailed essay about the history of bridges."
post("/start_profile"); t=chat(MSG,160); post("/stop_profile"); print("profiled",t)
