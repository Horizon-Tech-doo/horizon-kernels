# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json,urllib.request
Q="Find all integer solutions (x,y) with x,y between -20 and 20 of x^2 - 3xy + 2y^2 = 12. Explain briefly."
SYS="You are a careful assistant. "*3000
def ask(e):
    b={"model":"qwen3.8","messages":[{"role":"system","content":SYS},{"role":"user","content":Q}],"max_tokens":50,"reasoning_effort":e}
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"}),timeout=600))
    u=d["usage"]; c=(u.get("prompt_tokens_details") or {}).get("cached_tokens"); print(e, u["completion_tokens"], "output tokens | prompt", u["prompt_tokens"], "cached", c, flush=True)
for e in ["low","low"]: ask(e)
