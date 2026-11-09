# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Ops step 0 (27-09-2026): one torch-profiler trace per window of the live config (inside an ab_arm.sh arm whose
# VLLM_EXTRA_ARGS enables --profiler-config, see run_ops_profile.sh):
#   dec_u1 / dec_u8 / dec_u32   decode, short prompts, 1 / 8 / 32 concurrent users
#   pre_16k / pre_64k           one prefill of a 16k / 64k-token prompt (max_tokens 1)
#   ctx_32k / ctx_128k          1 user decoding 96 tokens after a 32k / 128k-token prompt (trace holds prefill too)
# Prompts of the long windows are random token ids (unique per window: no prefix-cache hits); kernel time does not
# depend on the text. Writes a JSON line per window to stdout; traces land in the profiler dir in window order.
import json, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
U = "http://localhost:8001"
TOPICS = ["bridges", "printing", "railways", "radio", "vaccines", "maps", "clocks", "glass"]
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None


def post(p, b=None, t=1800):
    return urllib.request.urlopen(urllib.request.Request(U + p, json.dumps(b or {}).encode(),
                                                         {"Content-Type": "application/json"}), timeout=t).read()


def chat(i, n):
    b = {"model": "qwen3.8", "messages": [{"role": "user", "content": "Write a detailed essay about the history of "
                                           + TOPICS[i % 8] + f" (part {i})."}],
         "max_tokens": n, "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.loads(post("/v1/chat/completions", b)); return r["usage"]["completion_tokens"]


def raw(ntok, n, seed):
    rng = random.Random(seed)
    ids = [rng.randrange(1000, 150000) for _ in range(ntok)]
    b = {"model": "qwen3.8", "prompt": ids, "max_tokens": n, "temperature": 0, "ignore_eos": True}
    r = json.loads(post("/v1/completions", b)); return r["usage"]["completion_tokens"]


def window(name, fn):
    if ONLY and name not in ONLY:
        return
    post("/start_profile"); time.sleep(0.5)
    s = time.time()
    toks = fn()
    d = time.time() - s
    post("/stop_profile")
    print(json.dumps({"window": name, "tokens": toks, "s": round(d, 2)}), flush=True)
    time.sleep(60)   # trace export happens in the engine after /stop_profile


chat(0, 32)
with ThreadPoolExecutor(32) as ex: list(ex.map(lambda i: chat(i, 32), range(32)))
raw(2048, 4, 1)
for users in (1, 8, 32):
    window(f"dec_u{users}", lambda: sum(ThreadPoolExecutor(users).map(lambda i: chat(i, 96), range(users))))
window("pre_16k", lambda: raw(16384, 1, 16))
window("pre_64k", lambda: raw(65536, 1, 64))
window("ctx_32k", lambda: raw(32768, 96, 32))
window("ctx_128k", lambda: raw(131072, 96, 128))
print("PROF_OPS_DONE", flush=True)
