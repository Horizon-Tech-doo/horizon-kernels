"""Track 5: load on a vLLM server. python vllm_mn_client.py <port> <served name> <rounds>. One JSON row per measurement.
Fixed prompts, greedy, ignore_eos, so tokens per call are equal across arms; hashes show whether the outputs match across arms.
Prefill uses a unique first line each trial, so the prefix cache cannot serve it."""
import concurrent.futures as cf
import hashlib
import json
import sys
import time
import urllib.request

port, name, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
U = f"http://127.0.0.1:{port}/v1/completions"
PROMPTS = [f"Write a short paragraph about topic number {i}: mountains, rivers, or cities.\n" for i in range(32)]
FILL = "The harbour council recorded the broken fence before the market opened. "


def call(prompt, max_tokens):
    body = {"model": name, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "seed": 0, "ignore_eos": True}
    t = time.perf_counter()
    r = urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900)
    o = json.load(r)
    return time.perf_counter() - t, o["usage"], o["choices"][0]["text"]


def decode(bs, tokens, r, pos):
    t = time.perf_counter()
    with cf.ThreadPoolExecutor(bs) as ex:
        res = list(ex.map(lambda p: call(p, tokens), PROMPTS[:bs]))
    dt = time.perf_counter() - t
    n = sum(x[1]["completion_tokens"] for x in res)
    h = hashlib.md5("\n".join(x[2] for x in res).encode()).hexdigest()[:12]
    print(json.dumps({"kind": "decode", "round": r, "pos": pos, "bs": bs, "gen_tokens": n, "s": round(dt, 3), "tok_per_s": round(n / dt, 1), "text_md5": h}), flush=True)


call(PROMPTS[0], 16)                                         # warm-up, not timed
for r in range(rounds):
    seq = [(1, 8, 32), (32, 8, 1), (8, 32, 1)][r % 3]
    for pos, bs in enumerate(seq):
        decode(bs, 128, r, pos)
    for reps in (600, 2400):                                 # the token count comes back in prompt_tokens
        p = f"Trial {r} repeats {reps}.\n" + FILL * reps
        dt, u, _ = call(p, 1)
        print(json.dumps({"kind": "prefill", "round": r, "prompt_tokens": u["prompt_tokens"], "s": round(dt, 3), "tok_per_s": round(u["prompt_tokens"] / dt, 1)}), flush=True)
