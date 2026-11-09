#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""bench_longctx.py TAG OUT.jsonl [BASE_URL]  (24-10-2026, vLLM 0.32 port)
Long-context decode and prefill speed through the public endpoint, one request at a time per row plus one
concurrent row. Prompts are real source code from an installed Python tree (like a coding agent's context), each
request unique in its first block so prefix caching and the CPU KV tier cannot serve it.
Per request: TTFT (prefill), decode tok/s = (completion_tokens - 1) / (t_last - t_first), prompt/completion tokens.
Row: ctx = target prompt tokens. Appends one JSON line per row to OUT."""
import json, os, random, sys, threading, time, urllib.request, zlib

TAG, OUT = sys.argv[1], sys.argv[2]
BASE = (sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000").rstrip("/")
MODEL = "Qwen3.8-27B-NVFP4"
CTXS = [int(x) for x in os.environ.get("CTXS", "4096,16384,32768,65536,131072").split(",")]
MAXTOK = int(os.environ.get("MAXTOK", "200"))
CONC = int(os.environ.get("CONC", "4"))       # concurrent row at 32k
TREE = os.environ.get("TREE", "$HK_VENV/lib/python3.12/site-packages/vllm")


def corpus():
    files = []
    for root, _, fs in os.walk(TREE):
        for f in fs:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                try:
                    if os.path.getsize(p) > 3000: files.append(p)
                except OSError:
                    pass
    files.sort()
    return files


FILES = corpus()


def make_prompt(target_tokens, seed):
    rng = random.Random(seed)
    order = FILES[:]; rng.shuffle(order)
    parts, chars = [f"Session {seed}-{rng.random():.9f}\n"], 0
    need = int(target_tokens * 3.6)          # ~3.6 chars per token for code; checked against usage.prompt_tokens
    for p in order:
        t = open(p, encoding="utf-8", errors="ignore").read()
        parts.append(f"\n# ---- file {os.path.relpath(p, TREE)} ----\n" + t); chars += len(t)
        if chars >= need: break
    text = "".join(parts)[:need]
    return text + "\n\n---\nExplain in detail what the code above does, file by file."


def one(ctx, seed):
    body = {"model": MODEL, "messages": [{"role": "user", "content": make_prompt(ctx, seed)}], "max_tokens": MAXTOK,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = tl = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"): continue
            d = line[5:].strip()
            if d == b"[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            ch = j.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                now = time.time()
                if tf is None: tf = now
                tl = now
    if tf is None or usage is None: return {"error": "no tokens"}
    n = usage["completion_tokens"]
    return {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": n, "ttft_s": round(tf - t0, 3),
            "decode_tps": round((n - 1) / (tl - tf), 2) if tl > tf and n > 1 else None, "total_s": round(tl - t0, 2)}


def row(ctx, users, rep):
    res = [None] * users
    def w(i):
        try: res[i] = one(ctx, seed=zlib.crc32(f"{ctx}-{rep}-{i}".encode()))
        except Exception as e: res[i] = {"error": repr(e)[:200]}
    th = [threading.Thread(target=w, args=(i,)) for i in range(users)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
    rec = {"tag": TAG, "ctx": ctx, "users": users, "rep": rep, "wall_s": round(time.time() - t0, 2), "reqs": res}
    ok = [r for r in res if r and "error" not in r]
    if ok:
        rec["prompt_tokens"] = ok[0]["prompt_tokens"]
        rec["ttft_s"] = round(sum(r["ttft_s"] for r in ok) / len(ok), 3)
        rec["decode_tps_per_user"] = round(sum(r["decode_tps"] or 0 for r in ok) / len(ok), 2)
        rec["decode_tps_total"] = round(sum(r["decode_tps"] or 0 for r in ok), 2)
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print(f"{TAG} ctx={ctx} users={users} rep={rep}: prompt {rec.get('prompt_tokens')} ttft {rec.get('ttft_s')}s "
          f"decode/user {rec.get('decode_tps_per_user')} total {rec.get('decode_tps_total')} tok/s", flush=True)


if __name__ == "__main__":
    print(f"{TAG}: {len(FILES)} source files in corpus", flush=True)
    one(1024, seed=1)                                  # warm-up (not recorded)
    for ctx in CTXS:
        for rep in range(2):
            row(ctx, 1, rep)
    if CONC > 1:
        for rep in range(2):
            row(32768, CONC, rep)
