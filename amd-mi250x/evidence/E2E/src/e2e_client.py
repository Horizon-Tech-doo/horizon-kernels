"""E2E load on a vLLM server; identical prompts in every arm and repetition. python e2e_client.py <port> <served> <arm> <rep> <outdir>
Streaming /v1/completions, greedy (temperature 0, seed 0), ignore_eos, fixed max_tokens, so output tokens per cell are fixed.
Synthetic prompts only. A phase tag at the start of every prompt (w = warm pass, t = timed pass) makes the timed prompts miss vLLM's prefix cache,
so time to first token and the whole-call window include a real prefill. One JSON row per cell with wall-clock window edges; energy is
integrated afterwards from the host sampler's CSV over those edges (e2e_analyse.py).
Windows: call = first request sent .. last token of the last stream; decode = after the last stream's first token .. before the first stream's last token."""
import concurrent.futures as cf, hashlib, json, sys, time, urllib.request
port, name, arm, rep, outdir = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
B = f"http://127.0.0.1:{port}/v1"
FILL = "The harbour council recorded the broken fence before the market opened. "
def short(i, tag): return f"Run {tag}. Write a short paragraph about topic number {i}: mountains, rivers, or cities.\n"
def long8k(i, tag): return f"Note {i}, run {tag}: summarise what follows.\n" + FILL * 640 + f"\nSay what happened next in case {i}.\n"
GEN = {"short": short, "8k": long8k}
CELLS = [("short", 1, 384), ("short", 8, 256), ("short", 32, 256), ("8k", 1, 384), ("8k", 8, 256), ("8k", 32, 256)]
OUT = open(f"{outdir}/texts-{arm}-r{rep}.jsonl", "a")
def P(**k): print(json.dumps({"arm": arm, "rep": rep, **k}), flush=True)
def call(prompt, n, lp=None):
    body = {"model": name, "prompt": prompt, "max_tokens": n, "temperature": 0, "seed": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    if lp is not None: body["logprobs"] = 2
    t_send = time.time()
    r = urllib.request.urlopen(urllib.request.Request(B + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1500)
    ts, txt, usage = [], [], None
    for line in r:
        if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"): continue
        o = json.loads(line[6:])
        if o.get("usage"): usage = o["usage"]
        if o.get("choices"):
            ts.append(time.time()); txt.append(o["choices"][0].get("text") or "")
            lo = o["choices"][0].get("logprobs")
            if lp is not None and lo:
                lp.extend([a, b] for a, b in zip(lo["tokens"], lo["top_logprobs"]))
    return t_send, ts, "".join(txt), usage
KEYS = ("vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")
def metrics():
    m = {}
    try:
        for l in urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode().splitlines():
            for k in KEYS:
                if l.startswith(k + "{") or l.startswith(k + " "): m[k] = m.get(k, 0.0) + float(l.rsplit(" ", 1)[1])
    except Exception as e:
        m["err"] = repr(e)[:80]
    return m
def cell(ctx, bs, n, phase, pos):
    prompts = [GEN[ctx](i, f"{phase}{ctx}x{bs}") for i in range(bs)]
    m0 = metrics(); t = time.time()
    with cf.ThreadPoolExecutor(bs) as ex:
        res = list(ex.map(lambda p: call(p, n), prompts))
    t_end = time.time(); time.sleep(0.3); m1 = metrics()
    ttft = sorted(x[1][0] - x[0] for x in res)
    t0, t1 = max(x[1][0] for x in res), min(x[1][-1] for x in res)
    win = sum(sum(1 for q in x[1] if t0 < q <= t1) for x in res)
    out = sum(x[3]["completion_tokens"] for x in res); pin = sum(x[3]["prompt_tokens"] for x in res)
    P(kind="cell", phase=phase, ctx=ctx, bs=bs, n=n, pos=pos, prompt_tokens=pin, prompt_tokens_each=sorted({x[3]["prompt_tokens"] for x in res}), out_tokens=out, total_tokens=pin + out,
      chunks=sum(len(x[1]) for x in res), t_start=round(t, 4), t_end=round(max(x[1][-1] for x in res), 4), call_s=round(max(x[1][-1] for x in res) - t, 4), call_tok_s=round(out / (max(x[1][-1] for x in res) - t), 3),
      dec_t0=round(t0, 4), dec_t1=round(t1, 4), dec_s=round(t1 - t0, 4), dec_tokens=win, dec_tok_s=round(win / max(t1 - t0, 1e-6), 3),
      ttft_median=round(ttft[len(ttft) // 2], 4), ttft_min=round(ttft[0], 4), ttft_max=round(ttft[-1], 4), ttft_mean=round(sum(ttft) / len(ttft), 4),
      text_md5=hashlib.md5("\n".join(x[2] for x in res).encode()).hexdigest()[:12], metrics_delta={k.split(":")[1]: m1.get(k, 0) - m0.get(k, 0) for k in KEYS}, client_end=round(t_end, 4))
    OUT.write(json.dumps({"kind": "cell", "phase": phase, "ctx": ctx, "bs": bs, "texts": [x[2] for x in res]}) + "\n"); OUT.flush()
for pos, (ctx, bs, n) in enumerate(CELLS):          # warm pass, not timed: brings up kernels and graphs on every shape
    cell(ctx, bs, 16, "w", pos)
time.sleep(5); t = time.time(); time.sleep(25)       # server loaded, no requests
P(kind="idle_loaded", t_start=round(t, 4), t_end=round(time.time(), 4))
seq = list(enumerate(CELLS)) if rep % 2 == 0 else list(enumerate(CELLS))[::-1]
for pos, (ctx, bs, n) in seq:
    time.sleep(3)
    cell(ctx, bs, n, "t", pos)
for i in range(8):                                    # fixed greedy set, one request at a time, logprobs: the equivalence check between arms
    lp = []; _, ts, txt, u = call(short(i, "det"), 128, lp)
    P(kind="det", ctx="short", i=i, text_md5=hashlib.md5(txt.encode()).hexdigest()[:12]); OUT.write(json.dumps({"kind": "det", "ctx": "short", "i": i, "text": txt, "lp": lp}) + "\n")
for i in range(4):
    lp = []; _, ts, txt, u = call(long8k(i, "det"), 64, lp)
    P(kind="det", ctx="8k", i=i, text_md5=hashlib.md5(txt.encode()).hexdigest()[:12]); OUT.write(json.dumps({"kind": "det", "ctx": "8k", "i": i, "text": txt, "lp": lp}) + "\n")
OUT.close(); P(kind="client_done")
