"""V1 load on a vLLM server, same prompts in every arm. python v1_client.py <port> <served> <rounds> <arm> <outdir>. One JSON row per measurement.
Streaming /v1/completions, greedy, ignore_eos. For each cell: rate over the window where every sequence of the batch is decoding (after the last first-token, before the first last-token), so prefill does not count, and the whole-call rate.
Each cell also records CPU fraction of the client, API server, EngineCore and workers over the call, and the engine-side decode rate from vLLM's /metrics (request_decode_time_seconds), so a client or API limit at 32 streams can be told apart from the engine rate.
det: prompts one at a time, the clean correctness compare. gsm: first 64 GSM8K questions, chat, thinking off, concurrency 8."""
import concurrent.futures as cf, hashlib, json, os, re, sys, time, urllib.request
port, name, rounds, arm, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
B = f"http://127.0.0.1:{port}/v1"
FILL = "The harbour council recorded the broken fence before the market opened. "
SHORT = [f"Write a short paragraph about topic number {i}: mountains, rivers, or cities.\n" for i in range(32)]
def long_prompt(i, reps):                 # odd rows 3/4 length; unique first line so rows do not share a prefix
    return f"Note {i}: summarise what follows.\n" + FILL * (reps if i % 2 == 0 else reps * 3 // 4) + f"\nSay what happened next in case {i}.\n"
CTX = {"short": SHORT, "2k": [long_prompt(i, 150) for i in range(32)], "8k": [long_prompt(i, 640) for i in range(32)]}
CELLS = [("short", 1), ("short", 8), ("short", 32), ("2k", 1), ("2k", 8), ("2k", 32), ("8k", 1), ("8k", 8)]
OUT = open(f"{outdir}/{arm}-texts.jsonl", "a")
def P(**k): print(json.dumps({"arm": arm, **k}), flush=True)
def call(prompt, n, lp=None):
    """lp: a list, filled with [token, {token: logprob of the top 2}] per step; det only, the timed cells do not ask for logprobs."""
    body = {"model": name, "prompt": prompt, "max_tokens": n, "temperature": 0, "seed": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    if lp is not None: body["logprobs"] = 2
    r = urllib.request.urlopen(urllib.request.Request(B + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200)
    ts, txt, usage = [], [], None
    for line in r:
        if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"): continue
        o = json.loads(line[6:])
        if o.get("usage"): usage = o["usage"]
        if o.get("choices"):
            ts.append(time.perf_counter()); txt.append(o["choices"][0].get("text") or "")
            lo = o["choices"][0].get("logprobs")
            if lp is not None and lo:
                lp.extend([a, b] for a, b in zip(lo["tokens"], lo["top_logprobs"]))
    return ts, "".join(txt), usage
JOB = os.environ.get("SLURM_JOB_ID", "")
def procs():
    """CPU seconds by role, this job's processes only."""
    out = {"api": 0.0, "engine": 0.0, "worker": 0.0}
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try:
            if f"job_{JOB}/" not in open(f"/proc/{d}/cgroup").read(): continue
            c = open(f"/proc/{d}/cmdline").read().replace("\0", " ")
            f = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
            cpu = (int(f[11]) + int(f[12])) / 100
        except OSError:
            continue
        if "Worker_TP" in c: out["worker"] += cpu
        elif "EngineCore" in c: out["engine"] += cpu
        elif "vllm serve" in c or "/vllm " in c: out["api"] += cpu
    return out
def metrics():
    """Sums from the server's Prometheus text."""
    m = {}
    try:
        for l in urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode().splitlines():
            for k in ("vllm:request_decode_time_seconds_sum", "vllm:request_decode_time_seconds_count", "vllm:generation_tokens_total"):
                if l.startswith(k): m[k] = m.get(k, 0.0) + float(l.rsplit(" ", 1)[1])
    except Exception as e:
        m["err"] = repr(e)[:80]
    return m
def cell(ctx, bs, r, pos, n=128):
    p0, m0, c0 = procs(), metrics(), time.process_time()
    t = time.perf_counter()
    with cf.ThreadPoolExecutor(bs) as ex:
        res = list(ex.map(lambda p: call(p, n), CTX[ctx][:bs]))
    dt = time.perf_counter() - t
    c1 = time.process_time(); time.sleep(0.3); p1, m1 = procs(), metrics()
    dc, ds = [m1.get(k, 0) - m0.get(k, 0) for k in ("vllm:request_decode_time_seconds_count", "vllm:request_decode_time_seconds_sum")]
    t0, t1 = max(x[0][0] for x in res), min(x[0][-1] for x in res)
    win = sum(sum(1 for q in x[0] if t0 < q <= t1) for x in res)
    tot = sum(x[2]["completion_tokens"] for x in res)
    h = hashlib.md5("\n".join(x[1] for x in res).encode()).hexdigest()[:12]
    P(kind="decode", ctx=ctx, bs=bs, round=r, pos=pos, prompt_tokens=[x[2]["prompt_tokens"] for x in res][:2], gen=tot, s=round(dt, 3), call_tok_s=round(tot / dt, 1), win_s=round(t1 - t0, 3), win_tokens=win, win_tok_s=round(win / max(t1 - t0, 1e-6), 1), text_md5=h,
      cpu_client=round((c1 - c0) / dt, 3), cpu_api=round((p1["api"] - p0["api"]) / dt, 3), cpu_engine=round((p1["engine"] - p0["engine"]) / dt, 3), cpu_workers=round((p1["worker"] - p0["worker"]) / dt, 3),
      engine_decode_tok_s=round(bs * (n - 1) / (ds / dc), 1) if dc and ds > 0 else None, metrics_n=dc, wall=round(time.time(), 2))
    OUT.write(json.dumps({"ctx": ctx, "bs": bs, "round": r, "texts": [x[1] for x in res]}) + "\n"); OUT.flush()
for ctx, bs in CELLS:                     # warm: fills prefix cache and brings up kernels, not timed
    cell(ctx, bs, -1, -1, 8)
for ctx, bs in (("short", 5), ("short", 20), ("2k", 5)):   # padded batches (n < B), forces HLD_VERIFY_PAD; not timed, text compared between arms
    cell(ctx, bs, -2, -1, 32)
P(kind="timed_start", wall=round(time.time(), 2))
for r in range(rounds):
    seq = CELLS if r % 2 == 0 else CELLS[::-1]
    for pos, (ctx, bs) in enumerate(seq):
        cell(ctx, bs, r, pos)
P(kind="timed_end", wall=round(time.time(), 2))
for i in range(8):
    lp = []; ts, txt, u = call(SHORT[i], 128, lp)
    P(kind="det", i=i, text_md5=hashlib.md5(txt.encode()).hexdigest()[:12]); OUT.write(json.dumps({"ctx": "det", "i": i, "text": txt, "lp": lp}) + "\n")
    lp = []; ts, txt, u = call(CTX["2k"][i], 64, lp)
    P(kind="det2k", i=i, text_md5=hashlib.md5(txt.encode()).hexdigest()[:12]); OUT.write(json.dumps({"ctx": "det2k", "i": i, "text": txt, "lp": lp}) + "\n"); OUT.flush()
qs = [json.loads(l) for l in open(os.environ["GSM8K_JSONL"])][:64]
def gsm(q):
    body = {"model": name, "messages": [{"role": "user", "content": q["question"] + "\nGive the final number on the last line."}], "max_tokens": 400, "temperature": 0, "seed": 0, "logprobs": True, "top_logprobs": 2, "chat_template_kwargs": {"enable_thinking": False}}
    o = json.load(urllib.request.urlopen(urllib.request.Request(B + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200))
    c = o["choices"][0]["message"]["content"] or ""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", c)
    a = nums[-1].replace(",", "").rstrip(".") if nums else ""
    try: ok = float(a) == float(q["gold"].replace(",", ""))
    except ValueError: ok = False
    lp = [[e["token"], {t["token"]: t["logprob"] for t in e["top_logprobs"]}] for e in (o["choices"][0].get("logprobs") or {}).get("content", [])]
    return q["id"], c, a, ok, o["usage"]["completion_tokens"], lp
t = time.perf_counter()
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(gsm, qs))
dt = time.perf_counter() - t
P(kind="gsm", n=len(res), correct=sum(x[3] for x in res), s=round(dt, 2), tok_s=round(sum(x[4] for x in res) / dt, 1), md5=hashlib.md5("\n".join(x[1] for x in res).encode()).hexdigest()[:12], answers_md5=hashlib.md5(",".join(x[2] for x in res).encode()).hexdigest()[:12])
for x in res:
    OUT.write(json.dumps({"ctx": "gsm", "id": x[0], "text": x[1], "answer": x[2], "ok": x[3], "lp": x[5]}) + "\n")
OUT.close(); P(kind="client_done")
