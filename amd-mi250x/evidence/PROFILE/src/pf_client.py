"""PROFILE load: fresh prefills and steady decode on a vLLM server; identical prompts in every arm. python pf_client.py <port> <served> <arm> <rep> <outdir> [cells]
Streaming /v1/completions, greedy, ignore_eos. Synthetic prompts only; every prompt starts with a unique tag, so none hits the prefix cache.
Cells: pf<reps> = one prompt at a time, max_tokens 4 (about 1.5k, 4k, 7.7k and 16.1k tokens); dc<streams> = that many 7.7k prompts at once, 96 output tokens (decode window as in e2e_client.py).
One JSON row per cell with the wall-clock window, so the server-side step records (horizon_lumi_profile) can be matched to cells by time."""
import concurrent.futures as cf, json, statistics, sys, time, urllib.request
port, name, arm, rep, outdir = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
CELLS = (sys.argv[6] if len(sys.argv) > 6 else "pf115,pf330,pf640,pf1340,dc1,dc8,dc32").replace(":", ",").split(",")   # ":" also separates: sbatch --export splits on commas
B = f"http://127.0.0.1:{port}/v1"
FILL = "The harbour council recorded the broken fence before the market opened. "
def prompt(reps, tag): return f"Note {tag}: summarise what follows.\n" + FILL * reps + f"\nSay what happened next in case {tag}.\n"
def P(**k): print(json.dumps({"arm": arm, "rep": rep, **k}), flush=True)
def call(p, n):
    body = {"model": name, "prompt": p, "max_tokens": n, "temperature": 0, "seed": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t = time.time(); r = urllib.request.urlopen(urllib.request.Request(B + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1500)
    ts, usage = [], None
    for line in r:
        if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"): continue
        o = json.loads(line[6:])
        if o.get("usage"): usage = o["usage"]
        if o.get("choices"): ts.append(time.time())
    return t, ts, usage
def pf(reps, phase, k):
    t0 = time.time(); tt, ptok, wins = [], set(), []
    for j in range(k):
        t, ts, u = call(prompt(reps, f"{phase}-pf{reps}-{j}"), 4)
        tt.append(ts[0] - t); ptok.add(u["prompt_tokens"]); wins.append([round(t, 4), round(ts[0], 4), round(ts[-1], 4)])
        time.sleep(1.0)
    P(kind="cell", cell=f"pf{reps}", phase=phase, k=k, prompt_tokens_each=sorted(ptok), ttft=[round(x, 4) for x in tt], ttft_median=round(statistics.median(tt), 4), calls=wins, t_start=round(t0, 4), t_end=round(time.time(), 4))
def dc(bs, phase, n):
    ps = [prompt(640, f"{phase}-dc{bs}-{i}") for i in range(bs)]
    t = time.time()
    with cf.ThreadPoolExecutor(bs) as ex: res = list(ex.map(lambda p: call(p, n), ps))
    ttft = sorted(x[1][0] - x[0] for x in res)
    t0, t1 = max(x[1][0] for x in res), min(x[1][-1] for x in res)
    win = sum(sum(1 for q in x[1] if t0 < q <= t1) for x in res)
    P(kind="cell", cell=f"dc{bs}", phase=phase, bs=bs, n=n, prompt_tokens_each=sorted({x[2]["prompt_tokens"] for x in res}), t_start=round(t, 4), t_end=round(max(x[1][-1] for x in res), 4), dec_t0=round(t0, 4), dec_t1=round(t1, 4),
      dec_tokens=win, dec_tok_s=round(win / max(t1 - t0, 1e-6), 3), step_ms=round(1e3 * bs * (t1 - t0) / max(win, 1), 3), ttft_median=round(ttft[len(ttft) // 2], 4), ttft_max=round(ttft[-1], 4))
    time.sleep(2.0)
for phase, k, n in (("w", 2, 16), ("t", 4, 96)):          # warm pass (compiles, loads kernels), then the timed pass
    for c in CELLS:
        if c.startswith("pf"): pf(int(c[2:]), phase, k)
        else: dc(int(c[2:]), phase, n)
    time.sleep(3)
P(kind="client_done")
