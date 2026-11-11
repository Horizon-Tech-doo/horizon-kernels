"""P1 load: time to first token of fresh prompts on a vLLM server; identical prompts in every arm and repetition. python p1_client.py <port> <served> <arm> <rep> <outdir>
Streaming /v1/completions, greedy, max_tokens 8. Synthetic prompts only. Every prompt starts with a unique tag, so none hits vLLM's prefix cache and the first token waits for a whole prefill.
Cells (filler repeats, concurrent prompts): one prompt at a time at about 1.5k, 3k, 6k, 8k and 15k tokens, then 4 at once at about 3k. A prompt above the server's max_num_batched_tokens
is prefilled in chunks; only its first chunk is a fresh step. One JSON row per cell with every request's time to first token and the wall-clock window (for e2e_power.py's samples)."""
import concurrent.futures as cf, hashlib, json, statistics, sys, time, urllib.request
port, name, arm, rep, outdir = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
B = f"http://127.0.0.1:{port}/v1"
FILL = "The harbour council recorded the broken fence before the market opened. "
def prompt(reps, tag): return f"Note {tag}: summarise what follows.\n" + FILL * reps + f"\nSay what happened next in case {tag}.\n"
CELLS = [(115, 1), (235, 1), (470, 1), (640, 1), (1180, 1), (235, 4)]
N_TIMED, N_WARM = 8, 2
OUT = open(f"{outdir}/texts-{arm}-r{rep}.jsonl", "a")
def P(**k): print(json.dumps({"arm": arm, "rep": rep, **k}), flush=True)
def call(p):
    body = {"model": name, "prompt": p, "max_tokens": 8, "temperature": 0, "seed": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t = time.time(); r = urllib.request.urlopen(urllib.request.Request(B + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1500)
    first, txt, usage = None, [], None
    for line in r:
        if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"): continue
        o = json.loads(line[6:])
        if o.get("usage"): usage = o["usage"]
        if o.get("choices"):
            first = first or time.time(); txt.append(o["choices"][0].get("text") or "")
    return first - t, time.time() - t, "".join(txt), usage
def cell(reps, bs, phase, k):
    t0 = time.time(); tt, texts, ptok = [], [], set()
    for j in range(k):
        ps = [prompt(reps, f"{phase}-{reps}x{bs}-{j}-{i}") for i in range(bs)]          # the tag does not hold arm or rep: same prompts in every arm
        with cf.ThreadPoolExecutor(bs) as ex: res = list(ex.map(call, ps))
        tt += [x[0] for x in res]; texts += [x[2] for x in res]; ptok |= {x[3]["prompt_tokens"] for x in res}
        time.sleep(0.5)
    P(kind="cell", phase=phase, reps=reps, bs=bs, k=k, prompt_tokens_each=sorted(ptok), ttft=[round(x, 4) for x in tt], ttft_median=round(statistics.median(tt), 4), ttft_min=round(min(tt), 4),
      t_start=round(t0, 4), t_end=round(time.time(), 4), text_md5=hashlib.md5("\n".join(texts).encode()).hexdigest()[:12])
    OUT.write(json.dumps({"kind": "cell", "phase": phase, "reps": reps, "bs": bs, "texts": texts}) + "\n"); OUT.flush()
for reps, bs in CELLS: cell(reps, bs, "w", N_WARM)                 # warm pass, not timed: loads kernels, and the fv_ arm's HLP_VERIFY steps fall here when the bucket matches
time.sleep(5)
for reps, bs in (CELLS if rep % 2 == 0 else CELLS[::-1]): cell(reps, bs, "t", N_TIMED)
OUT.close(); P(kind="client_done")
