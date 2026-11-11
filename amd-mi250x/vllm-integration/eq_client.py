"""Output equivalence set: fixed synthetic prompts, greedy (temperature 0, seed 0), one request at a time, top-5 logprobs per generated token. python eq_client.py <port> <served> <arm> <rep> <outdir>
12 prompts: 4 short (about 25 tokens), 3 of about 2.9k, 3 of about 7.7k and 2 of about 14.2k tokens; 48 output tokens each, ignore_eos. Identical in every arm and repetition.
One JSON row per prompt into <outdir>/eq-<arm>-r<rep>.jsonl: text, tokens, the chosen token's logprob per position and the top-5 map. Compared afterwards by g2_analyse.py."""
import json, sys, urllib.request
port, name, arm, rep, outdir = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
B = f"http://127.0.0.1:{port}/v1"
FILL = "The harbour council recorded the broken fence before the market opened. "
def short(i): return f"Equivalence run. Write a short paragraph about topic number {i}: mountains, rivers, or cities.\n"
def long(i, reps): return f"Note eq-{reps}-{i}: summarise what follows.\n" + FILL * reps + f"\nSay what happened next in case eq-{reps}-{i}.\n"
PROMPTS = [("short", i, short(i)) for i in range(4)] + [("2.9k", i, long(i, 235)) for i in range(3)] + [("7.7k", i, long(i, 640)) for i in range(3)] + [("14.2k", i, long(i, 1180)) for i in range(2)]
out = open(f"{outdir}/eq-{arm}-r{rep}.jsonl", "w")
for kind, i, p in PROMPTS:
    body = {"model": name, "prompt": p, "max_tokens": 48, "temperature": 0, "seed": 0, "ignore_eos": True, "logprobs": 5}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(B + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
    c = r["choices"][0]; lp = c["logprobs"]
    out.write(json.dumps({"arm": arm, "rep": rep, "kind": kind, "i": i, "prompt_tokens": r["usage"]["prompt_tokens"], "text": c["text"], "tokens": lp["tokens"], "token_logprobs": lp["token_logprobs"], "top": lp["top_logprobs"]}) + "\n"); out.flush()
print(json.dumps({"kind": "eq_done", "n": len(PROMPTS)}))
