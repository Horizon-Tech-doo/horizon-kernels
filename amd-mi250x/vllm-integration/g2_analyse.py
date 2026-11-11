"""G2 analysis. python3 g2_analyse.py <raw dir> <job> [job ...] > tables.md
Per job: time to first token per cell, arm and repetition; ratio of every plugin arm to each stock baseline in the job (s_rocm = vLLM defaults, s_tri = stock with --attention-backend TRITON_ATTN), below 1 =
plugin faster, with the arms' identical-code spread as noise; card energy per timed cell the same way; plugin counters; the equivalence set (12 fixed greedy prompts, top-5 logprobs) of every arm against
the first stock repetition of both baselines and of each baseline against its own second repetition."""
import bisect, csv, glob, json, os, statistics as st, sys
RAW, JOBS = sys.argv[1], sys.argv[2:]
med = st.median
sp = lambda v: (max(v) - min(v)) / med(v) if len(v) > 1 else float("nan")
BASE = ("s_rocm", "s_tri")
def arm(T): return T.rsplit("-r", 1)[0]
def ratio_table(title, cells, order, get, unit, fmt, better, worse):
    arms = sorted({arm(T) for T in order})
    for b in BASE:
        if b not in arms: continue
        for x in arms:
            if x in BASE and not (x == "s_tri" and b == "s_rocm"): continue
            if x == b: continue
            print(f"\n**{title}: {x} / {b}** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)\n\n| Cell | {b} {unit} | {x} {unit} | ratio | {b} spread | {x} spread | verdict |\n|---|---|---|---|---|---|---|")
            for key, c in sorted(cells.items()):
                s = [get(c[T]) for T in order if arm(T) == b and T in c]; f = [get(c[T]) for T in order if arm(T) == x and T in c]
                if not s or not f: continue
                r = med(f) / med(s); nz = max(sp(s), sp(f)); pt = sorted({v for q in c.values() for v in q["prompt_tokens_each"]})
                print(f"| {'/'.join(map(str, pt))} x{key[0]} | {', '.join(fmt % v for v in s)} | {', '.join(fmt % v for v in f)} | {r:.4f} | {100 * sp(s):.2f}% | {100 * sp(f):.2f}% | {'~ inside noise' if not abs(r - 1) > nz else (better if r < 1 else worse)} |")
for job in JOBS:
    d = os.path.join(RAW, job)
    hdr = open(os.path.join(d, "header.txt")).readline().strip() if os.path.exists(os.path.join(d, "header.txt")) else ""
    runs, order = {}, []
    for f in sorted(glob.glob(os.path.join(d, "client-*.jsonl")), key=os.path.getmtime):
        rows = [json.loads(l) for l in open(f) if l.startswith("{")]
        T = os.path.basename(f)[7:-6]; runs[T] = (rows, bool(rows) and rows[-1].get("kind") == "client_done"); order.append(T)
    print(f"\n## Job {job}\n\n- {hdr}\n- Run order and result rows (timed cells of 6, client finished; equivalence rows of 12): " + ", ".join(f"{T} {sum(1 for r in runs[T][0] if r.get('kind') == 'cell' and r['phase'] == 't')} {runs[T][1]} {sum(1 for _ in open(os.path.join(d, 'eq-' + T + '.jsonl'))) if os.path.exists(os.path.join(d, 'eq-' + T + '.jsonl')) else 0}" for T in order))
    cells = {}
    for T in order:
        for r in runs[T][0]:
            if r.get("kind") == "cell" and r["phase"] == "t": cells.setdefault((r["bs"], r["reps"]), {})[T] = r
    print("\n**Time to first token, s: median of the cell's timed requests [min-max]**\n\n| Cell (prompt tokens x streams) | " + " | ".join(order) + " |\n|---|" + "---|" * len(order))
    for key, c in sorted(cells.items()):
        pt = sorted({x for r in c.values() for x in r["prompt_tokens_each"]})
        print(f"| {'/'.join(map(str, pt))} x{key[0]} | " + " | ".join(f"{c[T]['ttft_median']:.4f} [{min(c[T]['ttft']):.4f}-{max(c[T]['ttft']):.4f}]" if T in c else "-" for T in order) + " |")
    ratio_table("Time to first token", cells, order, lambda r: r["ttft_median"], "s", "%.4f", "faster", "slower")
    # counters
    print("\n**Plugin counters per worker** (GEMM plugin: linear calls; prefill-attention plugin: attention-layer calls, 16 per step)\n\n| Arm | plugin | worker | hit | rows or tokens per hit | padded rows / pad share | left to vLLM | errors | kernels or buckets used |\n|---|---|---|---|---|---|---|---|---|")
    for T in order:
        for f in sorted(glob.glob(os.path.join(d, "ctr-" + T, "hlg-*.json"))):
            try: c = json.load(open(f))
            except ValueError: continue
            if c["hit"] or c["stock_m"]:
                print(f"| {T} | gemm | {c['pid']} | {c['hit']} | {c['rows'] / max(c['hit'], 1):.0f} | {c['pad_rows']} | rows {c['stock_m']} ({' '.join(f'{k}:{v}' for k, v in sorted(c['stock_m_rows'].items(), key=lambda kv: -kv[1])[:6])}), shape {c['stock_shape']}, capture {c['stock_capture']} | {c['error']} | {len(c['kernel_use'])} kernels: {' '.join(f'{k}:{v}' for k, v in sorted(c['kernel_use'].items(), key=lambda kv: -kv[1])[:8])} |")
        for f in sorted(glob.glob(os.path.join(d, "ctr-" + T, "hlp-*.json"))):
            try: c = json.load(open(f))
            except ValueError: continue
            if c["hit"] or c["stock_context"] or c["stock_mixed"]:
                print(f"| {T} | prefill attention | {c['pid']} | {c['hit']} | {c['tokens'] / max(c['hit'], 1):.0f} | {100 * c['pad_tokens'] / max(c['tokens'], 1):.2f}% | context {c['stock_context']}, mixed {c['stock_mixed']}, short {c['stock_short']}, long {c['stock_long']}, miss {c['stock_miss']}, guard {c['stock_guard']}, shape {c['stock_shape']} | {c['error']} | {' '.join(f'{k}:{v}' for k, v in sorted(c['bucket_use'].items(), key=lambda kv: (kv[0].startswith('ctx'), int(kv[0].replace('ctx', '')))))} |")
    # energy
    try:
        meta = json.load(open(os.path.join(d, "power-meta.json"))); rows = list(csv.DictReader(open(os.path.join(d, "power.csv")))); t = [float(r["t"]) for r in rows]
        allc = [c for T in order for c in runs[T][0] if c.get("kind") == "cell" and c["phase"] == "t"]
        a0, b0 = min(c["t_start"] for c in allc), max(c["t_end"] for c in allc)
        mean = [st.mean(float(r["a%d_W" % i]) for r, tt in zip(rows, t) if a0 <= tt <= b0 and r["a%d_W" % i]) for i in range(4)]
        card = max(range(4), key=lambda i: mean[i])
        pts = sorted({(int(r["a%d_us" % card]) / 1e6, int(r["a%d_J" % card])) for r in rows if r["a%d_us" % card]}); TT, JJ = [p[0] for p in pts], [p[1] for p in pts]
        def at(x):
            i = bisect.bisect_left(TT, x)
            if i <= 0: return JJ[0]
            if i >= len(TT): return JJ[-1]
            return JJ[i - 1] + (JJ[i] - JJ[i - 1]) * (x - TT[i - 1]) / (TT[i] - TT[i - 1])
        E = {}
        for T in order:
            for c in runs[T][0]:
                if c.get("kind") == "cell" and c["phase"] == "t":
                    E.setdefault((c["bs"], c["reps"]), {})[T] = dict(c, J=at(c["t_end"]) - at(c["t_start"]), W=(at(c["t_end"]) - at(c["t_start"])) / (c["t_end"] - c["t_start"]))
        print(f"\n**Energy per timed cell [observed]: Cray accel{card} energy counter (mean W of the four counters over all timed cells: {', '.join('%.0f' % m for m in mean)}; GCDs visible to the sampler {meta['gcds']}). J per cell (mean W)**\n\n| Cell | " + " | ".join(order) + " |\n|---|" + "---|" * len(order))
        for key, v in sorted(E.items()):
            pt = sorted({x for r in v.values() for x in r["prompt_tokens_each"]})
            print(f"| {'/'.join(map(str, pt))} x{key[0]} | " + " | ".join(f"{v[T]['J']:.0f} ({v[T]['W']:.0f} W)" if T in v else "-" for T in order) + " |")
        ratio_table("Energy per cell", E, order, lambda r: r["J"], "J", "%.0f", "less energy", "more energy")
    except Exception as e:
        print("\npower: not analysed (" + repr(e)[:160] + ")")
    # equivalence
    Q = {}
    for T in order:
        f = os.path.join(d, f"eq-{T}.jsonl")
        if os.path.exists(f): Q[T] = [json.loads(l) for l in open(f)]
    def cmp(a, b):
        """(identical texts, compared prompts, max |dlogprob| of the chosen token over the common prefix, [(kind, i, first differing token index, top-2 margin of b there)])"""
        same = 0; mx = 0.0; div = []
        for x, y in zip(a, b):
            n = 0
            for tx, ty, lx, ly in zip(x["tokens"], y["tokens"], x["token_logprobs"], y["token_logprobs"]):
                if tx != ty: break
                mx = max(mx, abs(lx - ly)); n += 1
            if x["tokens"] == y["tokens"]: same += 1
            else:
                top = sorted(y["top"][n].values(), reverse=True) if n < len(y["top"]) else []
                # at the first differing position: how far the reference's chosen token moved in this run (its logprob there minus the reference's), when it is in this run's top 5
                moved = round(x["top"][n][y["tokens"][n]] - y["token_logprobs"][n], 3) if n < len(x["top"]) and y["tokens"][n] in x["top"][n] else None
                div.append((x["kind"], x["i"], n, round(top[0] - top[1], 3) if len(top) > 1 else None, moved))
        return same, min(len(a), len(b)), mx, div
    if Q:
        print("\n**Equivalence set [observed]: 12 fixed greedy prompts x 48 tokens. Identical token sequences of 12; largest |difference| of the chosen token's logprob over the tokens before the first difference (nats); differing prompts as (kind, index, first differing token, top-2 margin of the reference there, change of the reference token's logprob there in this run)**\n\n| Run | against | identical | max abs logprob difference | differing prompts |\n|---|---|---|---|---|")
        for b in BASE:
            ref = next((T for T in order if arm(T) == b and T in Q), None)
            if ref is None: continue
            for T in order:
                if T == ref or T not in Q: continue
                if arm(T) in BASE and arm(T) != b and not (arm(T) == "s_tri" and b == "s_rocm"): continue
                s, n, mx, div = cmp(Q[T], Q[ref])
                print(f"| {T} | {ref} | {s} of {n} | {mx:.4f} | {', '.join(str(x) for x in div) or '-'} |")
