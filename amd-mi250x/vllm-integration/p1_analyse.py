"""P1 analysis. python3 p1_analyse.py <raw dir> <job> [job ...] > tables.md
Per job: time to first token per cell, arm and repetition (median of the cell's timed requests and all values), plugin over stock ratios (f / s, below 1 = plugin faster),
the identical-code spread of each arm (the job's noise bound), the plugin's counters per worker (calls are per attention layer: 16 per step) and the HLP_VERIFY differences."""
import glob, json, os, re, statistics as st, sys
RAW, JOBS = sys.argv[1], sys.argv[2:]
med = st.median
for job in JOBS:
    d = os.path.join(RAW, job)
    hdr = open(os.path.join(d, "header.txt")).readline().strip() if os.path.exists(os.path.join(d, "header.txt")) else ""
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "client-*.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.startswith("{")]
        T = os.path.basename(f)[7:-6]
        runs[T] = (rows, bool(rows) and rows[-1].get("kind") == "client_done")
    print(f"\n## Job {job}\n\n- {hdr}\n- Result rows (timed cells of 6, client finished): " + ", ".join(f"{T} {sum(1 for r in rows if r.get('kind') == 'cell' and r['phase'] == 't')} {ok}" for T, (rows, ok) in runs.items()))
    cells = {}
    for T, (rows, ok) in runs.items():
        for r in rows:
            if r.get("kind") == "cell" and r["phase"] == "t": cells.setdefault((r["reps"], r["bs"]), {})[T] = r
    Ts = list(runs)
    print("\n**Time to first token, s: median of the cell's timed requests [min-max of them]** (8 requests per cell at 1 stream, 32 at 4)\n\n| Cell (prompt tokens x streams) | " + " | ".join(Ts) + " |\n|---|" + "---|" * len(Ts))
    for (reps, bs), c in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        pt = sorted({x for r in c.values() for x in r["prompt_tokens_each"]})
        print(f"| {'/'.join(map(str, pt))} x{bs} | " + " | ".join(f"{c[T]['ttft_median']:.4f} [{min(c[T]['ttft']):.4f}-{max(c[T]['ttft']):.4f}]" if T in c else "-" for T in Ts) + " |")
    arms = sorted({T.rsplit("-r", 1)[0] for T in Ts})
    if "s_rocm" in arms and "f_rocm" in arms:
        print("\n**f_rocm / s_rocm, ratio of the arms' medians over repetitions (below 1: plugin faster); spread = (max - min) / median of one arm's repetition medians; `~` when |ratio - 1| <= the larger spread**\n\n| Cell | s reps | f reps | f / s | s spread | f spread | verdict |\n|---|---|---|---|---|---|---|")
        for (reps, bs), c in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            s = [c[T]["ttft_median"] for T in Ts if T.startswith("s_rocm-") and T in c]; f = [c[T]["ttft_median"] for T in Ts if T.startswith("f_rocm-") and T in c]
            if not s or not f: continue
            r = med(f) / med(s); sp = lambda v: (max(v) - min(v)) / med(v) if len(v) > 1 else float("nan")
            nz = max(sp(s), sp(f))
            pt = sorted({x for q in c.values() for x in q["prompt_tokens_each"]})
            print(f"| {'/'.join(map(str, pt))} x{bs} | {', '.join('%.4f' % x for x in s)} | {', '.join('%.4f' % x for x in f)} | {r:.4f} | {100 * sp(s):.2f}% | {100 * sp(f):.2f}% | {'~ inside noise' if abs(r - 1) <= nz else ('plugin faster' if r < 1 else 'plugin slower')} |")
    print("\n**Plugin counters per worker (attention-layer calls; 16 per step)**\n\n| Arm | worker | hit | steps taken (hit / 16) | tokens / hit | pad share | stock_context | stock_mixed | stock_short | stock_long | stock_miss | stock_guard | stock_shape | stock_capture | error | buckets used (bucket: calls) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for T in Ts:
        for f in sorted(glob.glob(os.path.join(d, "ctr-" + T, "hlp-*.json"))):
            c = json.load(open(f))
            if not (c["hit"] or c["stock_context"] or c["stock_mixed"] or c["stock_short"] or c["stock_long"]): continue
            print(f"| {T} | {c['pid']} | {c['hit']} | {c['hit'] / 16:g} | {c['tokens'] / max(c['hit'], 1):.0f} | {100 * c['pad_tokens'] / max(c['tokens'], 1):.2f}% | {c['stock_context']} | {c['stock_mixed']} | {c['stock_short']} | {c['stock_long']} | {c['stock_miss']} | {c['stock_guard']} | {c['stock_shape']} | {c['stock_capture']} | {c['error']} | {' '.join(f'{k}:{v}' for k, v in sorted(c['bucket_use'].items(), key=lambda kv: int(kv[0])))} |")
    for T in Ts:
        f = os.path.join(d, f"serve-{T}.log")
        if not T.startswith("fv") or not os.path.exists(f): continue
        V = {}
        for l in open(f, errors="replace"):
            if '"ev": "verify"' in l:
                o = json.loads(l[l.index("{"):]); V.setdefault(tuple(o["buckets"]), []).append(o)
        print(f"\n**HLP_VERIFY in {T}: ours against vLLM's kernel on the same step inputs, per bucket (both workers, all 16 layers)**\n\n| bucket | lengths | compared calls | rel (Frobenius) min-max | max abs difference | largest |reference| | NaN |\n|---|---|---|---|---|---|---|")
        for b, v in sorted(V.items()):
            print(f"| {b} | {sorted({tuple(o['lens']) for o in v})} | {len(v)} | {min(o['rel'] for o in v):.2e}-{max(o['rel'] for o in v):.2e} | {max(o['max_abs'] for o in v):g} | {max(o['rmax'] for o in v):g} | {sum(o['nan'] for o in v)} |")
