"""G1 analysis. python3 g1_analyse.py <raw dir> <job> [job ...] > tables.md
Per job: time to first token per cell, arm and repetition; x / s_rocm ratios for x in g_rocm (GEMM plugin) and fg_rocm (GEMM + prefill-attention plugins), below 1 = plugin faster, with each arm's
identical-code spread as the job's noise; the decode step of every arm (pf_client dc1, dc8); the GEMM plugin's counters per worker; the HLG_VERIFY differences; card power per arm (hwmon, timed cells)."""
import csv, glob, json, os, statistics as st, sys
RAW, JOBS = sys.argv[1], sys.argv[2:]
med = st.median
sp = lambda v: (max(v) - min(v)) / med(v) if len(v) > 1 else float("nan")
for job in JOBS:
    d = os.path.join(RAW, job)
    hdr = open(os.path.join(d, "header.txt")).readline().strip() if os.path.exists(os.path.join(d, "header.txt")) else ""
    runs, order = {}, []
    for f in sorted(glob.glob(os.path.join(d, "client-*.jsonl")), key=os.path.getmtime):
        rows = [json.loads(l) for l in open(f) if l.startswith("{")]
        T = os.path.basename(f)[7:-6]; runs[T] = (rows, bool(rows) and rows[-1].get("kind") == "client_done"); order.append(T)
    print(f"\n## Job {job}\n\n- {hdr}\n- Run order and result rows (timed cells of 6, client finished): " + ", ".join(f"{T} {sum(1 for r in runs[T][0] if r.get('kind') == 'cell' and r['phase'] == 't')} {runs[T][1]}" for T in order))
    cells = {}
    for T in order:
        for r in runs[T][0]:
            if r.get("kind") == "cell" and r["phase"] == "t": cells.setdefault((r["bs"], r["reps"]), {})[T] = r
    print("\n**Time to first token, s: median of the cell's timed requests [min-max]**\n\n| Cell (prompt tokens x streams) | " + " | ".join(order) + " |\n|---|" + "---|" * len(order))
    for key, c in sorted(cells.items()):
        pt = sorted({x for r in c.values() for x in r["prompt_tokens_each"]})
        print(f"| {'/'.join(map(str, pt))} x{key[0]} | " + " | ".join(f"{c[T]['ttft_median']:.4f} [{min(c[T]['ttft']):.4f}-{max(c[T]['ttft']):.4f}]" if T in c else "-" for T in order) + " |")
    arms = sorted({T.rsplit("-r", 1)[0] for T in order})
    for x in ("g_rocm", "fg_rocm", "f_rocm"):
        if x not in arms or "s_rocm" not in arms: continue
        print(f"\n**{x} / s_rocm: ratio of the arms' medians over repetitions (below 1: plugin faster); spread = (max - min) / median of one arm's repetition medians; verdict needs |ratio - 1| above the larger spread**\n\n| Cell | s reps | {x} reps | ratio | s spread | {x} spread | verdict |\n|---|---|---|---|---|---|---|")
        for key, c in sorted(cells.items()):
            s = [c[T]["ttft_median"] for T in order if T.startswith("s_rocm-") and T in c]; f = [c[T]["ttft_median"] for T in order if T.startswith(x + "-") and T in c]
            if not s or not f: continue
            r = med(f) / med(s); nz = max(sp(s), sp(f)); pt = sorted({v for q in c.values() for v in q["prompt_tokens_each"]})
            print(f"| {'/'.join(map(str, pt))} x{key[0]} | {', '.join('%.4f' % v for v in s)} | {', '.join('%.4f' % v for v in f)} | {r:.4f} | {100 * sp(s):.2f}% | {100 * sp(f):.2f}% | {'~ inside noise' if not abs(r - 1) > nz else ('plugin faster' if r < 1 else 'plugin slower')} |")
    D = {}
    for T in order:
        f = os.path.join(d, f"decode-{T}.jsonl")
        if os.path.exists(f):
            for l in open(f):
                if l.startswith("{"):
                    r = json.loads(l)
                    if r.get("kind") == "cell" and r["phase"] == "t": D.setdefault(r["cell"], {})[T] = r["step_ms"]
    if D:
        print("\n**Decode step, ms (8k contexts, 96 tokens, decode window; default attention backend)**\n\n| Cell | " + " | ".join(order) + " |\n|---|" + "---|" * len(order))
        for c, v in sorted(D.items()): print(f"| {c} | " + " | ".join(f"{v[T]:.2f}" if T in v else "-" for T in order) + " |")
    print("\n**GEMM plugin counters per worker**\n\n| Arm | worker | hit (linear calls taken) | rows / hit | pad rows | stock_m (rows not compiled) | stock_shape | stock_capture | error | kernels used (M x N x K: calls) | rows left to vLLM (rows: calls, first 8) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for T in order:
        for f in sorted(glob.glob(os.path.join(d, "ctr-" + T, "hlg-*.json"))):
            c = json.load(open(f))
            if not (c["hit"] or c["stock_m"]): continue
            print(f"| {T} | {c['pid']} | {c['hit']} | {c['rows'] / max(c['hit'], 1):.0f} | {c['pad_rows']} | {c['stock_m']} | {c['stock_shape']} | {c['stock_capture']} | {c['error']} | {' '.join(f'{k}:{v}' for k, v in sorted(c['kernel_use'].items(), key=lambda kv: [int(x) for x in kv[0].split('x')]))} | {' '.join(f'{k}:{v}' for k, v in sorted(c['stock_m_rows'].items(), key=lambda kv: -kv[1])[:8])} |")
    for T in order:
        f = os.path.join(d, f"serve-{T}.log")
        if not T.startswith("gv") or not os.path.exists(f): continue
        V = {}
        for l in open(f, errors="replace"):
            if "HLG " in l and '"ev": "verify"' in l:
                o = json.loads(l[l.index("{"):]); V.setdefault(tuple(o["key"]), []).append(o)
        print(f"\n**HLG_VERIFY in {T}: ours against vLLM's path on the same inputs, first 2 calls per kernel and worker**\n\n| kernel M x N x K | compared calls | rel (Frobenius) min-max | max abs difference | largest |reference| | non-finite ours / stock |\n|---|---|---|---|---|---|")
        for k, v in sorted(V.items()):
            print(f"| {k[0]} x {k[1]} x {k[2]} | {len(v)} | {min(o['rel'] for o in v):.2e}-{max(o['rel'] for o in v):.2e} | {max(o['max_abs'] for o in v):g} | {max(o['rmax'] for o in v):g} | {sum(o['nonfinite'] for o in v)} / {sum(o['ref_nonfinite'] for o in v)} |")
    try:
        meta = json.load(open(os.path.join(d, "power-meta.json"))); rows = list(csv.DictReader(open(os.path.join(d, "power.csv")))); t = [float(r["t"]) for r in rows]
        ng = sum(1 for k in rows[0] if k.endswith("_uW"))
        print(f"\n**Power during the timed cells [observed]: GCDs visible to the sampler {meta['gcds']}; per arm the mean hwmon power of each card that reports one (W) and the four Cray accel counters (W)**\n\n| Arm | hwmon W per reporting GCD | accel0-3 W |\n|---|---|---|")
        for T in order:
            cs = [r for r in runs[T][0] if r.get("kind") == "cell" and r["phase"] == "t"]
            if not cs: continue
            a, b = min(c["t_start"] for c in cs), max(c["t_end"] for c in cs); sel = [r for r, tt in zip(rows, t) if a <= tt <= b]
            hw = ["g%d %.0f" % (g, st.mean(float(r["g%d_uW" % g]) / 1e6 for r in sel if r["g%d_uW" % g])) for g in range(ng) if any(r["g%d_uW" % g] for r in sel)]
            print(f"| {T} | {', '.join(hw)} | {', '.join('%.0f' % st.mean(float(r['a%d_W' % i]) for r in sel if r['a%d_W' % i]) for i in range(4))} |")
        # energy of each timed cell on the card that holds the server's GCDs: the Cray accel counter with the highest mean power over the timed cells (meaningful only when no other job is on that card,
        # i.e. the whole-node jobs with the server on the two dies of one card). Counter difference interpolated on the counter's own timestamps at the cell's edges; nothing subtracted; the cell
        # includes its requests' 0.5 s pauses, the same in every arm.
        import bisect
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
                if c.get("kind") == "cell" and c["phase"] == "t": E.setdefault((c["bs"], c["reps"]), {})[T] = (at(c["t_end"]) - at(c["t_start"]), c["t_end"] - c["t_start"], c["prompt_tokens_each"][0] * c["bs"] * c["k"])
        print(f"\n**Energy per timed cell [observed]: card counter accel{card} (mean W of the four counters over all timed cells: {', '.join('%.0f' % m for m in mean)}; GCDs visible to the sampler {meta['gcds']}, so the card is this job's alone only on a whole node). J per cell (cell seconds)**\n\n| Cell | " + " | ".join(order) + " |\n|---|" + "---|" * len(order))
        for key, v in sorted(E.items()):
            print(f"| {next(iter(v.values()))[2] // (key[0] * 8)} x{key[0]} | " + " | ".join(f"{v[T][0]:.0f} ({v[T][1]:.1f} s)" if T in v else "-" for T in order) + " |")
        for x in ("g_rocm", "fg_rocm"):
            if not any(T.startswith(x + "-") for T in order) or not any(T.startswith("s_rocm-") for T in order): continue
            print(f"\n**Energy per cell, {x} / s_rocm: ratio of the arms' medians over repetitions (below 1: less energy), spreads as above; and J per 1,000 prompt tokens, stock -> {x}**\n\n| Cell | s J | {x} J | ratio | s spread | {x} spread | verdict | J per 1k prompt tokens |\n|---|---|---|---|---|---|---|---|")
            for key, v in sorted(E.items()):
                s = [v[T][0] for T in order if T.startswith("s_rocm-") and T in v]; f = [v[T][0] for T in order if T.startswith(x + "-") and T in v]
                if not s or not f: continue
                r = med(f) / med(s); nz = max(sp(s), sp(f)); tok = next(iter(v.values()))[2]
                print(f"| {tok // (key[0] * 8)} x{key[0]} | {', '.join('%.0f' % q for q in s)} | {', '.join('%.0f' % q for q in f)} | {r:.4f} | {100 * sp(s):.2f}% | {100 * sp(f):.2f}% | {'~ inside noise' if not abs(r - 1) > nz else ('less energy' if r < 1 else 'more energy')} | {1e3 * med(s) / tok:.1f} -> {1e3 * med(f) / tok:.1f} |")
    except Exception as e:
        print("\npower: not analysed (" + repr(e)[:160] + ")")
