"""QW: time to first token of the three arms (s_rocm stock, g_rocm GEMM plugin, fgw_rocm GEMM + query-window prefill attention) from two mirrored jobs.
python qw_analyse.py <raw dir> <job a> <job b> <label>. Per cell: every repetition's median, the three ratios per job (arm medians over repetitions), spread = (max - min) / median of an arm's
repetitions, verdict LEAD / LOSS only when |ratio - 1| exceeds the larger spread of the two arms in BOTH jobs with the same sign; else tie."""
import glob, json, os, statistics as st, sys
RAW, A, B, label = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
def load(job):
    cells, n = {}, {}
    for f in sorted(glob.glob(os.path.join(RAW, job, "client-*.jsonl")), key=os.path.getmtime):
        T = os.path.basename(f)[7:-6]; arm = T.rsplit("-r", 1)[0]
        rows = []
        for l in open(f):
            try: rows.append(json.loads(l))
            except ValueError: pass
        n[T] = (sum(1 for r in rows if r.get("kind") == "cell" and r["phase"] == "t"), bool(rows) and rows[-1].get("kind") == "client_done")
        for r in rows:
            if r.get("kind") == "cell" and r["phase"] == "t":
                cells.setdefault(("/".join(map(str, sorted(set(r["prompt_tokens_each"])))), r["bs"]), {}).setdefault(arm, []).append(r["ttft_median"])
    return cells, n
def sp(v): return (max(v) - min(v)) / st.median(v) if len(v) > 1 else float("nan")
ca, na = load(A); cb, nb = load(B)
print(f"## {label}: jobs {A} and {B}\n")
for j, n in ((A, na), (B, nb)): print(f"- {j} run order, timed cells, client finished: " + ", ".join(f"{T} {c} {d}" for T, (c, d) in n.items()))
for x, b in (("g_rocm", "s_rocm"), ("fgw_rocm", "s_rocm"), ("fgw_rocm", "g_rocm")):
    print(f"\n**{x} / {b}, time to first token (below 1 = {x} faster)**\n\n| Cell (prompt tokens x streams) | {b} s (job a; job b) | {x} s (job a; job b) | ratio a | ratio b | larger spread a | larger spread b | verdict |\n|---|---|---|---|---|---|---|---|")
    for key in sorted(ca, key=lambda k: (k[1], int(k[0].split("/")[0]))):
        if key not in cb or any(a not in c[key] for c in (ca, cb) for a in (x, b)): continue
        out = []
        for c in (ca, cb):
            s, f = c[key][b], c[key][x]; out.append((st.median(f) / st.median(s), max(sp(s), sp(f)), s, f))
        (ra, za, sa, fa), (rb, zb, sb, fb) = out
        v = "LEAD" if ra < 1 - za and rb < 1 - zb else "LOSS" if ra > 1 + za and rb > 1 + zb else "tie"
        fm = lambda v_: ", ".join("%.4f" % q for q in v_)
        print(f"| {key[0]} x{key[1]} | {fm(sa)}; {fm(sb)} | {fm(fa)}; {fm(fb)} | {ra:.4f} | {rb:.4f} | {100 * za:.2f}% | {100 * zb:.2f}% | {v} |")
