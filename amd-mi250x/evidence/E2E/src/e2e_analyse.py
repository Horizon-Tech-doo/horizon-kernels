"""E2E analysis. python3 e2e_analyse.py <raw dir with one folder per job> [job ...] > tables.md ; also writes <raw>/summary.json
Energy of a window = difference of the Cray pm_counters accelerator energy counter of the card that holds the job's two GCDs, linearly
interpolated on the counter's own timestamps at the window edges. Nothing is subtracted. The amdgpu hwmon package power (primary die reading)
is integrated by trapezoid as a second source. The card is identified per job as the accel counter whose 1 s power best correlates with hwmon."""
import bisect, csv, glob, json, os, statistics as st, sys
RAW = sys.argv[1]; JOBS = sys.argv[2:] or sorted(d for d in os.listdir(RAW) if d.isdigit())
ARMN = {"s_rocm": "A stock default", "igs128_rocm": "B ours (default backend)", "s_tri": "T stock Triton flag", "igs128_tri": "Bt ours (Triton backend)"}
ORDER = ["s_rocm", "igs128_rocm", "s_tri", "igs128_tri"]
CELLS = [("short", 1), ("short", 8), ("short", 32), ("8k", 1), ("8k", 8), ("8k", 32)]
def med(x): return st.median(x)
def fmt(x, d=1): return ("%." + str(d) + "f") % x
def rng(v, d=1): return "%s [%s-%s]" % (fmt(med(v), d), fmt(min(v), d), fmt(max(v), d))
class Power:
    def __init__(s, path):
        rows = list(csv.DictReader(open(path)))
        s.t = [float(r["t"]) for r in rows]
        s.acc = []
        for i in range(4):
            pts = sorted({(int(r["a%d_us" % i]) / 1e6, int(r["a%d_J" % i])) for r in rows if r["a%d_us" % i]})
            s.acc.append(([p[0] for p in pts], [p[1] for p in pts]))
        pts = sorted({(int(r["node_us"]) / 1e6, int(r["node_J"])) for r in rows if r["node_us"]})
        s.node = ([p[0] for p in pts], [p[1] for p in pts])
        allg = range(sum(1 for k in rows[0] if k.endswith("_uW")))
        s.mine = [g for g in allg if sum(1 for r in rows if float(r["g%d_busy" % g] or 0) >= 50) > 25]   # GCDs this job loaded (a whole-node job sees all 8)
        s.ng = len(s.mine); s.nall = len(allg)
        s.paired = s.ng == 2 and sum(1 for g in s.mine if rows[0]["g%d_uW" % g]) == 1                     # one primary die (has a power reading) and one secondary
        s.hw = [sum(float(r["g%d_uW" % g]) for g in s.mine if r["g%d_uW" % g]) / 1e6 for r in rows]
        s.sclk = [[float(r["g%d_sclk_Hz" % g] or 0) / 1e6 for r in rows] for g in s.mine]
        s.busy = [[float(r["g%d_busy" % g] or 0) for r in rows] for g in s.mine]
        s.card = s.pick()
    def at(s, ser, t):
        T, J = ser; i = bisect.bisect_left(T, t)
        if i <= 0: return J[0]
        if i >= len(T): return J[-1]
        return J[i - 1] + (J[i] - J[i - 1]) * (t - T[i - 1]) / (T[i] - T[i - 1])
    def joules(s, ser, a, b): return s.at(ser, b) - s.at(ser, a)
    def hw_joules(s, a, b):
        i, j = bisect.bisect_left(s.t, a), bisect.bisect_right(s.t, b)
        ts = [a] + s.t[i:j] + [b]; ps = [s.hw[min(i, len(s.hw) - 1)]] + s.hw[i:j] + [s.hw[max(j - 1, 0)]]
        return sum((ts[k + 1] - ts[k]) * (ps[k + 1] + ps[k]) / 2 for k in range(len(ts) - 1))
    def stat(s, arr, a, b):
        i, j = bisect.bisect_left(s.t, a), bisect.bisect_right(s.t, b); v = arr[i:j] or [float("nan")]
        return min(v), med(v), max(v)
    def pick(s):
        a, b = s.t[0] + 1, s.t[-1] - 1; n = int((b - a) // 2); s.corr = []
        hw = [s.hw_joules(a + 2 * k, a + 2 * k + 2) / 2 for k in range(n)]
        for i in range(4):
            x = [s.joules(s.acc[i], a + 2 * k, a + 2 * k + 2) / 2 for k in range(n)]
            mx, mh = sum(x) / n, sum(hw) / n
            den = (sum((p - mx) ** 2 for p in x) * sum((q - mh) ** 2 for q in hw)) ** 0.5
            s.corr.append(sum((p - mx) * (q - mh) for p, q in zip(x, hw)) / den if den else 0.0)
        return max(range(4), key=lambda i: s.corr[i])
def load(job):
    d = os.path.join(RAW, job); pw = Power(os.path.join(d, "power.csv")); meta = json.load(open(os.path.join(d, "power-meta.json")))
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "client-*.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.startswith("{")]
        if not rows or rows[-1].get("kind") != "client_done": continue
        runs[(rows[0]["arm"], rows[0]["rep"])] = rows
    idle = [json.loads(l) for l in open(os.path.join(d, "idle-bare.jsonl"))]
    return pw, meta, runs, idle
def W(pw, a, b): return pw.joules(pw.acc[pw.card], a, b) / (b - a), pw.hw_joules(a, b) / (b - a), pw.joules(pw.node, a, b) / (b - a)
out = {"jobs": {}}
METR = [("call_tok_s", "output tok/s, whole call", 1), ("dec_tok_s", "output tok/s, decode window", 1), ("step_ms", "decode step, ms (streams / decode-window tok/s) [computed]", 2), ("ttft_median", "TTFT median, s", 3), ("ttft_max", "TTFT max, s", 3),
        ("j_tok_call", "J per output token, whole call (card counter)", 2), ("w_call", "average W, whole call (card counter)", 0), ("j_tok_dec", "J per output token, decode window (card counter)", 2), ("w_dec", "average W, decode window (card counter)", 0),
        ("w_call_node", "average W, whole call (node counter)", 0), ("j_tok_call_hw", "J per output token, whole call (hwmon)", 2), ("j_tok_dec_hw", "J per output token, decode window (hwmon)", 2), ("j_tok_call_node", "J per output token, whole call (node counter: everything on the node; attributable to this job only when the job holds the whole node)", 2)]
for job in JOBS:
    pw, meta, runs, idle = load(job); D = {}; jo = out["jobs"][job] = {"meta": meta, "paired": pw.paired, "whole_node": pw.nall > pw.ng, "card_corr": [round(c, 3) for c in pw.corr], "card": pw.card, "cells": {}, "idle": {}, "tokens": {}}
    print("\n## Job %s\n" % job)
    print("- GCDs visible to the sampler %d, loaded by the server %d, loaded GCDs are one primary and one secondary die (one card): %s, sampler %.0f Hz, pm_counters scan %s Hz. Correlation of each accel counter's 2 s power with the hwmon power of this job's GCDs: %s; card used: accel%d [computed]." % (pw.nall, pw.ng, pw.paired, meta["hz"], " ".join(meta["pm_raw_scan_hz"]), ", ".join("%.3f" % c for c in pw.corr), pw.card))
    print("- Repetitions finished: " + ", ".join("%s x%d" % (a, sum(1 for k in runs if k[0] == a)) for a in ORDER if any(k[0] == a for k in runs)))
    ib = W(pw, idle[0]["t_start"], idle[0]["t_end"]); jo["idle"]["bare"] = ib
    print("- Idle, no process on the GCDs (30 s): card counter %.1f W, hwmon %.1f W, node counter %.0f W [observed]." % ib)
    print("\n| Arm | idle with server loaded, W, card counter, per repetition | hwmon |\n|---|---|---|")
    for a in ORDER:
        v = [W(pw, r["t_start"], r["t_end"]) for (aa, rep), rows in sorted(runs.items()) if aa == a for r in rows if r["kind"] == "idle_loaded"]
        if v: print("| %s | %s | %s |" % (ARMN[a], ", ".join(fmt(x[0]) for x in v), ", ".join(fmt(x[1]) for x in v))); jo["idle"][a] = v
    for (a, rep), rows in sorted(runs.items()):
        for r in rows:
            if r["kind"] != "cell" or r["phase"] != "t": continue
            c = dict(r); e = pw.joules(pw.acc[pw.card], r["t_start"], r["t_end"]); ed = pw.joules(pw.acc[pw.card], r["dec_t0"], r["dec_t1"])
            c.update(j_call=e, j_tok_call=e / r["out_tokens"], w_call=e / r["call_s"], j_tok_dec=ed / r["dec_tokens"], w_dec=ed / r["dec_s"],
                     j_tok_call_hw=pw.hw_joules(r["t_start"], r["t_end"]) / r["out_tokens"], j_tok_dec_hw=pw.hw_joules(r["dec_t0"], r["dec_t1"]) / r["dec_tokens"],
                     j_tok_call_node=pw.joules(pw.node, r["t_start"], r["t_end"]) / r["out_tokens"], w_call_node=pw.joules(pw.node, r["t_start"], r["t_end"]) / r["call_s"], step_ms=1000.0 * r["bs"] / r["dec_tok_s"],
                     sclk=[pw.stat(pw.sclk[g], r["dec_t0"], r["dec_t1"]) for g in range(pw.ng)], busy=[pw.stat(pw.busy[g], r["dec_t0"], r["dec_t1"])[1] for g in range(pw.ng)])
            D.setdefault((r["ctx"], r["bs"]), {}).setdefault(a, []).append(c)
    print("\nTokens per cell (identical in every arm and repetition unless listed) [observed]:\n\n| Cell | prompt tokens each | prompt tokens total | output tokens | total tokens | prefix-cache hits in timed cell |\n|---|---|---|---|---|---|")
    for cell in CELLS:
        allc = [c for a in D.get(cell, {}) for c in D[cell][a]]
        if not allc: continue
        u = lambda k: "/".join(sorted({str(c[k]) for c in allc}))
        print("| %s x%d | %s | %s | %s | %s | %s |" % (cell[0], cell[1], u("prompt_tokens_each"), u("prompt_tokens"), u("out_tokens"), u("total_tokens"), "/".join(sorted({str(int(c["metrics_delta"].get("prefix_cache_hits_total", -1))) for c in allc}))))
    for key, title, dg in METR:
        print("\n**%s** (per repetition in run order; median [min-max])\n\n| Cell | " % title + " | ".join(ARMN[a] for a in ORDER if any(a in D.get(c, {}) for c in CELLS)) + " |\n|---|" + "---|" * sum(1 for a in ORDER if any(a in D.get(c, {}) for c in CELLS)))
        for cell in CELLS:
            if cell not in D: continue
            line = "| %s x%d |" % cell
            for a in ORDER:
                if not any(a in D.get(c, {}) for c in CELLS): continue
                v = [c[key] for c in D[cell].get(a, [])]
                line += (" %s; %s |" % (", ".join(fmt(x, dg) for x in v), rng(v, dg)) if v else " - |")
                if v: jo["cells"].setdefault("%s x%d" % cell, {}).setdefault(a, {})[key] = v
            print(line)
    print("\n**sclk reading during decode windows, MHz, min/median/max over all repetitions, per GCD (the driver's target level, not a measured frequency)** and gpu_busy_percent median\n\n| Cell | Arm | g0 | g1 | busy g0, g1 |\n|---|---|---|---|---|")
    for cell in CELLS:
        for a in ORDER:
            cs = D.get(cell, {}).get(a, [])
            if not cs: continue
            g = lambda i: "%d/%d/%d" % (min(c["sclk"][i][0] for c in cs), med([c["sclk"][i][1] for c in cs]), max(c["sclk"][i][2] for c in cs)) if i < len(cs[0]["sclk"]) else "-"
            print("| %s x%d | %s | %s | %s | %s |" % (cell[0], cell[1], a, g(0), g(1), ", ".join(fmt(med([c["busy"][i] for c in cs]), 0) for i in range(len(cs[0]["busy"])))))
    # ratios with a noise test
    PAIRS = [("igs128_rocm", "s_rocm"), ("igs128_tri", "s_tri"), ("igs128_rocm", "s_tri"), ("s_tri", "s_rocm")]
    jo["ratios"] = {}
    for key, title, dg in (("dec_tok_s", "decode-window tok/s", 1), ("call_tok_s", "whole-call tok/s", 1), ("j_tok_dec", "decode-window J/token", 2), ("j_tok_call", "whole-call J/token", 2), ("w_dec", "decode-window average W", 0), ("j_tok_call_node", "whole-call J/token on the node counter", 2), ("ttft_median", "TTFT median", 3)):
        print("\n**Ratio of medians, %s** (x over y; in brackets the larger of the two arms' own repetition spread (max-min)/median = the A-vs-A noise; `~` marks |ratio-1| <= noise: no measurable difference)\n\n| Cell | " % title + " | ".join("%s / %s" % (x, y) for x, y in PAIRS) + " |\n|---|---|---|---|---|")
        for cell in CELLS:
            if cell not in D: continue
            line = "| %s x%d |" % cell
            for x, y in PAIRS:
                vx, vy = [c[key] for c in D[cell].get(x, [])], [c[key] for c in D[cell].get(y, [])]
                if not vx or not vy: line += " - |"; continue
                r = med(vx) / med(vy); nz = max((max(v) - min(v)) / med(v) for v in (vx, vy)) if min(len(vx), len(vy)) > 1 else float("nan")
                line += " %.3f [%.3f]%s |" % (r, nz, " ~" if abs(r - 1) <= nz else "")
                jo["ratios"].setdefault(key, {}).setdefault("%s x%d" % cell, {})["%s/%s" % (x, y)] = [round(r, 4), round(nz, 4)]
            print(line)
    # equivalence on the fixed greedy set
    T = {}
    for f in glob.glob(os.path.join(RAW, job, "texts-*.jsonl")):
        k = os.path.basename(f)[6:-6]
        T[k] = [json.loads(l) for l in open(f)]; T[k] = [r for r in T[k] if r["kind"] == "det"]
    def cmp(x, y):
        same, tot, margins = 0, 0, []
        for a, b in zip(T[x], T[y]):
            tot += 1
            if a["text"] == b["text"]: same += 1; continue
            for i, (p, q) in enumerate(zip(a["lp"], b["lp"])):
                if p[0] != q[0]:
                    v = sorted(p[1].values(), reverse=True); margins.append((i, round(v[0] - v[1], 3) if len(v) > 1 else None)); break
        return same, tot, margins
    ks = sorted(k for k in T if len(T[k]) == 12)
    print("\n**Fixed greedy set (8 short x 128 tokens, 4 long x 64 tokens, one request at a time): identical texts of 12; for differing rows (first differing token index, top-2 logprob margin there in the first-named run)**\n\n| Pair | identical | first divergences |\n|---|---|---|")
    jo["equiv"] = {}
    for i, x in enumerate(ks):
        for y in ks[i + 1:]:
            s, t, m = cmp(x, y); jo["equiv"]["%s vs %s" % (x, y)] = [s, t, m]
            ax, ay = x.rsplit("-r", 1)[0], y.rsplit("-r", 1)[0]
            if ax == ay or x.rsplit("-r", 1)[1] == y.rsplit("-r", 1)[1]:
                print("| %s vs %s | %d of %d | %s |" % (x, y, s, t, ", ".join("(%d, %s)" % z for z in m) or "-"))
json.dump(out, open(os.path.join(RAW, "summary.json"), "w"), indent=1)
