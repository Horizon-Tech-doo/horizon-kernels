"""PROFILE analysis. python3 pf_analyse.py <raw job dir> [more job dirs] > tables.md ; also writes <first dir>/pf-summary.json
Matches the server-side step records of horizon_lumi_profile (one file per tensor-parallel worker) to the client's cells by wall-clock time.
Prefill cell: per call, every step that starts between the request being sent and its first token and schedules more than one token for a request. Decode cell: every step inside the decode window with one token per request.
All times ms. Operators are grouped into classes; glue is the time between two timed operators (Inductor-compiled code and Python)."""
import glob, json, os, statistics as st, sys
DIRS = sys.argv[1:]
med = st.median
def load(d):
    arms = {}
    for f in sorted(glob.glob(os.path.join(d, "client-*.jsonl"))):
        T = os.path.basename(f)[7:-6]
        rows = [json.loads(l) for l in open(f) if l.startswith("{")]
        W = []
        for g in sorted(glob.glob(os.path.join(d, "hpf-" + T, "hpf-*.jsonl"))):
            rr = [json.loads(l) for l in open(g) if l.startswith("{")]
            meta = next((r for r in rr if r.get("kind") == "meta"), {})
            W.append((meta, [r for r in rr if r.get("kind") == "step"]))
        W.sort(key=lambda w: (w[0].get("rank") is None, w[0].get("rank") or 0))
        arms[T] = (rows, W, os.path.basename(d.rstrip("/")))
    return arms
def names(meta):
    """(N, K) -> layer name from the weight list."""
    m = {}
    for k in meta.get("weights", {}):
        n, shp, dt = k.split("|"); s = shp.split("x")
        if len(s) == 2: m.setdefault((int(s[0]), int(s[1])), set()).add(n)
    return {k: "/".join(sorted(v)) for k, v in m.items()}
CLS = [("mlp", ("gate_up_proj", "down_proj")), ("attn_proj", ("qkv_proj", "o_proj")), ("gdn_proj", ("in_proj_qkvz", "in_proj_ba", "out_proj", "in_proj_qkv", "in_proj_z")), ("lm_head", ("lm_head",))]
def gclass(nm):
    for c, keys in CLS:
        if any(k in nm for k in keys): return c
    return "other"
def opclass(label, nm):
    if label.startswith("gemm|"):
        p = label.split("|"); return "GEMM " + gclass(nm.get((int(p[3]), int(p[4])), "?"))
    if label == "attn": return "attention kernel"
    if label == "kv_write": return "KV-cache write"
    if label.startswith("gdn:"): return "GDN core (conv + delta rule)"
    if label in ("all_reduce", "all_gather", "reduce_scatter"): return "TP " + label
    if label.startswith("run:"): return "runner " + label[4:]
    return label
def agg(steps, nm):
    """Sum over the given steps: class -> ms, gemm table, glue pairs, sub."""
    c, g, gl, sub = {}, {}, {}, {}
    global CALLS
    CALLS = {}
    for s in steps:
        for k, (n, ms) in s["ops"].items():
            c[opclass(k, nm)] = c.get(opclass(k, nm), 0.0) + ms; CALLS[opclass(k, nm)] = CALLS.get(opclass(k, nm), 0) + n
            if k.startswith("gemm|"): e = g.setdefault(k, [0, 0.0]); e[0] += n; e[1] += ms
        for k, ms in s["glue"].items(): gl[k] = gl.get(k, 0.0) + ms
        for k, (n, ms) in s["sub"].items(): e = sub.setdefault(k, [0, 0.0]); e[0] += n; e[1] += ms
    return c, g, gl, sub
def glue_class(k, nm):
    a, b = k.split(">")
    def nn(x):
        if x.startswith("g") and "x" in x and x[1:].split("x")[0].isdigit():
            N, K = map(int, x[1:].split("x")); return nm.get((N, K), x)
        return x
    a, b = nn(a), nn(b)
    if a == "start" or b in ("exec_end",) or a in ("fwd_end", "exec_end") or a == "run" or b == "run": return "runner (input prep, bookkeeping, Python)", a + ">" + b
    return "compiled glue in the forward (norms, activations, residuals, RoPE, embedding, reshapes)", a + ">" + b
out = {}
for d in DIRS:
    arms = load(d)
    for T, (rows, W, job) in arms.items():
        cells = [r for r in rows if r.get("kind") == "cell" and r["phase"] == "t"]
        print(f"\n## Job {job} arm {T}: workers with step records {len(W)}, modes {[w[0].get('mode') for w in W]}, ranks {[w[0].get('rank') for w in W]}, steps {[len(w[1]) for w in W]}\n")
        if not cells: print("no timed cells"); continue
        o = out.setdefault(job, {}).setdefault(T, {})
        print("| Cell | prompt tokens | client TTFT median, s (all) | client decode step, ms | decode tok/s |\n|---|---|---|---|---|")
        for r in cells:
            print(f"| {r['cell']} | {r['prompt_tokens_each']} | {r['ttft_median']} ({r.get('ttft', '')}) | {r.get('step_ms', '-')} | {r.get('dec_tok_s', '-')} |")
            o[r["cell"]] = {"client": {k: r.get(k) for k in ("prompt_tokens_each", "ttft", "ttft_median", "step_ms", "dec_tok_s", "ttft_max")}}
        for wi, (meta, steps) in enumerate(W):
            nm = names(meta); mode = meta.get("mode")
            if wi == 0 and meta.get("weights"):
                print(f"\nWeights of two or more dimensions on this worker (name | shape | dtype: count, GiB) [observed]: " + "; ".join(f"{k}: {v[0]}, {v[1] / 2**30:.2f}" for k, v in sorted(meta["weights"].items(), key=lambda kv: -kv[1][1])[:14]) + f". All parameters {meta.get('param_bytes', 0) / 2**30:.2f} GiB.")
            for r in cells:
                key = "w%d" % wi
                if r["cell"].startswith("pf"):
                    per = []
                    for t_send, t_first, t_last in r["calls"]:
                        ss = [s for s in steps if t_send - 0.02 <= s["t"] <= t_first and s["maxq"] > 1]
                        if ss: per.append(ss)
                    if not per: continue
                    tot = lambda ss, k: sum(s.get(k) or 0.0 for s in ss)
                    row = {"calls": len(per), "steps_per_call": [len(ss) for ss in per], "ntok_per_step": [s["ntok"] for s in per[0]], "step_ms_sum": [round(tot(ss, "step_ms"), 2) for ss in per], "exec_ms_sum": [round(tot(ss, "exec_ms"), 2) for ss in per],
                           "sample_ms_sum": [round(tot(ss, "sample_ms"), 2) for ss in per], "span_ms": [round((ss[-1]["t"] - ss[0]["t"]) * 1e3 + (ss[-1].get("step_ms") or 0), 2) for ss in per]}
                    use = [s for ss in per for s in ss]; n = len(per)
                else:
                    use = [s for s in steps if r["dec_t0"] <= s["t"] <= r["dec_t1"] and s["maxq"] == 1 and s["nreq"] == r["bs"]]
                    if not use: continue
                    n = len(use)
                    row = {"steps": n, "step_ms_median": round(med([s.get("step_ms") or 0 for s in use]), 3), "exec_ms_median": round(med([s.get("exec_ms") or 0 for s in use]), 3), "sample_ms_median": round(med([s.get("sample_ms") or 0 for s in use]), 3),
                           "between_ms_median": round(med([s.get("between_ms") or 0 for s in use]), 3), "period_ms_median": round(med([s["period_ms"] for s in use if s["period_ms"]]), 3)}
                c, g, gl, sub = agg(use, nm)
                row["class_ms"] = {k: round(v / n, 3) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}
                row["class_calls"] = {k: round(v / n, 3) for k, v in CALLS.items()}
                gc = {}
                for k, v in gl.items():
                    cl, _ = glue_class(k, nm); gc[cl] = gc.get(cl, 0.0) + v
                row["glue_ms"] = {k: round(v / n, 3) for k, v in gc.items()}
                row["glue_pairs_ms"] = {glue_class(k, nm)[1]: round(v / n, 3) for k, v in sorted(gl.items(), key=lambda kv: -kv[1])[:14]}
                row["gemm"] = {k: [round(v[0] / n, 2), round(v[1] / n, 3), round(v[1] / v[0], 4), nm.get((int(k.split("|")[3]), int(k.split("|")[4])), "?")] for k, v in sorted(g.items(), key=lambda kv: -kv[1][1])}
                row["sub_ms"] = {k: [round(v[0] / n, 2), round(v[1] / n, 3)] for k, v in sorted(sub.items(), key=lambda kv: -kv[1][1])}
                o[r["cell"]][key] = row
        # tables for worker 0
        for r in cells:
            w = o[r["cell"]].get("w0")
            if not w: continue
            unit = "per prefill (all its steps)" if r["cell"].startswith("pf") else "per decode step"
            head = {k: v for k, v in w.items() if k not in ("class_ms", "class_calls", "glue_ms", "glue_pairs_ms", "gemm", "sub_ms")}
            w1 = o[r["cell"]].get("w1", {})
            print(f"\n### {job} {T} {r['cell']} ({unit}), worker 0: {json.dumps(head)}" + (f"; worker 1: {json.dumps({k: v for k, v in w1.items() if k in ('step_ms_sum', 'step_ms_median', 'period_ms_median')})}" if w1 else ""))
            if not w["class_ms"] and not w["glue_ms"]: continue
            tot = sum(w["class_ms"].values()) + sum(w["glue_ms"].values())
            print(f"\n| Class | ms {unit} | share of timed total {tot:.2f} ms |\n|---|---|---|")
            for k, v in list(w["class_ms"].items()) + list(w["glue_ms"].items()):
                print(f"| {k} | {v:.3f} | {100 * v / tot:.1f}% |")
            print(f"\n| GEMM (path, M x N x K, in dtype, weight dtype) | layer | calls | ms per call | ms total | TFLOP/s |\n|---|---|---|---|---|---|")
            for k, (cnt, ms, per, nme) in w["gemm"].items():
                p = k.split("|"); M, N, K = int(p[2]), int(p[3]), int(p[4])
                print(f"| {p[1]} {M} x {N} x {K} {p[5]} {p[6]} {p[7]} | {nme} | {cnt:g} | {per:.4f} | {ms:.3f} | {2.0 * M * N * K / per * 1e-9:.1f} |")
            print("\nGlue by neighbours, ms: " + "; ".join(f"{k} {v}" for k, v in w["glue_pairs_ms"].items()))
            if w["sub_ms"]: print("\nNested timers (inside the classes above; calls, ms): " + "; ".join(f"{k} {v[0]:g}, {v[1]}" for k, v in w["sub_ms"].items()))
json.dump(out, open(os.path.join(DIRS[0], "pf-summary.json"), "w"), indent=1)
