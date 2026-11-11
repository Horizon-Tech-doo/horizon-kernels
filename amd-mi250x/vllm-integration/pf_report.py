"""PROFILE report. python3 pf_report.py <pf-summary.json> [more] > report.md      (summaries written by pf_analyse.py; arms are looked up across all of them)
Per flag set (d = default max_num_batched_tokens, m16 = 16384) and cell: the cross-check between arms, then the per-operator table of the ops arm (po_*):
  raw        as timed: every operator between two torch.cuda.synchronize() calls
  corrected  raw minus c per synchronise charged to it (one per operator call and per glue segment, two per nested timer), with c = (ops-arm step - sync-arm step without graphs) / synchronise count of that cell,
             so the corrected column sums to the step time of the pyn_* arm (same flags, one synchronise per step) [computed, a model: it assumes every synchronise costs the same]."""
import json, sys
A = {}
for f in sys.argv[1:]:
    for job, arms in json.load(open(f)).items():
        for T, cells in arms.items(): A.setdefault(T.rsplit("-r", 1)[0], (job, cells))
med = lambda v: sorted(v)[len(v) // 2]
GROUP = [("MLP GEMM (gate_up_proj, down_proj)", ["GEMM mlp"]), ("GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj)", ["GEMM gdn_proj"]), ("attention projection GEMM (qkv_proj, o_proj)", ["GEMM attn_proj"]),
         ("attention kernel", ["attention kernel"]), ("KV-cache write", ["KV-cache write"]), ("GDN core (conv + delta rule)", ["GDN core (conv + delta rule)"]), ("tensor-parallel all-reduce", ["TP all_reduce"]),
         ("sampling incl. lm_head GEMM and all-gather", ["runner sample"]), ("runner: input prep, request bookkeeping, Python", ["runner add_requests", "runner update_requests", "runner finish_requests", "runner free_states", "runner prepare_inputs", "runner prepare_attn", "runner postprocess_sampled"])]
def tot(w, pf): return med(w["step_ms_sum"]) if pf else w["step_ms_median"]
for fl in ("d", "m16"):
    if "po_" + fl not in A: continue
    job, po = A["po_" + fl]
    print(f"\n## Flags `{fl}` (ops arm po_{fl}, job {job})\n")
    refs = [a for a in ("s_", "ps_", "py_", "pyn_", "po_") if a + fl in A]
    print("**Cross-check [observed]: client time to first token (s) or decode step (ms), and the server-side step time summed over the prefill's steps (ms) or per decode step, worker 0**\n\n| Cell | prompt tokens | steps (tokens) | " + " | ".join(f"{a}{fl} client" for a in refs) + " | " + " | ".join(f"{a}{fl} server" for a in refs if a != "s_") + " | ps period, ms |\n|---|---|---|" + "---|" * (2 * len(refs)))
    for cell in po:
        pf = cell.startswith("pf"); w = po[cell].get("w0")
        cl = [A[a + fl][1].get(cell, {}).get("client", {}) for a in refs]
        sv = [A[a + fl][1].get(cell, {}).get("w0") for a in refs if a != "s_"]
        print(f"| {cell} | {po[cell]['client']['prompt_tokens_each']} | {w['ntok_per_step'] if (w and pf) else (str(w['steps']) + ' steps' if w else '-')} | " + " | ".join(str(c.get("ttft_median") if pf else c.get("step_ms")) for c in cl) + " | " +
              " | ".join(("%.1f" % tot(x, pf)) if x else "-" for x in sv) + " | " + (str(A["ps_" + fl][1].get(cell, {}).get("w0", {}).get("period_ms_median", "-")) if ("ps_" + fl in A and not pf) else "-") + " |")
    for cell in po:
        w = po[cell].get("w0"); pf = cell.startswith("pf")
        if not w or not w["class_ms"]: continue
        ref = A.get("pyn_" + fl, (None, {}))[1].get(cell, {}).get("w0")
        calls = dict(w["class_calls"]); n_ops = sum(calls.values()); sub = w["sub_ms"]
        nest = {"GDN core (conv + delta rule)": sum(v[0] for k, v in sub.items() if k.startswith("gdn")), "runner sample": sum(v[0] for k, v in sub.items() if not k.startswith("gdn"))}
        nsync = 2 * n_ops + 2 * sum(nest.values())
        raw_tot = sum(w["class_ms"].values()) + sum(w["glue_ms"].values())
        c = (tot(w, pf) - tot(ref, pf)) / nsync if ref else 0.0
        fwd_glue = next((v for k, v in w["glue_ms"].items() if k.startswith("compiled")), 0.0); run_glue = next((v for k, v in w["glue_ms"].items() if k.startswith("runner")), 0.0)
        run_calls = sum(v for k, v in calls.items() if k.startswith("runner "))
        rows = []
        for name, keys in GROUP:
            r = sum(w["class_ms"].get(k, 0.0) for k in keys); n = sum(calls.get(k, 0) for k in keys); nn = sum(nest.get(k, 0) for k in keys)
            rows.append((name, n, r, r - c * (n + 2 * nn)))
        rows.append(("compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes", n_ops - run_calls, fwd_glue, fwd_glue - c * (n_ops - run_calls)))
        rows.append(("runner glue", run_calls + 1, run_glue, run_glue - c * run_calls))
        ct = sum(r[3] for r in rows)
        unit = "per prefill, all its steps" if pf else "per decode step"
        print(f"\n### `{fl}` {cell}: prompt {po[cell]['client']['prompt_tokens_each']} tokens, {unit}" + (f", steps {w['ntok_per_step']}" if pf else f", {w['steps']} steps") + f"\n\nOps arm {tot(w, pf):.1f} ms, timed parts sum {raw_tot:.1f} ms, same flags with one synchronise per step {tot(ref, pf) if ref else float('nan'):.1f} ms, {nsync:.0f} synchronises, c = {c * 1e3:.0f} us [computed].\n")
        print("| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |\n|---|---|---|---|---|---|")
        for name, n, r, cr in rows: print(f"| {name} | {n:g} | {r:.1f} | {100 * r / raw_tot:.1f}% | {cr:.1f} | {100 * cr / ct:.1f}% |")
        print(f"| sum | | {raw_tot:.1f} | | {ct:.1f} | |")
        print("\n| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |\n|---|---|---|---|---|---|---|")
        for k, (cnt, ms, per, nme) in w["gemm"].items():
            p = k.split("|"); M, N, K = int(p[2]), int(p[3]), int(p[4])
            print(f"| {p[1]} {M} x {N} x {K} | {nme} | {cnt:g} | {per:.4f} | {ms:.1f} | {2.0 * M * N * K / per * 1e-9:.1f} | {ms - c * cnt:.1f} |")
        print("\nNested timers, raw (calls, ms): " + "; ".join(f"{k.replace('gdnsub:', '').replace('gdn:', '')} {v[0]:g}, {v[1]:.1f}" for k, v in sub.items()) + ".")
        print("Glue by neighbours, raw ms (previous operator > next): " + "; ".join(f"{k} {v:.1f}" for k, v in w["glue_pairs_ms"].items()) + ".")
