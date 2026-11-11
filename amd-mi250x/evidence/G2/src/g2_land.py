"""G2 step 2: land the raced winners. python3 g2_land.py <forms.txt> <forward check.jsonl> <reversed check.jsonl> <gemm_bf16.py> > landed.md
Rule, fixed before the race: an entry lands when vLLM / ours exceeds 1 + noise in BOTH jobs, noise = that cell's |stock_b / stock - 1| in that job floored at 0.5%; and its error against the fp32
product is at most 1.25x vLLM's and the output is finite, the right shape and bitwise repeatable in both jobs. Everything else stays on vLLM's path. Rewrites the ROWS_FORMS block of gemm_bf16.py."""
import json, sys
forms = {}
for e in open(sys.argv[1]).read().strip().split(";"):
    if not e: continue
    nk, m, tile, sched = e.split(":"); N, K = map(int, nk.split("x")); bM, bN = map(int, tile.split("x")); forms[(int(m), N, K)] = (bM, bN, sched)
def load(f): return {(r["M"], r["N"], r["K"]): r for r in map(json.loads, open(f)) if r["kind"] == "cell" and r["M"] == r.get("M_kernel", r["M"])}
F, R = load(sys.argv[2]), load(sys.argv[3])
land, n = {}, {"LEAD": 0, "TIE": 0, "LOSS": 0, "missing": 0}
print("| M x N x K | form | vLLM / ours forward | reversed | noise forward, reversed | error ours / vLLM (fp32 reference) | verdict |\n|---|---|---|---|---|---|---|")
for key in sorted(forms):
    a, b = F.get(key), R.get(key); f = forms[key]
    if not a or not b:
        n["missing"] += 1; print(f"| {key[0]} x {key[1]} x {key[2]} | {f[0]}x{f[1]} {f[2]} | - | - | - | - | missing: not landed |"); continue
    na, nb = max(a["noise"], 0.005), max(b["noise"], 0.005); x, y = a["stock_over_ours"], b["stock_over_ours"]
    good = all(r["finite"] and r["shape_ok"] and r["repeat_bitwise"] and r["err_ours"] <= 1.25 * r["err_stock"] for r in (a, b))
    v = "LEAD" if x > 1 + na and y > 1 + nb else ("LOSS" if x < 1 - na and y < 1 - nb else "TIE")
    n[v] += 1
    if v == "LEAD" and good: land[key] = f
    print(f"| {key[0]} x {key[1]} x {key[2]} | {f[0]}x{f[1]} {f[2]} | {x:.4f} | {y:.4f} | {100 * a['noise']:.2f}%, {100 * b['noise']:.2f}% | {a['err_ours']:.3e} / {a['err_stock']:.3e} | {v}{'' if good or v != 'LEAD' else ' but check failed'}: {'landed' if key in land else 'not landed'} |")
print(f"\nCandidates {len(forms)}: " + ", ".join(f"{k} {v}" for k, v in n.items()) + f"; landed {len(land)}.")
s = open(sys.argv[4]).read(); i = s.index("    # BEGIN ROWS_FORMS"); j = s.index("    # END ROWS_FORMS")
body = "".join(f"    ({M}, {N}, {K}): ({f[0]}, {f[1]}, \"{f[2]}\"),\n" for (M, N, K), f in sorted(land.items()))
open(sys.argv[4], "w").write(s[:i] + "    # BEGIN ROWS_FORMS\n" + body + s[j:])
