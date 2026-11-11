"""PROFILE / GEMM gate 3 tables. python3 gm_report.py <fwd gm3.jsonl> <rev gm3.jsonl> > tables.md
Ratios are vLLM's path over ours in time (above 1: ours faster). Verdict per form, written before reading the numbers: LEAD when the ratio clears 1 + noise(job) in both jobs, LOSS when it is below 1 - noise(job)
in both, else TIE; noise(job) = the largest |stock_b / stock - 1| of that job (identical-code control), per job over all cells."""
import json, sys
J = []
for f in sys.argv[1:3]:
    rows = [json.loads(l) for l in open(f)]
    J.append(({(r["M"], r["N"], r["K"]): r for r in rows if r["kind"] == "cell"}, [r for r in rows if r["kind"] == "summary"]))
nz = [max(r.get("noise", 0) for r in j[0].values() if "noise" in r) for j in J]
print(f"Cells: {[len(j[0]) for j in J]}; errors: {[sum(1 for r in j[0].values() if 'error' in r) for j in J]}; summary: {[j[1] for j in J]}; noise(job) = {nz} [observed].\n")
def verdict(a, b):
    if a is None or b is None: return "-"
    if a > 1 + nz[0] and b > 1 + nz[1]: return "LEAD"
    if a < 1 - nz[0] and b < 1 - nz[1]: return "LOSS"
    return "TIE"
print("| M x N x K | stock ms (fwd, rev) | stock TFLOP/s | cell noise (fwd, rev) | bucket form: M used | stock / ours as wired (fwd, rev) | verdict | stock / ours on the 224x256 rows form, M as given (fwd, rev) | verdict | error vs fp32: stock, ours, rows form |\n|---|---|---|---|---|---|---|---|---|---|")
for key in J[0][0]:
    a, b = J[0][0][key], J[1][0].get(key, {})
    g = lambda r, k: r.get(k)
    f = lambda x: "-" if x is None else ("%.4f" % x)
    print(f"| {key[0]} x {key[1]} x {key[2]} | {f(g(a, 'stock_ms'))}, {f(g(b, 'stock_ms'))} | {g(a, 'stock_TF')} | {g(a, 'noise')}, {g(b, 'noise')} | {g(a, 'ours')}: {g(a, 'M_bucket')} | {f(g(a, 'stock_over_ours'))}, {f(g(b, 'stock_over_ours'))} | {verdict(g(a, 'stock_over_ours'), g(b, 'stock_over_ours'))} | "
          f"{f(g(a, 'stock_over_r224'))}, {f(g(b, 'stock_over_r224'))} | {verdict(g(a, 'stock_over_r224'), g(b, 'stock_over_r224'))} | {g(a, 'err_stock') or 0:.2e}, {g(a, 'err_ours') or 0:.2e}, {g(a, 'err_ours_r224') or 0:.2e} |" + (" ERROR " + a["error"][-120:] if "error" in a else ""))
