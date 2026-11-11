"""G2 step 1: candidate tile per (rows, shape) from the tile sweeps. python3 g2_forms.py <out forms.txt> <tiles.jsonl> [more] > candidates.md
For every swept (M, N, K) the form with the highest vLLM / ours ratio; a candidate only if that ratio is above 1 + the cell's noise (floored at 0.5%). The sweep is a search in one job, so the candidates
are biased upward by selection: g2_check.py then races each candidate against vLLM's path in two opposite-order jobs and g2_land.py keeps the winners. Output file: "NxK:M:bMxbN:sched;..." (HLG_FORMS)."""
import json, sys
best = {}
for f in sys.argv[2:]:
    for l in open(f):
        r = json.loads(l)
        if r.get("kind") != "cell" or not r["best"]: continue
        best[(r["M"], r["N"], r["K"])] = (r["best"][0][0], r["best"][0][1], r["noise"], r["all"].get("224x256:xs4", [None])[0], r["forms"], f.split("/")[-2])
forms = []
print("| M x N x K | sweep job | forms that built | best form | vLLM / ours | cell noise | 224x256 xs4 | candidate |\n|---|---|---|---|---|---|---|---|")
for (M, N, K), (r, nm, nz, base, nf, job) in sorted(best.items()):
    ok = r > 1 + max(nz, 0.005)
    print(f"| {M} x {N} x {K} | {job} | {nf} | {nm} | {r:.4f} | {100 * nz:.2f}% | {base} | {'yes' if ok else 'no: stays on vLLM'} |")
    if ok: forms.append(f"{N}x{K}:{M}:{nm}")
open(sys.argv[1], "w").write(";".join(forms) + "\n")
print(f"\n{len(forms)} candidates of {len(best)} cells.")
