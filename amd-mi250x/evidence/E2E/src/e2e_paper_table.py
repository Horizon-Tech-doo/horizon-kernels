"""Compact tables from summary.json (written by e2e_analyse.py). python3 e2e_paper_table.py <raw>/summary.json"""
import json, statistics as st, sys
S = json.load(open(sys.argv[1]))["jobs"]; A = [("s_rocm", "A"), ("igs128_rocm", "B"), ("s_tri", "T"), ("igs128_tri", "Bt")]
def cell(v, d): return ("%." + str(d) + "f [%." + str(d) + "f-%." + str(d) + "f]") % (st.median(v), min(v), max(v))
for job, J in S.items():
    n = {a: len(next(iter(J["cells"].values())).get(a, {}).get("dec_tok_s", [])) for a, _ in A}
    print("\n### Job %s (%s; repetitions %s)\n" % (job, "whole node" if J.get("whole_node") else "2 GCDs of a node", ", ".join("%s %d" % (l, n[a]) for a, l in A)))
    for key, title, d in (("dec_tok_s", "Output tokens/s, decode window", 1), ("call_tok_s", "Output tokens/s, whole call (prefill included)", 1), ("ttft_median", "Time to first token, median over streams, s", 3),
                          ("j_tok_dec", "J per output token, decode window, card counter", 2), ("j_tok_call", "J per output token, whole call, card counter", 2), ("w_dec", "Average W, decode window, card counter", 0)) + ((("j_tok_call_node", "J per output token, whole call, node counter", 2), ("w_call_node", "Average W, whole call, node counter", 0)) if J.get("whole_node") else ()):
        print("**%s** (median [min-max] over repetitions)\n\n| Cell | " % title + " | ".join(l for _, l in A) + " |\n|---|---|---|---|---|")
        for c, V in J["cells"].items():
            print("| %s | " % c + " | ".join(cell(V[a][key], d) if a in V else "-" for a, _ in A) + " |")
        print()
    print("**Ratios of medians** (`~`: inside the larger repetition spread of the two arms, no measurable difference)\n\n| Metric | Cell | B / A | Bt / T | B / T | T / A |\n|---|---|---|---|---|---|")
    for key, title in (("dec_tok_s", "tok/s, decode"), ("call_tok_s", "tok/s, call"), ("j_tok_dec", "J/token, decode"), ("j_tok_call", "J/token, call"), ("ttft_median", "TTFT")):
        for c, R in J["ratios"].get(key, {}).items():
            f = lambda k: ("%.3f%s" % (R[k][0], " ~" if abs(R[k][0] - 1) <= R[k][1] else "")) if k in R else "-"
            print("| %s | %s | %s | %s | %s | %s |" % (title, c, f("igs128_rocm/s_rocm"), f("igs128_tri/s_tri"), f("igs128_rocm/s_tri"), f("s_tri/s_rocm")))
