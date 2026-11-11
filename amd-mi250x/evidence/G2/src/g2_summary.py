"""G2 compact tables from g2_analyse.py output. python3 g2_summary.py <tables-A.md> <tables-B.md> [label] > summary.md
For every 'x / baseline' comparison present in both jobs: one row per cell with ratio and verdict of job A / job B (time to first token and energy), and the equivalence rows of both jobs."""
import re, sys
A, B = open(sys.argv[1]).read(), open(sys.argv[2]).read(); label = sys.argv[3] if len(sys.argv) > 3 else ""
def comps(t, kind):
    out = {}
    for m in re.finditer(r"\*\*" + kind + r": (\w+) / (\w+)\*\*[^\n]*\n\n((?:\|[^\n]*\n)+)", t):
        rows = [[c.strip() for c in l.split("|")[1:-1]] for l in m.group(3).strip().split("\n")[2:]]
        out[(m.group(1), m.group(2))] = {r[0]: (float(r[3]), r[6], max(float(x[:-1]) if x != "nan%" else 0.0 for x in (r[4], r[5]))) for r in rows}
    return out
short = lambda v: "~" if v.startswith("~") else v.split()[0]
for kind, title in (("Time to first token", "Time to first token"), ("Energy per cell", "Card energy per prefill cell")):
    ca, cb = comps(A, kind), comps(B, kind)
    keys = [k for k in ca if k in cb]
    if not keys: continue
    cells = list(next(iter(ca.values())))
    print(f"\n**{title}, {label}: ratio (verdict) in job A / job B; `~` = inside the arms' spread**\n\n| Comparison | " + " | ".join(cells) + " | largest spread A, B |\n|---|" + "---|" * (len(cells) + 1))
    for k in keys:
        print(f"| {k[0]} / {k[1]} | " + " | ".join(f"{ca[k][c][0]:.3f} {short(ca[k][c][1])} / {cb[k][c][0]:.3f} {short(cb[k][c][1])}" if c in ca[k] and c in cb[k] else "-" for c in cells) + f" | {max(v[2] for v in ca[k].values()):.2f}%, {max(v[2] for v in cb[k].values()):.2f}% |")
for nm, t in (("A", A), ("B", B)):
    m = re.search(r"\*\*Equivalence set.*?\n\n((?:\|[^\n]*\n)+)", t, re.S)
    if m: print(f"\n**Equivalence set, {label}, job {nm}**\n\n" + m.group(1))
