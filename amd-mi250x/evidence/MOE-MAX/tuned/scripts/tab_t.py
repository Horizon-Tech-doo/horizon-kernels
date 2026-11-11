"""Tuned-baseline table from two mxt.py JSONL files (opposite arm and case order). Graph replay only.
Cell = (seed tag, T, routing). Per job and cell, a time ratio is median over rounds of arm A / median over rounds of arm B
(each round value is the median of 60 replays). Lower ours/vLLM = ours faster.
Rule (written before the race jobs ran, same form as round 6): noise(job) = max over every cell of that job of
|same/ship - 1| and |vllm2/vllm - 1|. For a (T, routing) and a baseline B (vllm = default config, vllmt = tuned config):
  LEAD     B/ours > 1 + noise(job) in both seeds of both jobs
  REVERSED ours/B > 1 + noise(job) in both seeds of both jobs
  TIE      anything else (inside the noise, or not the same sign in all four cells)
Added after the first race pair (the rule above is unchanged): the envelope 'best' = per cell the fastest of the vLLM arms
present (default, tuned, and vllmz = tuned on zipf2 routing when that arm was raced); LEAD / REVERSED / TIE for 'best' apply
the same thresholds to that fastest arm. More than two job files may be given: ranges and verdicts then span every job (a
verdict needs every cell of every job); vllmz and 'best' use the arms each job raced. argv: jsonl_fwd jsonl_rev [more jsonl ...] [--md]"""
import json, sys, statistics as st
from collections import defaultdict
def load(p):
    t = defaultdict(list); err = defaultdict(lambda: [0.0, 0.0, 0]); tvd = [0.0, 0.0, 1.0, 0]; cfg = {}; chk = []
    for l in open(p):
        if not l.startswith("{"):
            continue
        try:
            r = json.loads(l)
        except ValueError:
            continue
        k = r.get("kind")
        if k == "time":
            t[(r["tag"], r["T"], r["route"], r["arm"].replace("g:", ""))].append(r["tot_med"])
        elif k == "err":
            e = err[r["arm"].replace("g:", "")]
            e[0] = max(e[0], r["rel_vs_fp32"]); e[1] = max(e[1], r["rel_vs_fp64"]); e[2] += r["nonfinite"]
        elif k == "tuned_vs_default":
            tvd[0] = max(tvd[0], r["maxabs"]); tvd[1] = max(tvd[1], r["rel"]); tvd[2] = min(tvd[2], r["bitwise_equal_share"]); tvd[3] += 1
        elif k == "vllm_config":
            cfg[(r["arm"], r["T"])] = r["cfg"]
        elif k == "cfgcheck":
            chk.append(r)
    return {k: st.median(v) for k, v in t.items()}, {k: v for k, v in err.items()}, tvd, cfg, chk, {k: len(v) for k, v in t.items()}
J = [load(p) for p in sys.argv[1:] if not p.startswith("--")]
md = "--md" in sys.argv
RT = ("uniform", "zipf1", "zipf2")
cells = sorted({k[:3] for k in J[0][0]}, key=lambda k: (k[1], k[2], k[0]))
q = lambda j, c, a, b: j[0][c + (a,)] / j[0][c + (b,)]
noise = [max(max(abs(q(j, c, "same", "ship") - 1), abs(q(j, c, "vllm2", "vllm") - 1)) for c in {k[:3] for k in j[0]}) for j in J]
for i, j in enumerate(J):
    cs = {k[:3] for k in j[0]}
    print(f"job{i}: cells {len(cs)}  rounds per arm {sorted(set(j[5].values()))}  noise {noise[i]*100:.2f}%  (ours same/ship max {max(abs(q(j, c, 'same', 'ship') - 1) for c in cs)*100:.2f}%, "
          f"vLLM vllm2/vllm max {max(abs(q(j, c, 'vllm2', 'vllm') - 1) for c in cs)*100:.2f}%)")
    print(f"job{i}: worst rel err vs fp32 / vs fp64 / non-finite per arm: " + "; ".join(f"{a} {e[0]:.3e} / {e[1]:.3e} / {e[2]}" for a, e in sorted(j[1].items())))
    print(f"job{i}: tuned vs default output over {j[2][3]} cases: max abs diff {j[2][0]:.3e}, max rel L2 {j[2][1]:.3e}, min bitwise-equal share {j[2][2]:.4f}")
    for r in j[4]:
        print(f"job{i}: cfgcheck ok={r['ok']} file_name_matches={r['file_name_matches']} loaded_keys={r['loaded_keys']} mismatch_T={r['mismatch_T']}")
print("\nper cell (jobs in argv order)")
print(f"{'seed':4s} {'T':>5s} {'route':7s} | ours/vllm-def j0 j1 | ours/vllm-tuned j0 j1 | tuned/def j0 j1 | same/ship j0 j1 | vllm2/vllm j0 j1 | us ours def tuned (j0)")
for c in cells:
    f = lambda a, b: " ".join(f"{q(j, c, a, b):.3f}" for j in J)
    print(f"{c[0]:4s} {c[1]:5d} {c[2]:7s} | {f('ship', 'vllm'):19s} | {f('ship', 'vllmt'):21s} | {f('vllmt', 'vllm'):15s} | {f('same', 'ship'):15s} | {f('vllm2', 'vllm'):16s} | "
          f"{J[0][0][c + ('ship',)]*1e3:9.1f} {J[0][0][c + ('vllm',)]*1e3:9.1f} {J[0][0][c + ('vllmt',)]*1e3:9.1f}")
VA = lambda j, c: [b for b in ("vllm", "vllmt", "vllmz") if c + (b,) in j[0]]
def qb(j, c, b):
    return max(q(j, c, "ship", x) for x in VA(j, c)) if b == "best" else q(j, c, "ship", b)
def verdict(T, route, b):
    cs = [c for c in cells if c[1] == T and c[2] == route]
    v = [(qb(j, c, b), noise[i]) for i, j in enumerate(J) for c in cs]
    if all(1 / r > 1 + n for r, n in v):
        return "LEAD"
    if all(r > 1 + n for r, n in v):
        return "REVERSED"
    return "TIE"
def rng(T, route, a, b):
    v = [(qb(j, c, b) if b == "best" else q(j, c, a, b)) for j in J for c in cells if c[1] == T and c[2] == route and (b == "best" or c + (b,) in j[0])]
    return f"{min(v):.3f}-{max(v):.3f}" if v else "-"
HZ = any(k[3] == "vllmz" for j in J for k in j[0])
JZ = next((j for j in J if any(k[3] == "vllmz" for k in j[0])), J[0])
TS = sorted({c[1] for c in cells})
print(f"\nranges over 2 seeds x {len(J)} jobs; verdict by the rule")
if md:
    print("| T | routing | ours / vLLM default | verdict | ours / vLLM tuned | verdict |" + (" ours / vLLM zipf2-tuned |" if HZ else "") + " ours / best vLLM | verdict | tuned / default | same / ship | vllm2 / vllm |")
    print("|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if HZ else ""))
for T in TS:
    for route in RT:
        z = (rng(T, route, "ship", "vllmz"),) if HZ else ()
        row = (T, route, rng(T, route, "ship", "vllm"), verdict(T, route, "vllm"), rng(T, route, "ship", "vllmt"), verdict(T, route, "vllmt")) + z + (
               rng(T, route, "ship", "best"), verdict(T, route, "best"), rng(T, route, "vllmt", "vllm"), rng(T, route, "same", "ship"), rng(T, route, "vllm2", "vllm"))
        if md:
            print("| " + " | ".join(str(x) for x in row) + " |")
        else:
            print(f"{row[0]:5d} {row[1]:7s} | def {row[2]} {row[3]:8s} | tuned {row[4]} {row[5]:8s} |" + (f" ztuned {z[0]:11s} |" if HZ else "") +
                  f" best {row[-5]} {row[-4]:8s} | tuned/def {row[-3]} | same {row[-2]} | vllm2 {row[-1]}")
print("\nvLLM configs used (job0)")
for T in TS:
    print(f"T={T:5d} default {json.dumps(J[0][3].get(('vllm', T)))}\n        tuned   {json.dumps(J[0][3].get(('vllmt', T)))}" + (f"\n        ztuned  {json.dumps(JZ[3].get(('vllmz', T)))}" if HZ else ""))
