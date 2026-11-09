"""Tables from n3r3r raw. usage: ana_r3r.py file... Ratio vLLM ms / our ms."""
import json, sys, statistics as st, collections as C
for f in sys.argv[1:]:
    T = C.defaultdict(list); E = {}; Pc = {}
    for line in open(f):
        if not line.startswith("{"): continue
        try: r = json.loads(line)
        except Exception: continue
        if r.get("t") == "time": T[(r["layer"], r["src"], r["dt"], r["B"], r["impl"])].append(r["ms"]); Pc[(r["layer"], r["src"], r["dt"], r["B"], r["impl"])] = r["pct"]
        elif r.get("t") == "err": E[(r["layer"], r["src"], r["dt"], r["B"], r["impl"])] = (r["rel"], r["finite"])
        elif r.get("t") == "data": print(r)
    print("==", f)
    M = {k: st.median(v) for k, v in T.items()}
    for k in sorted({k[:4] for k in M}):
        g = lambda i: M[k + (i,)]
        e = max(E[k + (i,)][0] for i in ["rh", "rh_vk", "n"]); ev = max(E[k + (i,)][0] for i in ["roc", "tri"]); fin = all(E[k + (i,)][1] for i in ["rh", "n", "roc", "tri"])
        print(*k, " ".join(f"{i}={g(i):.3f}" for i in ["rh", "rh_vk", "n", "roc", "tri"]), f"| rh/roc {g('roc')/g('rh'):.2f} n/tri {g('tri')/g('n'):.2f} n/roc {g('roc')/g('n'):.2f} | err ours {e:.2e} vllm {ev:.2e} finite {fin} %n {Pc[k+('n',)]}")
