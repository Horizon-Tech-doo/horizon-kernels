"""V1 tables, run in exp/V1/raw (or a copy with the same job folders). Ratios: first arm over second, win_tok_s, median of rounds within each job, higher is better."""
import collections as C, glob, json, statistics as st
def rows(J, a):
    for l in open(f"{J}/client-{a}.jsonl"):
        if l.startswith("{"):
            yield json.loads(l)
def rate(J, a):
    R = C.defaultdict(list)
    for d in rows(J, a):
        if d.get("kind") == "decode" and d["round"] >= 0:
            R[f'{d["ctx"]}x{d["bs"]}'].append(d["win_tok_s"])
    return {k: st.median(v) for k, v in R.items()}
CELLS = ["shortx1", "shortx8", "shortx32", "2kx1", "2kx8", "2kx32", "8kx1", "8kx8"]
PAIRS = [("ip_rocm", "s_rocm", (23526403, 23526408)), ("igs128_rocm", "s_rocm", (23526984, 23526990)), ("s_tri", "s_rocm", (23526403, 23526408, 23526984, 23526990)),
         ("ip_tri", "s_tri", (23526403, 23526408)), ("igs128_tri", "s_tri", (23526984, 23526990)), ("igs128_rocm", "s_tri", (23526984, 23526990)), ("ig_rocm", "s_rocm", (23525757,)), ("p_rocm", "s_rocm", (23525757,)), ("igs32_rocm", "s_rocm", (23526168,))]
print("pair job " + " ".join(CELLS))
for a, b, jobs in PAIRS:
    for J in jobs:
        x, y = rate(J, a), rate(J, b)
        print(a, b, J, " ".join(f"{x[c] / y[c]:.3f}" for c in CELLS))
print("absolute win_tok_s by arm and job")
for a, jobs in (("s_rocm", (23525757, 23526168, 23526403, 23526408, 23526984, 23526990)), ("s_tri", (23526403, 23526408, 23526984, 23526990)), ("igs128_rocm", (23526168, 23526984, 23526990)), ("igs128_tri", (23526984, 23526990)), ("ip_rocm", (23525757, 23526403, 23526408)), ("ip_tri", (23526403, 23526408))):
    for J in jobs:
        x = rate(J, a); print(a, J, " ".join(f"{x[c]:.1f}" for c in CELLS))
def T(J, a):
    o = {}
    for l in open(f"{J}/{a}-texts.jsonl"):
        d = json.loads(l); o[(d["ctx"], d.get("bs"), d.get("round"), d.get("i"), d.get("id"))] = d
    return o
def cmp(A, B):
    c = C.Counter(); m = []
    for k, d in A.items():
        if k not in B or "texts" in d:
            continue
        e = B[k]; c[k[0] + "_rows"] += 1; c[k[0] + "_diff"] += d["text"] != e["text"]
        if k[0] == "gsm":
            c["ansdiff"] += d["answer"] != e["answer"]; c["ok_a"] += bool(d["ok"]); c["ok_b"] += bool(e["ok"]); c["okflip_to_wrong"] += bool(d["ok"]) and not e["ok"]; c["okflip_to_right"] += bool(e["ok"]) and not d["ok"]
        if d.get("lp") and e.get("lp") and d["text"] != e["text"]:
            i = next((i for i, (x, y) in enumerate(zip(d["lp"], e["lp"])) if x[0] != y[0]), None)
            if i is not None:
                v = sorted(d["lp"][i][1].values(), reverse=True)
                m.append((round(v[0] - v[1], 4) if len(v) > 1 else None, k[0], k[3] if k[3] is not None else k[4], i))
    return c, m
print("text and answers: kind diff/rows; margin = nats between top 2 in the first arm at the first differing token")
Q = [(23526984, "s_rocm", 23526990, "s_rocm"), (23526403, "s_rocm", 23526408, "s_rocm"), (23526984, "s_tri", 23526990, "s_tri"), (23526403, "s_tri", 23526408, "s_tri")] + [(J, "s_rocm", J, a) for J, a in ((23526403, "ip_rocm"), (23526408, "ip_rocm"), (23526984, "igs128_rocm"), (23526990, "igs128_rocm"))] + [(J, "s_tri", J, a) for J, a in ((23526403, "ip_tri"), (23526408, "ip_tri"), (23526984, "igs128_tri"), (23526990, "igs128_tri"))] + [(J, "s_rocm", J, "s_tri") for J in (23526984, 23526990)]
for Ja, a, Jb, b in Q:
    c, m = cmp(T(Ja, a), T(Jb, b)); ms = sorted(x[0] for x in m if x[0] is not None)
    h = C.Counter(ms)
    print(f"{a}@{Ja} vs {b}@{Jb}", dict(c), "margins", sorted(h.items()), "n", len(ms))
print("counters, per worker")
for f in sorted(glob.glob("*/ctr-i*/*.json")):
    d = json.load(open(f))
    if d["layers_seen"]:
        t = d["hit"] + d["stock_shape"] + d["stock_miss"] + d["stock_capture"]
        print(f.split("/")[0], f.split("/")[1], {k: d[k] for k in ("hit", "capture_hit", "stock_shape", "stock_miss", "stock_capture", "error", "layers_seen", "layers_hit", "calls_by_D")}, "hit/hook calls", round(d["hit"] / max(t, 1), 3))
print("cpu fraction of one core at 32 streams, median over rounds")
for J, a in ((23526984, "s_rocm"), (23526984, "igs128_rocm"), (23526984, "s_tri"), (23526984, "igs128_tri"), (23526403, "ip_rocm")):
    for c in ("short", "2k"):
        r = [d for d in rows(J, a) if d.get("kind") == "decode" and d["round"] >= 0 and d["ctx"] == c and d["bs"] == 32]
        print(J, a, c, "client", st.median(x["cpu_client"] for x in r), "api", st.median(x["cpu_api"] for x in r), "engine", st.median(x["cpu_engine"] for x in r), "workers", st.median(x["cpu_workers"] for x in r), "client_win", st.median(x["win_tok_s"] for x in r), "engine_rate", st.median(x["engine_decode_tok_s"] for x in r))
