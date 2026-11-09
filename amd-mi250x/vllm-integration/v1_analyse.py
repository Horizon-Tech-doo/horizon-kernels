"""V1 analysis, run on LUMI in exp/V1/raw. python v1_analyse.py <job> <stock arm> <arm> [...]. Ratio: arm over stock, win_tok_s, median of rounds, higher is better. Text: rows that differ against the same-job stock arm, the stock arm's own round-to-round rate as the noise floor, and for logprob rows (det, det2k, gsm; client from job 23526168) the stock top-2 margin at the first differing token."""
import collections as C, json, statistics as st, sys
J, base, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
def rows(a):
    for l in open(f"{J}/client-{a}.jsonl"):
        if l.startswith("{"):
            yield json.loads(l)
def rate(a):
    R = C.defaultdict(list)
    for d in rows(a):
        if d.get("kind") == "decode" and d["round"] >= 0:
            R[(d["ctx"], d["bs"])].append(d)
    return R
def texts(a):
    out = {}
    for l in open(f"{J}/{a}-texts.jsonl"):
        d = json.loads(l)
        out[(d["ctx"], d.get("bs"), d.get("round"), d.get("i"), d.get("id"))] = d
    return out
def med(x):
    return round(st.median(x), 3)
Rb, Tb = rate(base), texts(base)
same = C.Counter()                            # stock round 0 against stock round 1, row level
for k, d in Tb.items():
    if d.get("round") == 0 and (k[0], k[1], 1, None, None) in Tb:
        for s, t in zip(d["texts"], Tb[(k[0], k[1], 1, None, None)]["texts"]):
            same[(k[0], k[1], "rows")] += 1; same[(k[0], k[1], "diff")] += s != t
print("stock against itself, rounds 0 and 1:", {f"{k[0]}x{k[1]}": f"{same[k]}/{same[(k[0], k[1], 'rows')]}" for k in same if k[2] == "diff"})
for a in arms:
    Ra, Ta = rate(a), texts(a)
    print(f"== {a} over {base}, win_tok_s, median of rounds")
    print("cell stock arm ratio spread_arm engine_stock engine_arm cpu_client cpu_api ms_per_step_stock ms_per_step_arm")
    for k in Rb:
        if k not in Ra:
            continue
        b, c = [x["win_tok_s"] for x in Rb[k]], [x["win_tok_s"] for x in Ra[k]]
        print(f"{k[0]}x{k[1]}", med(b), med(c), med(c) / med(b).__round__(3) if False else round(med(c) / med(b), 3), [round(min(c), 1), round(max(c), 1)], med([x["engine_decode_tok_s"] or 0 for x in Rb[k]]), med([x["engine_decode_tok_s"] or 0 for x in Ra[k]]),
              med([x["cpu_client"] for x in Ra[k]]), med([x["cpu_api"] for x in Ra[k]]), round(1000 * k[1] / med(b), 1), round(1000 * k[1] / med(c), 1))
    D = C.Counter(); ex = []
    for k, d in Tb.items():
        if k not in Ta:
            continue
        if "texts" in d:
            tag = f"{k[0]}x{k[1]}" + ("warm" if k[2] < 0 else "")
            for s, t in zip(d["texts"], Ta[k]["texts"]):
                D[(tag, "rows")] += 1; D[(tag, "diff")] += s != t
        else:
            D[(k[0], "rows")] += 1; D[(k[0], "diff")] += d["text"] != Ta[k]["text"]
            D[(k[0], "ansdiff")] += k[0] == "gsm" and d["answer"] != Ta[k]["answer"]
            D[(k[0], "ok_stock")] += bool(d.get("ok")); D[(k[0], "ok_arm")] += bool(Ta[k].get("ok"))
            la, lb = d.get("lp"), Ta[k].get("lp")
            if la and lb and d["text"] != Ta[k]["text"]:
                i = next((i for i, (x, y) in enumerate(zip(la, lb)) if x[0] != y[0]), None)
                if i is None:
                    continue
                v = sorted(la[i][1].values(), reverse=True); w = sorted(lb[i][1].values(), reverse=True)
                ex.append((k[0], k[3] if k[3] is not None else k[4], i, round(v[0] - v[1], 4) if len(v) > 1 else None, round(w[0] - w[1], 4) if len(w) > 1 else None, repr(la[i][0]), repr(lb[i][0]), set(la[i][1]) == set(lb[i][1]), round(abs(la[i][1].get(la[i][0], 0) - lb[i][1].get(la[i][0], 0)), 4)))
    print("text:", {str(k[0]): {x: D[(k[0], x)] for x in ("diff", "rows", "ansdiff", "ok_stock", "ok_arm") if (k[0], x) in D} for k in D if k[1] == "rows"})
    if ex:
        print("first differing token: kind id pos stock_margin arm_margin stock_tok arm_tok same_top2 |dlogp of stock token|")
        for e in ex:
            print(*e)
        print("margin at first difference, stock arm: median", med([e[3] for e in ex if e[3] is not None]), "max", max(e[3] for e in ex if e[3] is not None))
    try:
        for d in rows(a):
            if d.get("kind") == "gsm":
                print("gsm", {x: d[x] for x in ("correct", "n", "s", "tok_s")})
    except Exception as e:
        print(e)
