"""Tables for round 6. Ratio: ours / same-process vendor, TF, higher is better. Vendor read: mean of pre and post; pre-only and post-only ratios also shown. Usage: r6tab.py raw6/<job> ..."""
import collections, json, statistics as st, sys
for d in sys.argv[1:]:
    print("##", d)
    for fn, lab in (("forms.jsonl", "forms"), ("tile.jsonl", "hand"), ("tileb.jsonl", "handb"), ("l18.jsonl", "c18")):
        try:
            rows = [json.loads(x) for x in open(f"{d}/{fn}")]
        except FileNotFoundError:
            continue
        P = collections.defaultdict(list)
        for r in rows:
            if "case" in r:
                P[(r["tag"], r["case"], r["input"])].append(r)
        for (tag, case, inp), rs in sorted(P.items(), key=lambda x: (int(x[0][1].split("_")[1]), x[0][2], x[0][0])):
            ck = [r["tflops"] for r in rs if r.get("impl") == "ck"]
            sd = [r["tflops"] for r in rs if r.get("impl", "").startswith("sdpa")]
            ckm = sum(ck) / len(ck) if ck else None
            sdm = sum(sd) / len(sd)
            print(f"{lab} {tag} {case} {inp} CK {ck} SDPA {sd}")
            for r in sorted((r for r in rs if r.get("impl") == "tl"), key=lambda r: -r.get("tflops", 0)):
                if "tflops" not in r:
                    print("   ", r["name"], "ERR", r.get("err")); continue
                ex = f"/CK {r['tflops'] / ckm:.3f} [{r['tflops'] / ck[0]:.3f},{r['tflops'] / ck[-1]:.3f}]" if ck else ""
                print(f"    {r['name']:42s} {r['tflops']:7.2f} {ex} /SDPA {r['tflops'] / sdm:.3f} fp64 {r.get('rel_vs_fp64')} sdpa {r.get('rel_vs_sdpa')} rep {r.get('repeat_equal')} comp {r.get('compiles')}")
