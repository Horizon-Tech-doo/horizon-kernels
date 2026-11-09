"""Final table from raw jobs: per job, ratio to same-job CK and same-stack SDPA."""
import json, statistics as st, sys, collections as c
jobs = sys.argv[1:]
arms = {"d256": "P32_b32_t768_land_fullERDW", "d128": "r128_b32_t512_land_fullERDW", "d128b": "P32_b96_t768_land_fullERDW"}
for j in jobs:
    d = c.defaultdict(list)
    for l in open(f"raw/{j}/run.jsonl"):
        r = json.loads(l)
        if "tflops" in r and "case" in r and r.get("input", "randn0") == "randn0":
            d[(r["case"], r["name"].replace(r["case"] + "_", ""), r["stack"][0], r.get("patch", ""))].append(r["tflops"])
    print("job", j)
    for cs in ("d256", "d128"):
        for L in (1024, 4096, 8192, 16384):
            case = f"{cs}_{L}"
            ck = [v for k, v in d.items() if k[0] == case and k[1] == "ck"]
            if not ck:
                continue
            ck = st.median(ck[0])
            sd = {s: st.median(v) for k, v in d.items() if k[0] == case and k[1] == "sdpa" for s in [k[2]]}
            out = [case, f"CK {ck:.2f}", f"SDPA i {sd.get('i', 0):.2f} l {sd.get('l', 0):.2f}"]
            for k, v in sorted(d.items()):
                if k[0] == case and k[1] in arms.values() and k[3] == "OYBD":
                    m = st.median(v)
                    out.append(f"{k[1].split('_')[0]}_{k[1].split('_')[1]} {k[2]} {m:.2f} /CK {m / ck:.3f} /SDPA {m / sd.get(k[2], 1):.3f}")
            print(" | ".join(out))
