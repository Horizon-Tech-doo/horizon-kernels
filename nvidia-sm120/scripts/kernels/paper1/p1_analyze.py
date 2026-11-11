# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""p1_analyze.py FILE... : keep rule, per-cell ranges, noise bound and energy for the paper-1 NVIDIA races.

CPU only. A JOB is all block files of one target and one arm order (forward or reverse). Per job and cell:
  time of an arm      median over its cycles (>= 9 required) of the per-cycle value in microseconds per call
  ratio               vendor time / our time (higher is better), for every arm of role new / new_alt against every
                      arm of role vendor / vendor_alt; "net" repeats it with the separator's time subtracted from both
  cyc                 min..max of the per-cycle ratio (vendor cycle c / our cycle c): the raw range
  noise bound         per job: max over its cells of |median(ctl) / median(new) - 1| (ctl is the same code as new)
  verdict in a job    WIN if ratio - 1 > noise bound of that job, LOSE if ratio - 1 < -noise bound, else inside
Keep rule: a cell is KEEP (our arm is faster) only if it is WIN in at least one forward and one reverse job and in
every job given, its correctness gate passed in every job, every arm has >= 9 cycles, and no job marks the cell
contaminated (the co-resident server's request counters moved during the cell). BOTH-LOSE is the mirror statement.
Everything else is "inside noise / mixed"; with one direction only the verdict is "single job, not certified".

Energy (from the separate energy pass): joules per call = energy over the window / calls, average watts and calls
per second, per arm; the ratio vendor J / our J; the energy noise bound per job = max |J(ctl) / J(new) - 1|; idle
power of the same block. Raw records are never modified; the net figure printed here is derived:
(average watts - idle watts) x window / calls. Power is for the whole GPU, including the co-resident process.

usage: p1_analyze.py FILE... [--ours new] [--vendor NAME] [--allow-contaminated] [--json OUT]
"""
import argparse
import collections
import json
import statistics as st
import sys


def load(files):
    jobs = collections.OrderedDict()
    for fn in files:
        rows = []
        with open(fn) as f:
            for ln in f:
                if ln.startswith("{"):
                    try:
                        rows.append(json.loads(ln))
                    except ValueError:
                        pass
        head = next((r for r in rows if r.get("t") == "header"), None)
        if head is None or head.get("warm_only"):
            continue
        key = (head["target"], "reverse" if head.get("reverse") else "forward", head.get("seed"))
        j = jobs.setdefault(key, {"files": [], "heads": [], "time": collections.defaultdict(list), "role": {},
                                  "cells": {}, "energy": collections.defaultdict(list), "idle": [], "checks": [],
                                  "errors": [], "order": []})
        j["files"].append(fn)
        j["heads"].append(head)
        for r in rows:
            t = r.get("t")
            if t == "time":
                j["time"][(r["cell"], r["arm"])].append((r["cycle"], r["us"]))
                j["role"][r["arm"]] = r["role"]
            elif t == "energy":
                j["energy"][(r["cell"], r["arm"])].append(r)
                j["role"].setdefault(r["arm"], r["role"])
            elif t == "idle":
                j["idle"].append(r)
            elif t == "check":
                j["checks"].append(r)
            elif t == "cell_done":
                if "params" not in r:      # the gated-delta "gates" pseudo-cell has no arms and no times;
                    j.setdefault("gate_cells", {})[r["cell"]] = r     # it is kept apart from the timed cells
                    continue
                j["cells"][r["cell"]] = r
                if r["cell"] not in j["order"]:
                    j["order"].append(r["cell"])
            elif t == "cell_error":
                j["errors"].append(r)
    return jobs


def tag(key):
    return "rev" if key[1] == "reverse" else "fwd"


def med(j, cell, arm):
    v = [u for _, u in j["time"].get((cell, arm), [])]
    return st.median(v) if v else None


def ncyc(j, cell, arm):
    return len(j["time"].get((cell, arm), []))


def arms_of(j, roles):
    return [a for a, r in j["role"].items() if r in roles]


def noise(j):
    v = []
    for c in j["cells"]:
        n, k = med(j, c, "new"), med(j, c, "ctl")
        if n and k:
            v.append(abs(k / n - 1))
    return max(v) if v else None


def joules(rec):
    e = rec.get("energy_j_counter")
    if e is None:
        e = rec.get("energy_j_integrated")
    return e


def energy_cell(j, cell, arm):
    recs = j["energy"].get((cell, arm))
    if not recs:
        return None
    r = recs[-1]
    e = joules(r)
    if e is None or not r.get("calls"):
        return None
    return {"j_per_call": e / r["calls"], "watts": r.get("watts_mean") if r.get("watts_mean") is not None else
            e / r["wall_s"], "calls_per_s": r["calls"] / r["wall_s"], "wall_s": r["wall_s"], "calls": r["calls"],
            "source": "counter" if r.get("energy_j_counter") is not None else "integrated",
            "moved": r.get("server_counters_moved"), "n_samples": r.get("n_samples")}


def idle_watts(j):
    v = [r["watts_mean"] for r in j["idle"] if r.get("watts_mean") is not None]
    return st.mean(v) if v else None


def fmt_ratio(x):
    return "   n/a" if x is None else f"{x:6.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ours", help="only this arm of ours (default: every arm of role new / new_alt)")
    ap.add_argument("--vendor", help="only this vendor arm (default: every arm of role vendor / vendor_alt)")
    ap.add_argument("--allow-contaminated", action="store_true",
                    help="do not exclude cells during which the co-resident server's counters moved")
    ap.add_argument("--json", help="also write the per-cell results to this file")
    a = ap.parse_args()
    jobs = load(a.files)
    if not jobs:
        print("no measurement blocks found")
        return 1
    out = []
    targets = collections.OrderedDict()
    for key, j in jobs.items():
        targets.setdefault(key[0], []).append((key, j))
    for target, lst in targets.items():
        print(f"\n=== {target} ===")
        for key, j in lst:
            h = j["heads"][0]
            nz = noise(j)
            cyc = [ncyc(j, c, x) for c in j["cells"] for x in ("new", "ctl")]
            cont = [c for c, r in j["cells"].items() if r.get("server_counters_moved")]
            unk = [c for c, r in j["cells"].items() if r.get("server_counters_moved") is None]
            gate = [c for c, r in j["cells"].items() if not r.get("gate_pass")]
            iw = idle_watts(j)
            vers = {json.dumps(x.get("versions"), sort_keys=True) for x in j["heads"]}
            print(f"job {key[1]} seed {key[2]}: {len(j['files'])} block(s), {len(j['cells'])} cells done, "
                  f"{len(j['errors'])} cell errors; cycles per arm min {min(cyc) if cyc else 0}; "
                  f"noise bound (max |ctl/new-1|) {'n/a' if nz is None else format(nz, '.2%')}; ws {h.get('ws_mib')} MiB; "
                  f"graph replay {h.get('graph_replay')}; idle {'n/a' if iw is None else format(iw, '.1f') + ' W'}")
            if len(vers) > 1:
                print("  WARNING: software versions differ between the blocks of this job")
            if gate:
                print(f"  correctness gate FAILED in {len(gate)} cell(s): {', '.join(gate[:8])}")
            if cont:
                print(f"  contaminated (server counters moved) in {len(cont)} cell(s): {', '.join(cont[:8])}")
            if unk:
                print(f"  server counters unreadable in {len(unk)} cell(s): contamination unknown")
            for e in j["errors"][:5]:
                print(f"  cell error {e.get('cell')}: {str(e.get('error'))[:120]}")
        cells = []
        for key, j in lst:
            for c in j["order"]:
                if c not in cells:
                    cells.append(c)
        ours = sorted({x for _, j in lst for x in arms_of(j, ("new", "new_alt"))}, key=lambda x: (x != "new", x))
        vend = sorted({x for _, j in lst for x in arms_of(j, ("vendor", "vendor_alt"))})
        if a.ours:
            ours = [x for x in ours if x == a.ours]
        if a.vendor:
            vend = [x for x in vend if x == a.vendor]
        dirs = {key[1] for key, _ in lst}
        for o in ours:
            for v in vend:
                print(f"\n-- {v} time / {o} time (higher: ours faster); per job: ratio (net, ctl, cyc min..max) verdict")
                tally = collections.Counter()
                ratios = []
                for c in cells:
                    parts, verdicts, rs, flags = [], [], [], []
                    for key, j in lst:
                        mo, mv = med(j, c, o), med(j, c, v)
                        if mo is None or mv is None:
                            parts.append(f"{tag(key)} missing")
                            verdicts.append("?")
                            continue
                        nz = noise(j) or 0.0
                        r = mv / mo
                        ms = med(j, c, "sep")
                        net = (mv - ms) / (mo - ms) if ms is not None and mo > ms else None
                        ct = med(j, c, "ctl")
                        co, cv = dict(j["time"][(c, o)]), dict(j["time"][(c, v)])
                        rr = [cv[i] / co[i] for i in co if i in cv]
                        w = "W" if r - 1 > nz else ("L" if r - 1 < -nz else "-")
                        info = j["cells"].get(c, {})
                        if not info.get("gate_pass"):
                            flags.append("gate")
                        if info.get("server_counters_moved") and not a.allow_contaminated:
                            flags.append("contaminated")
                        if min(ncyc(j, c, o), ncyc(j, c, v)) < 9:
                            flags.append("cycles<9")
                        verdicts.append(w)
                        rs.append(r)
                        parts.append(f"{tag(key)} {r:6.3f} (net {fmt_ratio(net).strip()}, ctl "
                                     f"{'n/a' if not ct else format(ct / med(j, c, 'new') - 1, '+.2%')}, cyc "
                                     f"{min(rr):.3f}..{max(rr):.3f}) {w}")
                        out.append({"target": target, "cell": c, "ours": o, "vendor": v, "job": key[1], "ratio": r,
                                    "net_ratio": net, "ours_us": mo, "vendor_us": mv, "noise": nz, "verdict": w,
                                    "cycle_ratio_min": min(rr), "cycle_ratio_max": max(rr), "flags": list(flags)})
                    if flags:
                        final = "NOT CERTIFIED (" + ",".join(sorted(set(flags))) + ")"
                    elif "?" in verdicts:
                        final = "incomplete"
                    elif len(dirs) < 2:
                        final = "single job, not certified (" + "".join(verdicts) + ")"
                    elif all(x == "W" for x in verdicts):
                        final = "KEEP: both-win > noise"
                    elif all(x == "L" for x in verdicts):
                        final = "BOTH-LOSE > noise"
                    else:
                        final = "inside noise / mixed"
                    tally[final.split(":")[0].split(" (")[0]] += 1
                    if rs:
                        ratios.append((min(rs), max(rs)))
                    rng = f"range {min(rs):.3f}..{max(rs):.3f}" if rs else "range n/a"
                    print(f" {c:<28s} " + " | ".join(parts) + f"  {rng}  {final}")
                if ratios:
                    lo, hi = min(x for x, _ in ratios), max(y for _, y in ratios)
                    mid = st.median([(x + y) / 2 for x, y in ratios])
                    print(f" summary {v}/{o}: {len(ratios)} cells, ratio {lo:.3f}..{hi:.3f}, median {mid:.3f}; "
                          + ", ".join(f"{k}: {n}" for k, n in tally.items()))
        # ---- energy
        have = any(j["energy"] for _, j in lst)
        if not have:
            print("\n-- energy: no energy records")
            continue
        print("\n-- energy per call (separate tight-loop pass; whole-GPU power, co-resident process included)")
        print("   cell / job / arm: J per call, average W, calls/s, net J per call (idle subtracted, derived), source")
        for c in cells:
            for key, j in lst:
                en = {x: energy_cell(j, c, x) for x in j["role"]}
                en = {x: e for x, e in en.items() if e}
                if not en:
                    continue
                iw = idle_watts(j)
                for x in sorted(en, key=lambda x: (j["role"][x] not in ("new",), x)):
                    e = en[x]
                    net = (e["watts"] - iw) * e["wall_s"] / e["calls"] if iw is not None else None
                    print(f" {c:<28s} {tag(key)} {x:<8s} {e['j_per_call'] * 1e3:10.4f} mJ  {e['watts']:7.1f} W  "
                          f"{e['calls_per_s']:10.1f} /s  net {'n/a' if net is None else format(net * 1e3, '.4f') + ' mJ'}"
                          f"  {e['source']}{'  CONTAMINATED' if e['moved'] else ''}"
                          f"{'  counters unreadable' if e['moved'] is None else ''}")
                    out.append({"target": target, "cell": c, "job": key[1], "arm": x, "energy": e,
                                "net_j_per_call": net, "idle_watts": iw})
                if "new" in en and "ctl" in en:
                    en_noise = abs(en["ctl"]["j_per_call"] / en["new"]["j_per_call"] - 1)
                    line = f" {c:<28s} {tag(key)} energy noise |ctl/new-1| {en_noise:.2%}"
                    for v in vend:
                        if v in en:
                            line += f"; {v} J / new J = {en[v]['j_per_call'] / en['new']['j_per_call']:.3f}"
                    print(line)
        for key, j in lst:
            v = []
            for c in j["cells"]:
                n, k = energy_cell(j, c, "new"), energy_cell(j, c, "ctl")
                if n and k:
                    v.append(abs(k["j_per_call"] / n["j_per_call"] - 1))
            if v:
                print(f" job {key[1]}: energy noise bound (max |J(ctl)/J(new)-1| over {len(v)} cells) {max(v):.2%}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
