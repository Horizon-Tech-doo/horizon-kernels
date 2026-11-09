#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Expert pruning of a Qwen3.8-Flash-Next (Qwen4Exp) checkpoint, directly on its NVFP4 safetensors (03-10-2026).

No requantization: every kept tensor is copied byte for byte (NVFP4 weights, their E4M3 block scales, the FP32 global
scales `weight_scale_2`, the static `input_scale`s, FP8-block MTP experts with `weight_scale_inv`). Dropped experts
lose all of their tensors. The router (`mlp.gate.weight`, [num_experts, hidden]) keeps the rows of the kept experts,
in the original order, and the kept experts are renumbered 0..N-1 in that same order, so router row i and expert i
stay paired. Streaming, shard by shard; peak RAM is one 64 MiB copy buffer.

Why uniform: vLLM 0.30's Qwen4Exp builds every MoE block from the one `text_config.num_experts`
(Qwen3NextSparseMoeBlock -> ReplicatedLinear(hidden, num_experts) + FusedMoE(num_experts=...)), and the MTP layer uses
the same text config. So every MoE block (48 main + the MTP layer) keeps the same COUNT N; WHICH experts are kept is
decided per layer. A non-uniform allocation is reported (and can be emitted as a runtime mask for quality screens with
horizon_flashnext's mask plugin) but can't be loaded as a smaller checkpoint without a vLLM patch.
The router has no bias in vLLM (bias=False, and '.bias' checkpoint tensors are silently ignored), so "masking with a
-inf bias" is not possible through the checkpoint either; the mask plugin does it at runtime instead.

Usage
  prune_checkpoint.py --src DIR --out DIR --scores scores.npz --prune-ratio 0.35 [--criterion reap]
  prune_checkpoint.py --src DIR --out DIR --keep-file keep.json          # reuse a selection (e.g. NVIDIA -> lil)
  prune_checkpoint.py --src DIR --scores scores.npz --prune-ratio 0.5 --dry-run --emit-keep keep50.json
Options: --round 8 (kept count rounded to a multiple), --mtp score|drop, --min-retained 0.0 --strict,
         --emit-mask mask.npz (uniform + non-uniform suggestion), --verify none|sample|full, --license-file PATH.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn_common as fc  # noqa: E402

CRITERIA = ("reap", "ean", "freq", "gate")
PER_EXPERT_QUANT_RE = re.compile(r"experts\.\d+")


def sha256_file(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                return h.hexdigest()
            h.update(b)


def sha256_chunks(chunks, bufsize=16 << 20):
    h = hashlib.sha256()
    for (p, start, n) in chunks:
        with open(p, "rb") as f:
            f.seek(start)
            left = n
            while left:
                b = f.read(min(bufsize, left))
                if not b:
                    raise IOError(f"{p}: short read")
                h.update(b)
                left -= len(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# scores -> selection
# ---------------------------------------------------------------------------------------------------------------------

def load_scores(path):
    """scores.npz from reap_calibrate.py: '<prefix>|count', '|gsum', '|reap_sum', '|ean_sum' per MoE prefix."""
    import numpy as np  # noqa: PLC0415
    z = np.load(path, allow_pickle=False)
    out: dict = {}
    for k in z.files:
        if "|" not in k:
            continue
        prefix, stat = k.rsplit("|", 1)
        out.setdefault(prefix, {})[stat] = np.asarray(z[k], dtype=np.float64)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return out, meta


def criterion_values(st, criterion):
    import numpy as np  # noqa: PLC0415
    cnt = st["count"]
    safe = np.where(cnt > 0, cnt, 1.0)
    if criterion == "reap":     # REAP (Lasby et al. 2025): mean over routed tokens of gate * ||expert output||
        v = st["reap_sum"] / safe
    elif criterion == "ean":    # expert activation norm, mean over routed tokens
        v = st["ean_sum"] / safe
    elif criterion == "freq":
        v = cnt.copy()
    elif criterion == "gate":   # total renormalized gate mass
        v = st["gsum"].copy()
    else:
        raise ValueError(criterion)
    return np.where(cnt > 0, v, 0.0)


def select_uniform(scores, n_keep, criterion):
    """{prefix: sorted kept original ids}. Ties: higher score, then more tokens, then lower id."""
    import numpy as np  # noqa: PLC0415
    keep = {}
    for prefix, st in scores.items():
        v = criterion_values(st, criterion)
        order = np.lexsort((np.arange(len(v)), -st["count"], -v))  # last key is primary
        keep[prefix] = sorted(int(i) for i in order[:n_keep])
    return keep


def retained(st, kept):
    """Share of the layer's calibration signal that the kept experts carry."""
    import numpy as np  # noqa: PLC0415
    idx = np.array(kept, dtype=np.int64)
    out = {}
    for name, key in (("reap_mass", "reap_sum"), ("tokens", "count"), ("gate_mass", "gsum")):
        tot = float(st[key].sum())
        out[name] = round(float(st[key][idx].sum()) / tot, 4) if tot > 0 else None
    out["unrouted_kept"] = int((st["count"][idx] == 0).sum())
    out["unrouted_total"] = int((st["count"] == 0).sum())
    return out


def nonuniform_suggestion(scores, n_keep, prefixes, min_keep):
    """Report-only allocation with the same total as uniform: experts ranked across layers by their share of their
    own layer's REAP mass (sum of gate*||out|| over tokens); each layer keeps at least min_keep."""
    import numpy as np  # noqa: PLC0415
    budget = n_keep * len(prefixes)
    alloc = {}
    cand = []
    for p in prefixes:
        st = scores[p]
        mass = st["reap_sum"]
        share = mass / mass.sum() if mass.sum() > 0 else np.zeros_like(mass)
        order = list(np.lexsort((np.arange(len(share)), -st["count"], -share)))
        alloc[p] = set(int(i) for i in order[:min_keep])
        cand += [(float(share[i]), p, int(i)) for i in order[min_keep:]]
    left = budget - sum(len(v) for v in alloc.values())
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    for s, p, i in cand[:max(0, left)]:
        alloc[p].add(i)
    return {p: sorted(v) for p, v in alloc.items()}


# ---------------------------------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------------------------------

class Plan:
    def __init__(self, src, keep, num_experts, mtp_policy):
        self.src = src
        self.keep = keep                    # {prefix: sorted original ids}
        self.num_experts = num_experts
        self.mtp_policy = mtp_policy
        self.newid = {p: {o: n for n, o in enumerate(ids)} for p, ids in keep.items()}
        self.warnings = []

    def drop_mtp(self, name):
        return self.mtp_policy == "drop" and name.startswith("mtp.")

    def map_tensor(self, ti: fc.TensorInfo):
        """-> None (drop) or OutTensor."""
        name = ti.name
        if self.drop_mtp(name):
            return None
        m = fc.EXPERT_RE.match(name)
        if m:
            p = m["prefix"]
            if p not in self.keep:
                raise SystemExit(f"MoE block {p} has no selection (scores or keep file); refusing to copy it unpruned")
            new = self.newid[p].get(int(m["eid"]))
            if new is None:
                return None
            return fc.OutTensor(f"{p}.experts.{new}.{m['rest']}", ti.dtype, ti.shape, [(ti.path, ti.start, ti.nbytes)])
        m = fc.FUSED_EXPERT_RE.match(name) or fc.GATE_RE.match(name)
        if m and m["prefix"] in self.keep:
            if ti.shape and ti.shape[0] == self.num_experts:
                return self.slice_rows(ti, self.keep[m["prefix"]])
            self.warnings.append(f"{name}: shape {ti.shape} has no expert dim 0 ({self.num_experts}); copied unchanged")
        return fc.OutTensor(name, ti.dtype, ti.shape, [(ti.path, ti.start, ti.nbytes)])

    @staticmethod
    def slice_rows(ti, rows):
        row = ti.nbytes // ti.shape[0]
        if row * ti.shape[0] != ti.nbytes:
            raise SystemExit(f"{ti.name}: {ti.nbytes} bytes do not split into {ti.shape[0]} rows")
        chunks = []
        start = prev = None
        for r in rows:
            if start is None:
                start = prev = r
            elif r == prev + 1:
                prev = r
            else:
                chunks.append((ti.path, ti.start + start * row, (prev - start + 1) * row))
                start = prev = r
        if start is not None:
            chunks.append((ti.path, ti.start + start * row, (prev - start + 1) * row))
        return fc.OutTensor(ti.name, ti.dtype, [len(rows)] + list(ti.shape[1:]), chunks)


def check_quant_configs(src):
    """Per-expert names in a quantization config would need renumbering; both NVFP4 sources list `...mlp.experts`
    modules only (checked 03-10-2026). Refuse rather than silently write a wrong config."""
    for fn in ("hf_quant_config.json", "config.json"):
        p = os.path.join(src, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        q = d.get("quantization_config", d) if fn == "config.json" else d
        txt = json.dumps(q)
        if PER_EXPERT_QUANT_RE.search(txt):
            raise SystemExit(f"{fn}: the quantization config names single experts ('experts.<n>'); this tool does not "
                             "renumber those entries. Extend check_quant_configs before pruning this checkpoint.")


def build(args):
    src = args.src
    cfg = json.load(open(os.path.join(src, "config.json")))
    tc = fc.text_config(cfg)
    E = int(tc["num_experts"])
    top_k = int(tc.get("num_experts_per_tok", 0))
    nhl = int(tc["num_hidden_layers"])
    wmap, imeta = fc.load_index(src)
    layout = fc.expert_layout(wmap.keys())
    moe_prefixes = sorted(layout, key=lambda p: (p.startswith("mtp."), int(re.search(r"layers\.(\d+)", p).group(1))))
    for p in moe_prefixes:
        eids = layout[p]["eids"]
        if eids and eids != set(range(E)):
            raise SystemExit(f"{p}: expert ids {min(eids)}..{max(eids)} ({len(eids)}) do not match num_experts={E}")
    mtp_prefixes = [p for p in moe_prefixes if p.startswith("mtp.")]
    main_prefixes = [p for p in moe_prefixes if not p.startswith("mtp.")]
    if len(main_prefixes) != sum(1 for i in range(nhl)):
        print(f"note: {len(main_prefixes)} MoE blocks in the main model, num_hidden_layers={nhl}", file=sys.stderr)

    report = {"src": os.path.abspath(src), "num_experts_src": E, "top_k": top_k, "moe_blocks": moe_prefixes}
    if args.keep_file:
        kf = json.load(open(args.keep_file))
        if int(kf.get("num_experts_src", E)) != E:
            raise SystemExit(f"keep file is for num_experts={kf['num_experts_src']}, checkpoint has {E}")
        keep = {p: sorted(int(i) for i in v) for p, v in kf["keep"].items()}
        n_keep = len(next(iter(keep.values())))
        report["selection"] = {"from_keep_file": os.path.abspath(args.keep_file), "keep_file_meta": kf.get("meta", {})}
        scores = None
    else:
        if not args.scores:
            raise SystemExit("need --scores or --keep-file")
        scores, smeta = load_scores(args.scores)
        if args.keep_experts:
            n_keep = args.keep_experts
        else:
            raw = E * (1.0 - args.prune_ratio)
            n_keep = int(round(raw / args.round)) * args.round if args.round > 1 else int(round(raw))
        keep = select_uniform({p: scores[p] for p in moe_prefixes if p in scores}, n_keep, args.criterion)
        report["selection"] = {"scores": os.path.abspath(args.scores), "scores_sha256": sha256_file(args.scores),
                               "criterion": args.criterion, "prune_ratio_requested": args.prune_ratio,
                               "scores_meta": smeta}
    if not (top_k <= n_keep <= E):
        raise SystemExit(f"kept experts {n_keep} must be within [top_k={top_k}, num_experts={E}]")

    missing = [p for p in moe_prefixes if p not in keep]
    missing_mtp = [p for p in missing if p.startswith("mtp.")]
    missing_main = [p for p in missing if not p.startswith("mtp.")]
    if missing_main:
        raise SystemExit(f"no selection for main MoE blocks: {missing_main[:4]}{'...' if len(missing_main) > 4 else ''}")
    if missing_mtp and args.mtp == "score":
        raise SystemExit(f"no scores for the MTP MoE block(s) {missing_mtp}: calibrate with --mtp (drafter hooks), "
                         "or prune with --mtp drop (serve without MTP speculative decoding)")
    if args.mtp == "drop":
        keep = {p: v for p, v in keep.items() if not p.startswith("mtp.")}
    bad = {p: len(v) for p, v in keep.items() if len(v) != n_keep or len(set(v)) != len(v)}
    if bad:
        raise SystemExit(f"selection is not uniform at {n_keep} experts per block: {list(bad.items())[:4]}")
    for p, v in keep.items():
        if min(v) < 0 or max(v) >= E:
            raise SystemExit(f"{p}: expert id out of range in selection")

    report.update(num_experts_out=n_keep, pruned_fraction=round(1 - n_keep / E, 4), mtp_policy=args.mtp)
    per_layer = {}
    if scores is not None:
        for p in keep:
            if p in scores:
                per_layer[p] = retained(scores[p], keep[p])
        low = {p: r for p, r in per_layer.items() if r["reap_mass"] is not None and r["reap_mass"] < args.min_retained}
        report["low_retention_layers"] = sorted(low)
        if low:
            msg = (f"{len(low)} block(s) keep less than {args.min_retained:.0%} of their REAP mass: "
                   + ", ".join(f"{p}={r['reap_mass']}" for p, r in sorted(low.items())[:6]))
            print("WARNING: " + msg, file=sys.stderr)
            if args.strict:
                raise SystemExit(msg)
    report["per_layer"] = per_layer
    return cfg, tc, E, n_keep, wmap, imeta, keep, scores, report, main_prefixes


def write_keep_and_mask(args, E, n_keep, keep, scores, main_prefixes, report):
    if args.emit_keep:
        json.dump({"num_experts_src": E, "num_experts_out": n_keep, "keep": keep,
                   "meta": {k: report[k] for k in ("src", "selection", "pruned_fraction", "mtp_policy")}},
                  open(args.emit_keep, "w"), indent=1)
        print(f"keep list -> {args.emit_keep}")
    if args.emit_mask:
        import numpy as np  # noqa: PLC0415
        arrays = {"meta": np.array(json.dumps({"num_experts": E, "kind": "keep-mask", "uniform_keep": n_keep}))}
        for p, ids in keep.items():
            m = np.zeros(E, dtype=bool)
            m[ids] = True
            arrays[f"{p}|uniform"] = m
        if scores is not None:
            nu = nonuniform_suggestion(scores, n_keep, [p for p in main_prefixes if p in scores],
                                       min(args.nonuniform_min, n_keep))
            sizes = {}
            for p, ids in nu.items():
                m = np.zeros(E, dtype=bool)
                m[ids] = True
                arrays[f"{p}|nonuniform"] = m
                sizes[p] = len(ids)
            report["nonuniform_suggestion"] = {"min_per_block": min(args.nonuniform_min, n_keep),
                                               "sizes": sizes,
                                               "retained_reap_mass": {p: retained(scores[p], nu[p])["reap_mass"]
                                                                      for p in nu}}
        np.savez(args.emit_mask, **arrays)
        print(f"masks -> {args.emit_mask}")


def run(args):
    cfg, tc, E, n_keep, wmap, imeta, keep, scores, report, main_prefixes = build(args)
    write_keep_and_mask(args, E, n_keep, keep, scores, main_prefixes, report)
    check_quant_configs(args.src)
    plan = Plan(args.src, keep, E, args.mtp)

    shards = sorted(set(wmap.values()))
    src_bytes = out_bytes = 0
    out_shards = {}
    for sh in shards:
        tensors, _, meta = fc.read_header(os.path.join(args.src, sh))
        listed = {n for n, s in wmap.items() if s == sh}
        if set(tensors) != listed:
            extra, gone = set(tensors) - listed, listed - set(tensors)
            raise SystemExit(f"{sh}: index and shard disagree (+{len(extra)} / -{len(gone)} tensors)")
        outs = []
        for ti in sorted(tensors.values(), key=lambda t: t.start):
            src_bytes += ti.nbytes
            o = plan.map_tensor(ti)
            if o is not None:
                outs.append(o)
                out_bytes += o.nbytes
        out_shards[sh] = (outs, meta)
    for w in plan.warnings:
        print("WARNING: " + w, file=sys.stderr)
    report.update(bytes_src=src_bytes, bytes_out=out_bytes, gib_saved=round((src_bytes - out_bytes) / 2**30, 2))
    print(f"experts per MoE block {E} -> {n_keep}; tensor bytes {src_bytes / 2**30:.2f} -> {out_bytes / 2**30:.2f} GiB"
          f" ({(src_bytes - out_bytes) / 2**30:.2f} GiB saved), {len(keep)} MoE block(s) pruned, MTP: {args.mtp}")
    if args.dry_run:
        if args.report:
            json.dump(report, open(args.report, "w"), indent=1)
        return report

    if not args.out:
        raise SystemExit("--out is required unless --dry-run")
    if os.path.abspath(args.out) == os.path.abspath(args.src):
        raise SystemExit("--out must differ from --src")
    os.makedirs(args.out, exist_ok=True)
    if any(f.endswith(".safetensors") for f in os.listdir(args.out)) and not args.overwrite:
        raise SystemExit(f"{args.out} already has safetensors files (use --overwrite)")

    new_map = {}
    total = 0
    for i, sh in enumerate(shards):
        outs, meta = out_shards[sh]
        if not outs:
            continue
        n = fc.write_safetensors(os.path.join(args.out, sh), outs, meta or {"format": "pt"})
        for o in outs:
            new_map[o.name] = sh
            total += o.nbytes
        print(f"  [{i + 1}/{len(shards)}] {sh}: {len(outs)} tensors, {n / 2**30:.2f} GiB", flush=True)
    idx_meta = dict(imeta)
    idx_meta["total_size"] = total
    json.dump({"metadata": idx_meta, "weight_map": dict(sorted(new_map.items()))},
              open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=2)

    # config: the one uniform expert count (+ provenance)
    tc["num_experts"] = n_keep
    if "text_config" in cfg and "num_experts" in cfg:
        cfg["num_experts"] = n_keep
    if args.mtp == "drop":
        tc["mtp_num_hidden_layers"] = 0
        if isinstance(tc.get("mtp"), dict):
            tc["mtp"]["num_hidden_layers"] = 0
    cfg["horizon_pruning"] = {
        "by": "Horizon Tech d.o.o.", "date": datetime.date.today().strftime("%d-%m-%Y"),
        "method": "expert pruning (REAP-style saliency), router rows sliced, experts renumbered in original order",
        "num_experts_src": E, "num_experts_out": n_keep, "mtp_policy": args.mtp,
        "details": "horizon_pruning.json",
    }
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    for root, _, files in os.walk(args.src):
        rel = os.path.relpath(root, args.src)
        for f in files:
            if f.endswith(".safetensors") or f in ("model.safetensors.index.json", "config.json") or ".partial" in f:
                continue
            dst_dir = os.path.join(args.out, rel)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(os.path.join(root, f), os.path.join(dst_dir, f))
    if not os.path.exists(os.path.join(args.out, "LICENSE")):
        if args.license_file:
            shutil.copy2(args.license_file, os.path.join(args.out, "LICENSE"))
        else:
            print("WARNING: no LICENSE in the source and no --license-file: the Qwen Community License requires its "
                  "notice in every copy of the model or its derivatives", file=sys.stderr)
    readme = os.path.join(args.src, "README.md")
    nvidia_oml = os.path.exists(readme) and "nvidia-open-model-license" in open(readme, errors="ignore").read().lower()
    with open(os.path.join(args.out, "NOTICE"), "w") as f:
        if nvidia_oml:   # NVIDIA Open Model License: required attribution in a "Notice" file of every copy
            f.write("Licensed by NVIDIA Corporation under the NVIDIA Open Model License\n\n")
        f.write(f"This model is a derivative work of {cfg.get('_name_or_path') or report['src']} (Qwen3.8-Flash-Next),\n"
                f"modified by Horizon Tech d.o.o. on {cfg['horizon_pruning']['date']}: routed experts pruned from {E} "
                f"to {n_keep} per MoE block\n(router rows sliced, experts renumbered, no requantization). "
                "Details: horizon_pruning.json.\nUse is subject to the Qwen Community License 1.0 (LICENSE), and to "
                "the source checkpoint's own licence terms where it adds any.\n")
    report["keep"] = keep
    json.dump(report, open(os.path.join(args.out, "horizon_pruning.json"), "w"), indent=1)
    if args.report:
        json.dump(report, open(args.report, "w"), indent=1)
    if args.verify != "none":
        verify(args.src, args.out, keep, E, n_keep, args.mtp, full=args.verify == "full", seed=args.seed)
    return report


# ---------------------------------------------------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------------------------------------------------

def _infos(model_dir):
    wmap, _ = fc.load_index(model_dir)
    infos = {}
    for sh in sorted(set(wmap.values())):
        t, _, _ = fc.read_header(os.path.join(model_dir, sh))
        infos.update(t)
    if set(infos) != set(wmap):
        raise AssertionError(f"{model_dir}: index and shards disagree")
    return infos


def verify(src, out, keep, E, n_keep, mtp_policy, full=False, seed=0):
    """Re-read both checkpoints. Structure: every pruned block has experts 0..N-1 with the source's tensor set; no
    other tensor changed name, dtype, shape. Content: hashes of every router and of all (full) or ~3% (sample) of the
    expert tensors and ~3% of the other tensors."""
    rnd = random.Random(seed)
    si, oi = _infos(src), _infos(out)
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert int(fc.text_config(cfg)["num_experts"]) == n_keep, "config num_experts"
    expected = {}
    for name, ti in si.items():
        if mtp_policy == "drop" and name.startswith("mtp."):
            continue
        m = fc.EXPERT_RE.match(name)
        if m and m["prefix"] in keep:
            ids = keep[m["prefix"]]
            if int(m["eid"]) in ids:
                expected[f"{m['prefix']}.experts.{ids.index(int(m['eid']))}.{m['rest']}"] = ("copy", ti)
            continue
        m = fc.FUSED_EXPERT_RE.match(name) or fc.GATE_RE.match(name)
        if m and m["prefix"] in keep and ti.shape and ti.shape[0] == E:
            expected[name] = ("rows", ti, keep[m["prefix"]])
            continue
        expected[name] = ("copy", ti)
    if set(expected) != set(oi):
        a, b = set(expected) - set(oi), set(oi) - set(expected)
        raise AssertionError(f"tensor set mismatch: missing {sorted(a)[:5]} unexpected {sorted(b)[:5]}")
    checked = 0
    for name, exp in sorted(expected.items()):
        o = oi[name]
        ti = exp[1]
        if o.dtype != ti.dtype:
            raise AssertionError(f"{name}: dtype {o.dtype} != {ti.dtype}")
        if exp[0] == "copy":
            if o.shape != ti.shape or o.nbytes != ti.nbytes:
                raise AssertionError(f"{name}: shape {o.shape} != {ti.shape}")
            is_exp = bool(fc.EXPERT_RE.match(name))
            if full or rnd.random() < 0.03 or (not is_exp and o.nbytes < (4 << 20) and rnd.random() < 0.2):
                if sha256_chunks([(o.path, o.start, o.nbytes)]) != sha256_chunks([(ti.path, ti.start, ti.nbytes)]):
                    raise AssertionError(f"{name}: bytes differ")
                checked += 1
        else:
            rows = exp[2]
            if o.shape != [len(rows)] + ti.shape[1:]:
                raise AssertionError(f"{name}: shape {o.shape}")
            row = ti.nbytes // ti.shape[0]
            want = Plan.slice_rows(ti, rows).chunks
            if sha256_chunks([(o.path, o.start, o.nbytes)]) != sha256_chunks(want) or o.nbytes != len(rows) * row:
                raise AssertionError(f"{name}: sliced rows differ")
            checked += 1
    for p in keep:
        eids = {int(fc.EXPERT_RE.match(n)["eid"]) for n in oi if fc.EXPERT_RE.match(n) and
                fc.EXPERT_RE.match(n)["prefix"] == p}
        if eids and eids != set(range(n_keep)):
            raise AssertionError(f"{p}: output experts are not 0..{n_keep - 1}")
    print(f"verify OK: {len(oi)} tensors, structure checked, {checked} hashed ({'full' if full else 'sample'})")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--scores")
    g.add_argument("--keep-file")
    ap.add_argument("--prune-ratio", type=float, default=0.35)
    ap.add_argument("--keep-experts", type=int, default=0, help="exact kept count (overrides --prune-ratio)")
    ap.add_argument("--round", type=int, default=8, help="round the kept count to a multiple of this")
    ap.add_argument("--criterion", choices=CRITERIA, default="reap")
    ap.add_argument("--mtp", choices=("score", "drop"), default="score")
    ap.add_argument("--min-retained", type=float, default=0.0, help="warn when a block keeps less REAP mass")
    ap.add_argument("--strict", action="store_true", help="fail instead of warning on --min-retained")
    ap.add_argument("--nonuniform-min", type=int, default=64, help="min experts per block in the non-uniform report")
    ap.add_argument("--emit-keep")
    ap.add_argument("--emit-mask")
    ap.add_argument("--report")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", choices=("none", "sample", "full"), default="sample")
    ap.add_argument("--license-file")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
