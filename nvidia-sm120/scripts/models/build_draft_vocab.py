# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# MTP plan, stage 2: build the draft vocabulary from the model's own answers (gen_v2.jsonl).
# Tokens are counted exactly as the model emits them (thinking block, content, end token); frequency ranking on a 90%
# split, coverage measured on the held-out 10% (overall, per language, per task) for several vocabulary sizes.
# Special/added tokens (tool calls, think tags, end tokens) are always included.
#   build_draft_vocab.py [SIZES=16384,24576,32768,49152,65536] [GEN=$HK_DATA/models/mtp-data/gen_v2.jsonl]
import collections, json, os, random, sys, time
import torch
from transformers import AutoTokenizer

MODEL = "$HK_DATA/models/Qwen3.8-27B-NVFP4"
SIZES = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "16384,24576,32768,49152,65536").split(",")]
OUT_SUB = sys.argv[3] if len(sys.argv) > 3 else "horizon_mtp"   # e.g. horizon_mtp_v3
GEN = sys.argv[2] if len(sys.argv) > 2 else "$HK_DATA/models/mtp-data/gen_v2.jsonl"
OUTDIR = f"{MODEL}/{OUT_SUB}"
os.makedirs(OUTDIR, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)


def emitted(r):
    if r.get("gen_text"):                      # data v3: exact emitted text incl. tool-call markup
        return tok(r["gen_text"], add_special_tokens=False)["input_ids"]
    th = f"<think>\n{r['reasoning'].strip()}\n</think>\n\n" if r["think"] else "<think>\n\n</think>\n\n"
    return tok(th + r["content"].strip() + "<|im_end|>", add_special_tokens=False)["input_ids"]


rows = [json.loads(l) for l in open(GEN)]
random.Random(0).shuffle(rows)
cut = int(len(rows) * 0.9)
train, held = rows[:cut], rows[cut:]
cnt = collections.Counter()
for r in train:
    cnt.update(emitted(r))
special = sorted(set(tok.all_special_ids) | {i for i in tok.get_added_vocab().values()})
ranked = special + [t for t, _ in cnt.most_common() if t not in set(special)]
held_ids = [(r.get("lang", "en"), r.get("kind", "?"), emitted(r)) for r in held]
report = {"date": time.strftime("%d-%m-%Y %H:%M"), "answers": len(rows), "train_tokens": sum(cnt.values()),
          "distinct_tokens_seen": len(cnt), "special_always_included": len(special), "coverage": {}}
for V in SIZES:
    vocab = ranked[:V]
    vs = set(vocab)
    tot = cov = 0
    by = collections.defaultdict(lambda: [0, 0])
    for lang, kind, ids in held_ids:
        c = sum(1 for t in ids if t in vs)
        tot += len(ids); cov += c
        for key in (f"lang:{lang}", f"kind:{kind}"):
            by[key][0] += c; by[key][1] += len(ids)
    report["coverage"][V] = {"all": round(cov / max(1, tot), 5), **{k: round(a / max(1, b), 5) for k, (a, b) in sorted(by.items())}}
    torch.save(torch.tensor(vocab, dtype=torch.long), f"{OUTDIR}/draft_vocab_{V}.pt")
json.dump(report, open(f"{OUTDIR}/draft_vocab_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
