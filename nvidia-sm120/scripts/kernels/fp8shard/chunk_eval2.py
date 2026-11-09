# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# chunk_eval2.py TAG DOCS (fp8fix, 27-09-2026): per-token prompt log-probs of logprob_eval.py's prose / code documents
# (same texts, same request), saved as ~/logprob_eval/TAG.json with the character span of every scored token, so arms
# of models with different tokenizers can be compared in nats per character.
#   LP_MODULE=logprob_eval (Qwen3.8 tokenizer, default) | logprob_eval_moe (HORIZON_LP_TOKENIZER, the MoE's tokenizer)
#   DOCS: comma list of prose_<L> / code_<L>, L in 1024, 4096, 16384, 32768
import glob, importlib, json, os, sys, time
sys.path.insert(0, os.path.expanduser("~"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module(os.environ.get("LP_MODULE", "logprob_eval"))
from transformers import AutoTokenizer  # noqa: E402

tokp = (os.environ.get("HORIZON_LP_TOKENIZER", "$HK_DATA/models/nvidia-Qwen3.6-35B-A3B-NVFP4")
        if L.__name__.endswith("_moe") else "$HK_DATA/models/Qwen3.8-27B-NVFP4")
tok = AutoTokenizer.from_pretrained(tokp)
V = "$HK_VENV/lib/python3.12/site-packages/vllm"
texts = {"prose": open(os.path.expanduser("$HK_WORK/corpus/prose.md"), encoding="utf-8", errors="ignore").read(),
         "code": "\n\n".join(open(f).read() for f in sorted(glob.glob(f"{V}/v1/spec_decode/*.py")
                                                              + glob.glob(f"{V}/v1/worker/gpu/*.py")
                                                              + glob.glob(f"{V}/model_executor/layers/*.py")))}
tag, names = sys.argv[1], sys.argv[2].split(",")
out = {"_meta": {"tokenizer": tokp, "module": L.__name__}}
for name in names:
    kind, n = name.rsplit("_", 1)
    enc = tok(texts[kind], add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"][:int(n)], enc["offset_mapping"][:int(n)]
    t0 = time.time()
    lp, top1 = L.prompt_logprobs(ids)
    out[name] = {"lp": lp, "top1": top1, "span": [list(o) for o in offs[1:1 + len(lp)]]}
    print(f"{tag} {name}: {len(lp)} tokens, mean NLL {-sum(lp) / len(lp):.4f}, chars {offs[-1][1]} "
          f"({time.time() - t0:.0f} s)", flush=True)
json.dump(out, open(f"{L.OUTDIR}/{tag}.json", "w"))
