"""Real routing and real weights for the N1 MoE harness. Runs Qwen/Qwen3.6-35B-A3B (BF16, transformers forward, two GCDs) over pilot-val
conversations with the chat template, hooks every layer's router, and writes data files the harness reads:
  data_real_L<L>.pt : W13, W2 of layer L as fp16, and (T, 'prefill') / (T, 'decode') -> X, ids, wts of that layer.
prefill(T): T consecutive tokens, conversations joined if one is too short, as one vLLM prefill chunk. decode(T): T tokens at random
positions of distinct conversations, as one decode step of T requests. route_stats.json: per layer routing statistics, no text."""
import json, os, sys, time
import torch
from transformers import AutoConfig, AutoTokenizer
MP, DS, OUT = sys.argv[1:4]
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
LAYERS = (0, 19, 39)
tok = AutoTokenizer.from_pretrained(MP)
cfg = AutoConfig.from_pretrained(MP)
tc = cfg.text_config
print(json.dumps({"kind": "config", "hidden": tc.hidden_size, "moe_intermediate": tc.moe_intermediate_size, "experts": tc.num_experts, "topk": tc.num_experts_per_tok, "layers": tc.num_hidden_layers, "shared_intermediate": tc.shared_expert_intermediate_size}), flush=True)
n = tc.num_hidden_layers
from accelerate import dispatch_model
from safetensors import safe_open
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
dtype = torch.bfloat16 if len(sys.argv) <= 5 else getattr(torch, sys.argv[5])
idx0 = json.load(open(f"{MP}/model.safetensors.index.json"))["weight_map"]
t0 = time.time()
with torch.device("meta"):
    model = Qwen3_5MoeTextModel(tc)
PFX = "model.language_model."
sd = {}
for fn in sorted({v for k, v in idx0.items() if k.startswith(PFX)}):
    with safe_open(f"{MP}/{fn}", "pt") as f:
        for k in f.keys():
            if k.startswith(PFX):
                sd[k[len(PFX):]] = f.get_tensor(k).to(dtype)
miss = model.load_state_dict(sd, strict=False, assign=True)
print(json.dumps({"kind": "state", "missing": list(miss.missing_keys)[:20], "unexpected": list(miss.unexpected_keys)[:20]}), flush=True)
del sd
model.rotary_emb = type(model.rotary_emb)(tc)
dm = {"embed_tokens": 0, "rotary_emb": 0, "norm": 1}
for i in range(n):
    dm[f"layers.{i}"] = 0 if i < n // 2 else 1
model = dispatch_model(model, device_map=dm)
model.eval()
print(json.dumps({"kind": "loaded", "class": type(model).__name__, "s": round(time.time() - t0, 1), "alloc_gb": [round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(2)]}), flush=True)
layers = model.layers
cur = {}
def mk(i):
    def hook(mod, inp, out):
        _, wts, ids = out
        cur[i] = (ids.to("cpu", torch.uint8), wts.float().to("cpu"), inp[0].to(torch.float16).to("cpu") if i in LAYERS else None)
    return hook
for i in range(n):
    layers[i].mlp.gate.register_forward_hook(mk(i))
rows = [json.loads(l) for l in open(DS)]
store = []     # per conversation: {layer: (ids, wts, X)}
cats = {}
t0 = time.time(); tot = 0
with torch.no_grad():
    for r in rows:
        msgs = list(r["messages"]) + [{"role": "assistant", "content": r.get("answer", "")}]
        s = tok.apply_chat_template(msgs, tokenize=False)
        ii = tok(s, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.cuda(0)
        cur.clear()
        model(input_ids=ii, use_cache=False)
        store.append({k: v for k, v in cur.items()})
        cats[r.get("category", "?")] = cats.get(r.get("category", "?"), 0) + 1
        tot += ii.shape[1]
print(json.dumps({"kind": "forward", "conversations": len(rows), "tokens": tot, "s": round(time.time() - t0, 1), "categories": cats}), flush=True)
del model
g = torch.Generator().manual_seed(4)
def blocks(ids, bm):
    h = ids.flatten().long().bincount(minlength=256)
    return int(((h + bm - 1) // bm).sum())
stats = {"tokens": tot, "categories": cats, "layers": {}}
for L in range(n):
    A = torch.cat([c[L][0] for c in store]).long()
    W = torch.cat([c[L][1] for c in store])
    h = A.flatten().bincount(minlength=256).float()
    p = (h / h.sum()).sort(descending=True).values
    entry = dict(top1_share=round(p[0].item(), 4), top8_share=round(p[:8].sum().item(), 4), top32_share=round(p[:32].sum().item(), 4), experts_never=int((h == 0).sum()), weight_top1=round(W[:, 0].mean().item(), 4), by_T={})
    for Tn in (16, 64, 256, 1024, 4096):
        hits, b16, b64, b128 = [], [], [], []
        for _ in range(8):
            s0 = torch.randint(0, A.shape[0] - Tn, (1,), generator=g).item()
            x = A[s0:s0 + Tn]
            hits.append(int((x.flatten().bincount(minlength=256) > 0).sum())); b16.append(blocks(x, 16)); b64.append(blocks(x, 64)); b128.append(blocks(x, 128))
        entry["by_T"][Tn] = dict(hit=sum(hits) / 8, blocks16=sum(b16) / 8, blocks64=sum(b64) / 8, blocks128=sum(b128) / 8)
    stats["layers"][L] = entry
json.dump(stats, open(f"{OUT}/route_stats.json", "w"), indent=0)
print(json.dumps({"kind": "stats", "L0": stats["layers"][0], "L19": stats["layers"][19], "L39": stats["layers"][39]}), flush=True)
idx = json.load(open(f"{MP}/model.safetensors.index.json"))["weight_map"]
for L in LAYERS:
    d = {}
    for nm, key in (("W13", "gate_up_proj"), ("W2", "down_proj")):
        k = f"model.language_model.layers.{L}.mlp.experts.{key}"
        with safe_open(f"{MP}/{idx[k]}", "pt") as f:
            d[nm] = f.get_tensor(k).to(torch.float16)
    Xa = torch.cat([c[L][2] for c in store]); Ia = torch.cat([c[L][0] for c in store]).int(); Wa = torch.cat([c[L][1] for c in store])
    for Tn in (1, 4, 16, 64, 256, 1024, 4096, 8192):
        if Tn <= Xa.shape[0]:
            s0 = torch.randint(0, Xa.shape[0] - Tn, (1,), generator=g).item()
            d[(Tn, "prefill")] = {"X": Xa[s0:s0 + Tn].clone(), "ids": Ia[s0:s0 + Tn].clone(), "wts": Wa[s0:s0 + Tn].clone()}
    # one position per conversation first, then further positions
    pos = [(i, torch.randint(0, store[i][L][0].shape[0], (1,), generator=g).item()) for i in range(len(store))]
    off = [0]
    for c in store: off.append(off[-1] + c[L][0].shape[0])
    while len(pos) < 256:
        i = torch.randint(0, len(store), (1,), generator=g).item()
        pos.append((i, torch.randint(0, store[i][L][0].shape[0], (1,), generator=g).item()))
    fl = torch.tensor([off[i] + p for i, p in pos])
    for Tn in (1, 4, 16, 64, 256):
        ix = fl[:Tn]
        d[(Tn, "decode")] = {"X": Xa[ix].clone(), "ids": Ia[ix].clone(), "wts": Wa[ix].clone()}
    torch.save(d, f"{OUT}/data_real_L{L}.pt")
    print(json.dumps({"kind": "saved", "L": L, "cases": [list(k) for k in d if isinstance(k, tuple)], "W13": list(d["W13"].shape), "W2": list(d["W2"].shape), "x_rms": round(Xa.float().pow(2).mean().sqrt().item(), 4)}), flush=True)
