"""Capture real attention inputs: Q and K after norm and rotary, V after projection, at chosen full-attention layers of Qwen3.8-27B (BF16 weights, torch fallback GDN), on real tokens from a held-out prompt text (not included). Writes fp16 [1, L, H, 256] tensors to scratch. No text or tokens are saved."""
import glob, json, os, sys, time
import torch
from safetensors import safe_open
import transformers.models.qwen3_5.modeling_qwen3_5 as m
from transformers import AutoConfig, AutoTokenizer
MD = "/scratch/project_XXXXXXXXX/models/Qwen3.8-27B"
OUT = sys.argv[1]; LS = [int(x) for x in sys.argv[2].split(",")]; LAYERS = [int(x) for x in sys.argv[3].split(",")]
os.makedirs(OUT, exist_ok=True)
cfg = AutoConfig.from_pretrained(MD).text_config
n = max(LAYERS) + 1
cfg.num_hidden_layers = n; cfg.layer_types = cfg.layer_types[:n]
assert all(cfg.layer_types[i] == "full_attention" for i in LAYERS), cfg.layer_types
cfg._attn_implementation = "sdpa"
with torch.device("meta"):
    M = m.Qwen3_5TextModel(cfg)
sd = {}
P = "model.language_model."
want = set(M.state_dict().keys())
for f in sorted(glob.glob(MD + "/model-*.safetensors")):
    with safe_open(f, "pt") as sf:
        for k in sf.keys():
            if k.startswith(P) and k[len(P):] in want:
                sd[k[len(P):]] = sf.get_tensor(k).to(torch.bfloat16)
miss = want - set(sd)
print("loaded", len(sd), "missing", sorted(miss)[:8], flush=True)
M.load_state_dict(sd, strict=False, assign=True)
del sd
M.rotary_emb = m.Qwen3_5TextRotaryEmbedding(cfg)
M = M.cuda().eval()
tok = AutoTokenizer.from_pretrained(MD)
ids = []
for ln in open("/scratch/project_XXXXXXXXX/data/prompts/val.jsonl"):
    r = json.loads(ln)
    ids += tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=False, return_dict=False)
    if len(ids) > max(LS) + 1:
        break
print("tokens", len(ids), flush=True)
cap = {}
orig_rot = m.apply_rotary_pos_emb
stack = {}


def rot(q, k, c, s):
    a, b = orig_rot(q, k, c, s)
    stack["q"], stack["k"] = a, b
    return a, b


m.apply_rotary_pos_emb = rot
for i in LAYERS:
    def h(mod, inp, out, i=i):
        cap[i] = (stack["q"].transpose(1, 2).to(torch.float16).cpu(), stack["k"].transpose(1, 2).to(torch.float16).cpu())
    M.layers[i].self_attn.register_forward_hook(h)
    def hv(mod, inp, out, i=i):
        cap[("v", i)] = out.view(*out.shape[:2], -1, cfg.head_dim).to(torch.float16).cpu()
    M.layers[i].self_attn.v_proj.register_forward_hook(hv)
for L in LS:
    t0 = time.time(); cap.clear()
    with torch.inference_mode():
        M(input_ids=torch.tensor([ids[:L]], device="cuda"), use_cache=False)
    for i in LAYERS:
        Q, K = cap[i]; V = cap[("v", i)]
        torch.save(dict(Q=Q.clone(), K=K.clone(), V=V.clone()), f"{OUT}/real_L{L}_l{i}.pt")
        st = lambda x: dict(mean=round(x.float().mean().item(), 4), std=round(x.float().std().item(), 4), maxabs=round(x.float().abs().max().item(), 3), nan=int(torch.isnan(x).sum()))
        print(json.dumps(dict(L=L, layer=i, Q=list(Q.shape), K=list(K.shape), V=list(V.shape), qs=st(Q), ks=st(K), vs=st(V), t=round(time.time() - t0, 1))), flush=True)
