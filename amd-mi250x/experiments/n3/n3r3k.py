"""N3 round 3: live vLLM 0.30 start on gfx90a. Prints the backend and layout it picks, runs a short generate, and with real weights dumps a real decode step of 3 full-attention layers
(query, K, V by sequence, lengths) for the kernel race on real data. Tokens are from Python source text in the image, not user data.
usage: n3r3k.py real|dummy OUT [backend]. env VLLM_KV_CACHE_LAYOUT applies. Weights: Qwen3.8-27B (BF16). dummy uses 4 layers and random weights, only for log lines and the code path."""
import os, sys, glob, json
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ["VLLM_NO_USAGE_STATS"] = "1"
mode, OUT = sys.argv[1], sys.argv[2]; backend = sys.argv[3] if len(sys.argv) > 3 else None
import torch
from vllm import LLM, SamplingParams
from vllm.v1.attention.backends import rocm_attn, triton_attn
M = "/flash/project_XXXXXXXXX/models/Qwen3.8-27B"; NSEQ, LMAX = 8, 8000
seen, saved, shown = {}, [], set()
want = {0, 8, 15}
orig = rocm_attn.RocmAttentionImpl.forward
def fwd(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **k):
    i = seen.setdefault(id(layer), len(seen))
    if i < 2 and i not in shown and attn_metadata is not None:
        shown.add(i); kc, vc = rocm_attn.PagedAttention.split_kv_cache(kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size)
        print(json.dumps({"t": "view", "impl": type(self).__name__, "layer": i, "kv": list(kv_cache.shape), "kv_stride": list(kv_cache.stride()), "kv_off": kv_cache.storage_offset(), "storage_elems": kv_cache.untyped_storage().size() // kv_cache.element_size(), "dtype": str(kv_cache.dtype), "k": list(kc.shape), "k_stride": list(kc.stride()), "k_off": kc.storage_offset(), "v": list(vc.shape), "v_stride": list(vc.stride()), "v_off": vc.storage_offset(), "native": rocm_attn.has_native_kv_cache_layout(kc, vc) if hasattr(rocm_attn, "has_native_kv_cache_layout") else None}), flush=True)
    if attn_metadata is not None and attn_metadata.max_query_len == 1 and mode == "real" and i in want and i not in saved and query.shape[0] == NSEQ:
        kc, vc = rocm_attn.PagedAttention.split_kv_cache(kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size)
        bt, sl = attn_metadata.block_table, attn_metadata.seq_lens
        print(json.dumps({"t": "capture", "layer": i, "q": list(query.shape), "kc": list(kc.shape), "vc": list(vc.shape), "bt": list(bt.shape), "lens": sl.tolist(), "dtype": str(query.dtype), "impl": type(self).__name__}), flush=True)
        seqs = []
        for b in range(query.shape[0]):
            n = int(sl[b]); np_ = -(-n // kc.shape[3]); ids = bt[b, :np_].long()
            seqs.append({"K": kc[ids].clone().cpu(), "V": vc[ids].clone().cpu(), "n": n, "ids": ids.cpu()})
        torch.save({"Q": query.clone().cpu(), "seqs": seqs, "layer": i}, f"{OUT}/layer{i}.pt"); saved.append(i)
    return orig(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **k)
rocm_attn.RocmAttentionImpl.forward = fwd
torig = triton_attn.TritonAttentionImpl.forward if hasattr(triton_attn, "TritonAttentionImpl") else None
def tfwd(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **k):
    i = seen.setdefault(id(layer), len(seen))
    if i < 2 and i not in shown and attn_metadata is not None:
        shown.add(i); print(json.dumps({"t": "view", "impl": type(self).__name__, "layer": i, "kv": list(kv_cache.shape), "kv_stride": list(kv_cache.stride()), "kv_off": kv_cache.storage_offset(), "storage_elems": kv_cache.untyped_storage().size() // kv_cache.element_size(), "dtype": str(kv_cache.dtype)}), flush=True)
    return torig(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **k)
if torig: triton_attn.TritonAttentionImpl.forward = tfwd
kw = {}
if backend: kw["attention_config"] = {"backend": backend}
if mode == "dummy":
    kw.update(load_format="dummy", hf_overrides={"text_config": {"num_hidden_layers": 8, "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2}})
llm = LLM(model=M, max_model_len=9000, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0}, gpu_memory_utilization=0.9, seed=1, **kw)
tok = llm.get_tokenizer()
src = sorted(glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/**/*.py", recursive=True) + glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/**/*.py", recursive=True))
texts = "\n".join(open(f, errors="ignore").read() for f in src); ids = tok(texts, add_special_tokens=False).input_ids
print(json.dumps({"t": "tokens", "n": len(ids)}), flush=True)
n = NSEQ if mode == "real" else 2; L = LMAX if mode == "real" else 256
prompts = [{"prompt_token_ids": ids[i * 3000: i * 3000 + L - i * 500 * (mode == "real")]} for i in range(n)]
# real decode batch: sequences must be running together, so max_tokens > 1
out = llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
print(json.dumps({"t": "gen", "lens": [len(o.prompt_token_ids) for o in out], "saved": saved, "layers_seen": len(seen), "txt": out[0].outputs[0].text[:40], "tokens": [list(o.outputs[0].token_ids) for o in out]}), flush=True)
print("DONE", flush=True)
