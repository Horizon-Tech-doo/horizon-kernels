"""N3 round 4 frame check: live vLLM 0.30 start with real Qwen3.8-27B weights on 2 GCDs, tensor parallel 2, default backend, BF16. Prints what each worker's cache looks like (Hkv, page, pages) and the engine's own capacity lines, and runs a short real generate.
Top-level patches run in spawned workers too, because spawn re-imports this module. usage: n3r4k.py [backend]"""
import os, sys, glob, json
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0"); os.environ["VLLM_NO_USAGE_STATS"] = "1"
import torch
from vllm import LLM, SamplingParams
from vllm.v1.attention.backends import rocm_attn, triton_attn
M = "/flash/project_XXXXXXXXX/models/Qwen3.8-27B"; seen, shown = {}, set()
def show(self, layer, kv_cache, attn_metadata, extra):
    i = seen.setdefault(id(layer), len(seen))
    if i < 4 and i not in shown and attn_metadata is not None and kv_cache.dim() == 4:
        shown.add(i); print(json.dumps({"t": "view", "impl": type(self).__name__, "layer": i, "kv": list(kv_cache.shape), "kv_stride": list(kv_cache.stride()), "storage_elems": kv_cache.untyped_storage().size() // kv_cache.element_size(), "dtype": str(kv_cache.dtype), "num_heads": getattr(self, "num_heads", None), "num_kv_heads": getattr(self, "num_kv_heads", None), "head_size": getattr(self, "head_size", None), "bt": list(attn_metadata.block_table.shape) if getattr(attn_metadata, "block_table", None) is not None else None, **extra}), flush=True)
orig = rocm_attn.RocmAttentionImpl.forward
def fwd(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **k):
    ex = {}
    if len(shown) < 4 and kv_cache.dim() == 4:
        kc, vc = rocm_attn.PagedAttention.split_kv_cache(kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size); ex = {"k": list(kc.shape), "v": list(vc.shape), "q": list(query.shape)}
    show(self, layer, kv_cache, attn_metadata, ex); return orig(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **k)
rocm_attn.RocmAttentionImpl.forward = fwd
torig = triton_attn.TritonAttentionImpl.forward
def tfwd(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **k):
    show(self, layer, kv_cache, attn_metadata, {"q": list(query.shape)}); return torig(self, layer, query, key, value, kv_cache, attn_metadata, output, *a, **k)
triton_attn.TritonAttentionImpl.forward = tfwd
if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
    kw = {"attention_config": {"backend": backend}} if backend else {}
    llm = LLM(model=M, max_model_len=9000, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0}, gpu_memory_utilization=0.9, seed=1, tensor_parallel_size=2, **kw)
    tok = llm.get_tokenizer()
    src = sorted(glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/**/*.py", recursive=True) + glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/**/*.py", recursive=True))
    ids = tok("\n".join(open(f, errors="ignore").read() for f in src), add_special_tokens=False).input_ids
    prompts = [{"prompt_token_ids": ids[i * 3000: i * 3000 + 8000 - i * 500]} for i in range(8)]
    out = llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
    print(json.dumps({"t": "gen", "lens": [len(o.prompt_token_ids) for o in out], "layers_seen_main": len(seen), "txt": out[0].outputs[0].text[:40], "tokens": [list(o.outputs[0].token_ids) for o in out]}), flush=True)
    print("DONE", flush=True); os._exit(0)
