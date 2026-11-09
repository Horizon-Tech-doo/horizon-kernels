# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import os


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def _install_mtp_draft_head():
    """Fused FP8 draft head (K1a, horizon_kernels/draft_head.py) for the MTP drafter, switched by HORIZON_FP8_DRAFT_HEAD=1.

    vLLM 0.30: this was an in-tree edit of the venv's qwen3_5_mtp.py (Qwen3_5MTP.get_top_tokens, "HORIZON PATCH
    25-09-2026"). vLLM 0.32's venv is stock, so the same logic is installed here as a class patch instead (ported
    26-10-2026 / 24-10-2026 port032). Drafts only propose ids, the target model verifies with its own BF16 lm_head, so this
    can change acceptance but never output. Needs "use_local_argmax_reduction": true in the speculative config (the drafter
    then calls model.get_top_tokens). Where the venv already carries the in-tree patch (0.30), nothing is done."""
    if os.environ.get("HORIZON_FP8_DRAFT_HEAD") != "1":
        return
    import torch
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    if "get_top_tokens" in Qwen3_5MTP.__dict__ or getattr(Qwen3_5MTP.get_top_tokens, "_horizon_draft_head", False):
        return          # in-tree patch (0.30 venv) or already installed
    orig = Qwen3_5MTP.get_top_tokens       # LocalArgmaxMixin.get_top_tokens (D2T-aware vocab-parallel argmax)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        head = getattr(self, "_horizon_draft_head", None)
        if head is None and getattr(self.lm_head, "tp_size", 1) == 1:   # fused head assumes an unsharded lm_head
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("horizon_kernels draft head must be built before CUDA graph capture")
            from horizon_kernels.draft_head import DraftHeadFP8

            weight = self.lm_head.weight[: self.config.vocab_size]
            act = os.environ.get("HORIZON_DRAFT_HEAD_ACT", "fp8")  # fp8 = K1a v1, bf16 = K1a v2
            # optional draft vocabulary (MTP plan): score only those rows; optional separately trained head
            vocab_ids = None
            vpath = os.environ.get("HORIZON_DRAFT_VOCAB")
            if vpath:
                vocab_ids = torch.load(vpath, map_location=weight.device).to(torch.long)
                wpath = os.environ.get("HORIZON_DRAFT_HEAD_WEIGHTS")
                if wpath:
                    from safetensors.torch import load_file
                    weight = load_file(wpath, device=str(weight.device))["weight"].to(weight.dtype)
                    assert weight.shape == (vocab_ids.numel(), self.config.hidden_size), weight.shape
            head = self._horizon_draft_head = DraftHeadFP8(weight, act=act, vocab_ids=vocab_ids)
            _log().info("horizon_kernels: fused FP8 draft head active (%s x %s, act=%s, draft vocab=%s, trained head=%s)",
                        head.N, head.K, act, None if vocab_ids is None else vocab_ids.numel(),
                        bool(vpath and os.environ.get("HORIZON_DRAFT_HEAD_WEIGHTS")))
        dump = os.environ.get("HORIZON_DUMP_DRAFT_H")
        if dump and hidden_states.shape[0] <= 16 and not torch.cuda.is_current_stream_capturing():
            # Debug only (needs --enforce-eager): save real draft hidden states for offline analysis.
            buf = self.__dict__.setdefault("_horizon_dump", [])
            n_have = sum(x.shape[0] for x in buf)
            if n_have < 20000 and bool(hidden_states.abs().amax() > 0):
                buf.append(hidden_states.detach().to("cpu", torch.bfloat16))
                n_have += hidden_states.shape[0]
                if len(buf) % 200 == 0 or n_have >= 20000:
                    torch.save(torch.cat(buf), dump)
        if head is None or hidden_states.shape[0] > head.BM:
            return orig(self, hidden_states)
        top = head(hidden_states)
        d2t = getattr(self, "draft_id_to_target_id", None)   # same remap as LocalArgmaxMixin (draft vocab)
        if d2t is not None:
            top = top + d2t[top]
        return top

    get_top_tokens._horizon_draft_head = True
    Qwen3_5MTP.get_top_tokens = get_top_tokens

    # Build the head right after the drafter is loaded (eagerly, lm_head sharing with the target is done by then) so the
    # first call can never land inside a CUDA-graph capture. A failure here only means the lazy path above builds it.
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
    orig_load = DraftModelSpeculator.load_model

    def load_model(self, target_model):
        r = orig_load(self, target_model)
        try:
            m = self.model
            if isinstance(m, Qwen3_5MTP):
                with torch.inference_mode():
                    m.get_top_tokens(torch.zeros(1, m.config.hidden_size, dtype=self.dtype, device=self.device))
        except Exception as e:  # noqa: BLE001
            _log().warning("horizon_kernels: draft head pre-build failed (%s: %s); it will be built on first use",
                           type(e).__name__, e)
        return r

    load_model._horizon_draft_head = True
    DraftModelSpeculator.load_model = load_model
    _log().info("horizon_kernels: draft head hook installed on Qwen3_5MTP.get_top_tokens (vLLM without the in-tree patch)")


def register():
    if os.environ.get("HORIZON_LMHEAD_SCREEN", "0") == "1":
        from . import lmhead_screen
        lmhead_screen.install()
    try:
        _install_mtp_draft_head()
    except Exception as e:  # noqa: BLE001  (vLLM layout changed: drafts fall back to the stock full-vocab argmax)
        try:
            _log().warning("horizon_kernels: draft head hook NOT installed (%s: %s); stock draft argmax is used",
                           type(e).__name__, e)
        except Exception:  # noqa: BLE001
            pass
