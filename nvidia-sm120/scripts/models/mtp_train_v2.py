#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# MTP stage 2, v2: retrain the Qwen3.8-27B-NVFP4 MTP draft head for a trimmed draft vocabulary (FR-Spec style).
#
# Trained: the MTP module (fc + 1 full-attention decoder layer + 3 norms, fp32 master / bf16 compute) and a new
# draft head (V x hidden, initialised from the main lm_head rows of the draft-vocabulary ids).
# Frozen: the main model, embeddings, lm_head. Teacher = the NVFP4 main model itself, decompressed to BF16 on the GPU.
#
# Safety: stop your vLLM service first (training needs the whole GPU); RAM watchdog (MemAvailable < 4 GB,
# or RSS > 30 GB while loading = CPU-side decompression); never touches the original model_mtp.safetensors; saves to
# OUTDIR only if the retrained head beats the untrained draft-vocab head at 3 AND 5 drafts.
#
# Run it inside tmux (smoke first):
#   cd $HK_WORK/mtp
#   LD_LIBRARY_PATH=$HK_DATA/cuda-compat PYTHONPATH=$HK_DATA/llmc-pkgs \
#   DATA=$HK_DATA/models/mtp-data/gen_v2.jsonl \
#   VOCAB=$HK_DATA/models/Qwen3.8-27B-NVFP4/horizon_mtp/draft_vocab_32768.pt \
#   $HK_VENV/bin/python mtp_train_v2.py --smoke 2>&1 | tee ~/mtp_train_v2_smoke.log
# Every option can be given as an env var (NAME) or a flag (--name); see parse_args().
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MODEL_DEFAULT = "$HK_DATA/models/Qwen3.8-27B-NVFP4"
DATA_DEFAULT = "$HK_DATA/models/mtp-data/gen.jsonl"
OUTDIR_DEFAULT = MODEL_DEFAULT + "/horizon_mtp"
VOCAB_DEFAULT = OUTDIR_DEFAULT + "/draft_vocab_32768.pt"
TEMPLATE_DEFAULT = "$HK_DATA/chat_template.jinja"   # the template vLLM serves with
LOGIT_CHUNK = int(os.environ.get("LOGIT_CHUNK", "1024"))
TRUNCATE = os.environ.get("TRUNCATE", "left")          # left | right: which end of an over-long document to cut
GIB = 1024 ** 3


def now(fmt="%d-%m-%Y %H:%M:%S"):
    return time.strftime(fmt)


def log(*a):
    print(now(), *a, flush=True)


# =====================================================================================================================
# Pure pieces (CPU-testable, see test_mtp_train_v2.py)
# =====================================================================================================================
#
# ALIGNMENT (identical to v1's mtp_steps; this is how vLLM's MTP proposer feeds the head)
#
#   tokens      x_0 x_1 ... x_{T-1}
#   h_p         main model's last hidden state at position p, AFTER the final norm (what vLLM hands the MTP)
#   teacher[p]  main model's next-token distribution at position p, i.e. the distribution of x_{p+1}
#
#   Slot i is a chain root: the main model has verified x_0..x_{i+1}; the proposer gets h_i and the newest token
#   x_{i+1} (vLLM: target hidden states with input_ids shifted left by one).
#     step 0:  out_0[i] = MTP(emb(x_{i+1}),   h_i)          predicts x_{i+2},    target = teacher[i+1]
#     step k:  out_k[i] = MTP(emb(x_{i+k+1}), out_{k-1}[i]) predicts x_{i+k+2},  target = teacher[i+k+1]
#   So step k has n_k = T-2-k slots (x_{i+k+2} must exist); the embedding index and the teacher index are both
#   i+k+1 (teacher_index); step k consumes the previous step's normed output at the SAME slot (vLLM feeds the MTP
#   output back as the hidden state). For k > 0 the input token x_{i+k+1} is teacher-forced with the data token; in
#   greedy decoding it equals the accepted draft whenever the chain is still alive and the data follows the greedy
#   path (reported as data_greedy_frac). Slot i uses rotary position i at every step: a uniform shift per step, so
#   relative positions inside the step-k causal attention are the same as vLLM's i+k.
#   Remaining approximation (same as v1 and DeepSeek-V3 MTP training): at step k the MTP attention runs causally
#   over the step-k slots, while vLLM's step k attends to step-0 KV for the context plus the chain's own KV.
#
# DRAFTED TOKENS: vLLM only drafts during decode. Chain root i is a real decode position iff x_{i+1} is a generated
# token (resp[i+1]); its step-k target x_{i+k+2} must be generated too (resp[i+k+2]). Prompt/system tokens are
# never drafted, so they are excluded from the loss and the metric (v1 trained and evaluated on them).

Doc = namedtuple("Doc", "ids resp kind")


def teacher_index(idx, k):
    """Teacher position (and embedding position) used by draft step k at slot idx."""
    return idx + k + 1


def n_slots(T, k):
    return T - 2 - k


def step_mask(resp, k, n):
    """(n,) bool: slot i of step k is a real decode chain root with a generated target."""
    return resp[1:1 + n] & resp[k + 2:k + 2 + n]


def root_mask(resp, nsteps, n):
    """(n,) bool: slot i can run a full chain of nsteps drafts inside the generated part."""
    m = resp[1:1 + n].clone()
    for k in range(nsteps):
        m &= resp[k + 2:k + 2 + n]
    return m


def build_vocab_maps(ids, full_vocab):
    """ids: 1-D integer tensor of draft-vocabulary token ids (draft index = position in the file).
    Returns d2t (draft index -> full id, int64) and t2d (full id -> draft index or -1, int64)."""
    if not isinstance(ids, torch.Tensor):
        ids = torch.as_tensor(ids)
    if ids.dim() != 1 or ids.numel() == 0:
        raise ValueError(f"draft vocabulary must be a non-empty 1-D tensor, got shape {tuple(ids.shape)}")
    if ids.dtype.is_floating_point or ids.dtype == torch.bool:
        raise ValueError(f"draft vocabulary must hold integer token ids, got {ids.dtype}")
    ids = ids.long()
    if int(ids.min()) < 0 or int(ids.max()) >= full_vocab:
        raise ValueError(f"draft vocabulary ids must lie in [0, {full_vocab}), got [{int(ids.min())}, {int(ids.max())}]")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("draft vocabulary contains duplicate ids")
    d2t = ids.clone()
    t2d = torch.full((full_vocab,), -1, dtype=torch.long)
    t2d[d2t] = torch.arange(d2t.numel(), dtype=torch.long)
    return d2t, t2d


def teacher_from_logits(logits, topk):
    """logits: (n, V_full), bf16 as vLLM's verifier computes them. Returns the top-k probabilities of the FULL softmax
    (fp32), their ids, and the greedy argmax exactly as vLLM picks it (argmax, not topk[0]: bf16 ties are common)."""
    lf = logits.float()
    lse = torch.logsumexp(lf, -1, keepdim=True)
    v, i = lf.topk(topk, -1)
    return (v - lse).exp(), i, lf.argmax(-1)


def restrict_to_vocab(p, ids, t2d):
    """Restrict teacher top-k (p, ids: (n, k)) to the draft vocabulary: drop out-of-vocab ids, renormalise.
    Returns q (n, k) fp32 (0 at dropped entries), di (n, k) draft indices (0 at dropped entries, q is 0 there),
    valid (n,) bool (some in-vocab mass), mass (n,) in-vocab share of the teacher's full probability."""
    di = t2d[ids]
    inv = di >= 0
    pm = torch.where(inv, p, torch.zeros_like(p))
    mass = pm.sum(-1)
    valid = mass > 0
    q = pm / mass.clamp_min(1e-30).unsqueeze(-1)
    q = torch.where(valid.unsqueeze(-1), q, torch.zeros_like(q))
    return q, di.clamp_min(0), valid, mass


def kl_rows(logits, q, di):
    """sum over rows of KL(q || softmax(logits)) with q given on its support (q, di: (n, k)). Exact KL: the student
    softmax runs over its whole (draft) vocabulary, q is zero outside its support."""
    logp = logits.float().log_softmax(-1)
    return (torch.xlogy(q, q) - q * logp.gather(-1, di)).sum()


def _kl_chunk(x, w, q, di):
    logits = F.linear(x.to(w.dtype), w)
    return kl_rows(logits, q, di), logits.argmax(-1)


def chain_counts(M):
    """M: (K, R) bool, M[k, r] = draft k of chain root r equals the teacher's greedy token at the aligned position.
    Greedy speculative decoding accepts drafts until the first mismatch: reach[k] = #roots whose first k+1 drafts
    were all accepted; agree[k] = #roots whose draft k matched regardless of the chain (diagnostic)."""
    Ml = M.long()
    return {"n": int(M.shape[1]), "reach": Ml.cumprod(0).sum(1).tolist(), "agree": Ml.sum(1).tolist()}


def new_counts(K):
    return {"n": 0, "reach": [0] * K, "agree": [0] * K}


def add_counts(a, b):
    a["n"] += b["n"]
    a["reach"] = [x + y for x, y in zip(a["reach"], b["reach"])]
    a["agree"] = [x + y for x, y in zip(a["agree"], b["agree"])]
    return a


def finalize_counts(c, K):
    """acc_len[m] = mean accepted length with m drafts, counted as vLLM does (1 + accepted drafts, the bonus token
    included) = 1 + sum_{k<m} P(chain >= k+1); truncating the chain at m is exact since draft k never depends on later
    drafts. per_pos[k] = P(draft k accepted) as vLLM's per-position acceptance rate."""
    n = c["n"]
    if n == 0:
        return {"n_roots": 0, "acc_len": {str(m): float("nan") for m in range(1, K + 1)},
                "per_pos": [float("nan")] * K, "agree": [float("nan")] * K}
    reach = [r / n for r in c["reach"]]
    acc, s = {}, 0.0
    for m in range(1, K + 1):
        s += reach[m - 1]
        acc[str(m)] = round(1.0 + s, 4)
    return {"n_roots": n, "acc_len": acc, "per_pos": [round(r, 4) for r in reach],
            "agree": [round(a / n, 4) for a in c["agree"]]}


def gate_steps(nsteps):
    """Draft counts at which the retrained head must beat the baseline: 3 and 5 (clipped to NSTEPS) and NSTEPS."""
    s = {min(3, nsteps), nsteps}
    if nsteps >= 5:
        s.add(5)
    return sorted(s)


def gate_passes(c, b, steps):
    return all(c["acc_len"][str(m)] > b["acc_len"][str(m)] for m in steps)


def better(ev, best_ev, nsteps):
    key = lambda e: (e["acc_len"][str(nsteps)], e["acc_len"][str(min(3, nsteps))])  # noqa: E731
    return best_ev is None or key(ev) > key(best_ev)


def pack_docs(docs, maxlen):
    """Greedy packing of WHOLE documents (never split), in order, into sequences of at most maxlen tokens.
    docs: list of Doc (ids, resp same length). Returns list of (ids, resp) lists."""
    packs, cur_ids, cur_resp = [], [], []
    for d in docs:
        if len(d.ids) != len(d.resp):
            raise ValueError("ids/resp length mismatch")
        if len(d.ids) > maxlen:
            raise ValueError(f"document of {len(d.ids)} tokens exceeds MAXLEN {maxlen}: truncate first")
        if cur_ids and len(cur_ids) + len(d.ids) > maxlen:
            packs.append((cur_ids, cur_resp))
            cur_ids, cur_resp = [], []
        cur_ids = cur_ids + list(d.ids)
        cur_resp = cur_resp + list(d.resp)
    if cur_ids:
        packs.append((cur_ids, cur_resp))
    return packs


def response_mask(offsets, gen_start, gen_end):
    """1 for tokens whose characters lie inside the generated span [gen_start, gen_end)."""
    return [1 if (s >= gen_start and e <= gen_end and e > s) else 0 for s, e in offsets]


def render_texts(tok, template, row):
    """(prompt, generated, suffix) strings. The prompt is rendered with the SERVED chat template, exactly as vLLM
    built it for mtp_gen (for thinking requests the served chat template adds a 'Reasoning effort' system message, which
    v1's hand-written format dropped). The generated part mirrors what the model emitted."""
    if row.get("prompt_text") and row.get("gen_text"):
        # data v3 (mtp_gen_v3.py): the generator stored exactly what the model saw (served template incl. system
        # prompt and tools) and emitted (thinking, content, tool-call markup) -> use it verbatim
        return row["prompt_text"], row["gen_text"], "\n"
    think = bool(row.get("think"))
    reasoning = (row.get("reasoning") or "").strip()
    content = (row.get("content") or "").strip()
    if template is not None:
        prompt = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], chat_template=template,
                                         add_generation_prompt=True, tokenize=False, enable_thinking=think)
    else:
        prompt = (f"<|im_start|>user\n{row['prompt']}<|im_end|>\n<|im_start|>assistant\n"
                  + ("<think>\n" if think else "<think>\n\n</think>\n\n"))
    want = "<think>\n" if think else "<think>\n\n</think>\n\n"
    if not prompt.endswith(want):
        raise ValueError(f"rendered prompt does not end with the expected generation prefix {want!r}")
    gen = (f"{reasoning}\n</think>\n\n" if think else "") + content + "<|im_end|>"
    return prompt, gen, "\n"


def build_docs(rows, tok, template, maxlen, min_len=64, min_resp=8, batch=256):
    docs, st = [], {"rows": len(rows), "render_errors": 0, "too_short": 0, "truncated": 0}
    for b in range(0, len(rows), batch):
        texts, spans, kinds = [], [], []
        for r in rows[b:b + batch]:
            try:
                prompt, gen, suffix = render_texts(tok, template, r)
            except Exception:  # noqa: BLE001
                st["render_errors"] += 1
                continue
            texts.append(prompt + gen + suffix)
            spans.append((len(prompt), len(prompt) + len(gen)))
            kinds.append(str(r.get("kind") or "all"))
        if not texts:
            continue
        enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
        for ids, offs, (a, z), kind in zip(enc["input_ids"], enc["offset_mapping"], spans, kinds):
            resp = response_mask(offs, a, z)
            if len(ids) > maxlen:
                # TRUNCATE=left (default): drop the START of the prompt, keep its end + the whole answer (long inputs
                # and agent sessions are prompt-heavy; cutting the end threw their answers away). right = old behaviour.
                if TRUNCATE == "left":
                    ids, resp = ids[-maxlen:], resp[-maxlen:]
                else:
                    ids, resp = ids[:maxlen], resp[:maxlen]
                st["truncated"] += 1
            if len(ids) <= min_len or sum(resp) < min_resp:
                st["too_short"] += 1
                continue
            docs.append(Doc(list(ids), resp, kind))
    st["docs"] = len(docs)
    return docs, st


def data_stat_counts(ids, resp, targ, t2d, mass, valid):
    """Counts over positions p whose next token x_{p+1} is a drafted token (resp[p] & resp[p+1]):
    oov = teacher greedy token outside the draft vocab, greedy = data token equals the teacher's greedy token,
    mass = in-vocab share of the teacher top-k, nomass = no top-k entry in the draft vocab (dropped from the loss)."""
    m = resp[:-1] & resp[1:]
    return {"n": int(m.sum()),
            "oov": int(((t2d[targ[:-1].long()] < 0) & m).sum()),
            "greedy": int(((ids[1:].long() == targ[:-1].long()) & m).sum()),
            "mass": float(mass[:-1][m].double().sum()),
            "nomass": int(((~valid[:-1]) & m).sum())}


def finalize_data_stats(c):
    n = max(1, c["n"])
    return {"positions": c["n"], "teacher_argmax_oov_frac": round(c["oov"] / n, 5),
            "data_greedy_frac": round(c["greedy"] / n, 5), "in_vocab_topk_mass": round(c["mass"] / n, 5),
            "no_in_vocab_topk_frac": round(c["nomass"] / n, 5)}


def lr_scale(step, total, warm):
    return min(1.0, (step + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total))))


# =====================================================================================================================
# System helpers: memory, vLLM control, RAM watchdog
# =====================================================================================================================

STATE = {"vllm_stopped": False, "restart_scheduled": False, "shutdown": False}
_restart_lock = threading.Lock()
WATCHDOG = None


class RamAbort(BaseException):
    """Raised in the main thread by the RAM watchdog (BaseException so no `except Exception` can swallow it)."""


def parse_kb_table(text):
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        parts = v.split()
        if parts and parts[0].isdigit():
            out[k.strip()] = int(parts[0])
    return out


def mem_snapshot():
    with open("/proc/meminfo") as f:
        mi = parse_kb_table(f.read())
    with open("/proc/self/status") as f:
        st = parse_kb_table(f.read())
    g = lambda d, k: d.get(k, 0) / 2 ** 20  # kB -> GiB  # noqa: E731
    return {"avail_gb": g(mi, "MemAvailable"), "rss_gb": g(st, "VmRSS"), "anon_gb": g(st, "RssAnon"),
            "file_gb": g(st, "RssFile"), "hwm_gb": g(st, "VmHWM")}


def fmt_mem(s):
    return (f"host MemAvailable {s['avail_gb']:.1f} GiB | RSS {s['rss_gb']:.1f} (anon {s['anon_gb']:.1f}, "
            f"file {s['file_gb']:.1f}, peak {s['hwm_gb']:.1f}) GiB")


def is_vllm_cmdline(cmd):
    cmd = cmd.strip()
    return "vllm serve" in cmd or cmd.startswith("VLLM::")


def vllm_processes():
    me, out = os.getpid(), []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if is_vllm_cmdline(cmd):
            out.append((int(d), cmd.strip()[:100]))
    return out


def stop_vllm(timeout=300):
    log("waiting for vLLM to stop (site-specific: stop your vLLM service before this step)")
    STATE["vllm_stopped"] = True          # set before the attempt: from here on the finally block restarts vLLM
    # (site-specific: stop your vLLM service here)
    t0 = time.time()
    while True:
        ps = vllm_processes()
        if not ps:
            break
        if time.time() - t0 > timeout:
            raise RuntimeError(f"vLLM processes still alive {timeout}s after stop: {ps}")
        time.sleep(2)
    time.sleep(5)                          # let the driver release the GPU memory
    log(f"vLLM stopped ({time.time() - t0:.0f}s), no process matches 'vllm serve'")


def schedule_vllm_restart(reason):
    """Placeholder for restarting vLLM once this process has exited (see the site-specific marker below)."""
    with _restart_lock:
        if STATE["restart_scheduled"]:
            return
        STATE["restart_scheduled"] = True
    log(f"vLLM can be restarted now ({reason})")
    # (site-specific: restart your vLLM service here, once this process has exited)


class RamWatchdog(threading.Thread):
    def __init__(self, min_avail_gb, load_rss_gb, period=0.5, grace=45.0):
        super().__init__(daemon=True, name="ram-watchdog")
        self.min_avail_gb, self.load_rss_gb, self.period, self.grace = min_avail_gb, load_rss_gb, period, grace
        self.phase = "init"               # "load": the RSS limit applies too
        self.tripped, self.tripped_at = None, 0.0
        self.min_avail_seen = float("inf")
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        while not self._stop_evt.wait(self.period):
            try:
                s = mem_snapshot()
            except Exception:  # noqa: BLE001
                continue
            self.min_avail_seen = min(self.min_avail_seen, s["avail_gb"])
            if self.tripped is None:
                reason = None
                if s["avail_gb"] < self.min_avail_gb:
                    reason = f"MemAvailable {s['avail_gb']:.1f} GiB < {self.min_avail_gb} GiB"
                elif self.phase == "load" and s["rss_gb"] > self.load_rss_gb:
                    reason = (f"RSS {s['rss_gb']:.1f} GiB > {self.load_rss_gb} GiB while loading the model "
                              f"(anon {s['anon_gb']:.1f}, file {s['file_gb']:.1f}): CPU-side decompression?")
                if reason:
                    self.tripped, self.tripped_at = reason, time.time()
                    log("RAM WATCHDOG:", reason, "- aborting")
                    if not STATE["shutdown"]:
                        os.kill(os.getpid(), signal.SIGUSR1)
            elif time.time() - self.tripped_at > self.grace:
                log(f"RAM WATCHDOG: process still alive {self.grace:.0f}s after the abort - hard kill")
                if STATE["vllm_stopped"]:
                    schedule_vllm_restart("watchdog hard kill")
                sys.stdout.flush()
                os.kill(os.getpid(), signal.SIGKILL)


def _on_usr1(signum, frame):
    if STATE["shutdown"]:
        return
    raise RamAbort(WATCHDOG.tripped if WATCHDOG is not None and WATCHDOG.tripped else "RAM watchdog")


def _on_term(signum, frame):
    if STATE["shutdown"]:
        return
    raise SystemExit(f"received signal {signum}")


# =====================================================================================================================
# Model pieces
# =====================================================================================================================

def make_mtp(tcfg):
    """MTP module as vLLM's Qwen3_5MultiTokenPredictor computes it (v1 reconstruction, unchanged):
    h_next = norm(decoder_layer_full_attention(fc(cat(pre_fc_norm_embedding(emb), pre_fc_norm_hidden(h)))))."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

    H, eps = tcfg.hidden_size, tcfg.rms_norm_eps
    full_idx = list(tcfg.layer_types).index("full_attention")

    class MTP(nn.Module):
        def __init__(s):
            super().__init__()
            s.fc = nn.Linear(2 * H, H, bias=False)
            s.pre_fc_norm_embedding = Qwen3_5RMSNorm(H, eps=eps)
            s.pre_fc_norm_hidden = Qwen3_5RMSNorm(H, eps=eps)
            s.layers = nn.ModuleList([Qwen3_5DecoderLayer(tcfg, full_idx)])
            s.norm = Qwen3_5RMSNorm(H, eps=eps)

        def forward(s, emb, h, pe):
            x = s.fc(torch.cat([s.pre_fc_norm_embedding(emb), s.pre_fc_norm_hidden(h)], -1))
            x = s.layers[0](x, position_embeddings=pe, attention_mask=None)   # causal SDPA
            return s.norm(x)

    return MTP()


class Ctx:
    """Everything the train/eval code needs; device-agnostic so the CPU tests can drive it with a tiny model."""

    def __init__(self, lm, lm_head_w, mtp, head_w, d2t, t2d, topk, grad_ckpt=False):
        self.lm, self.embed, self.lm_head_w = lm, lm.embed_tokens, lm_head_w
        self.mtp, self.head_w = mtp, head_w
        self.device = lm_head_w.device
        self.d2t, self.t2d = d2t.to(self.device), t2d.to(self.device)
        self.topk, self.grad_ckpt = topk, grad_ckpt


def rope_for(lm, h):
    T = h.shape[1]
    pos = torch.arange(T, device=h.device).view(1, 1, -1).expand(3, 1, -1)   # text-only mrope: 3 equal rows
    return lm.rotary_emb(h, pos)                                             # (cos, sin), dtype of h like vLLM


def slice_pe(pe, n):
    return tuple(t[..., :n, :] for t in pe)


def autocast_for(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16)


def run_chain(mtp, h, e, pe, nsteps, grad_ckpt=False):
    """Unrolled draft chain, see ALIGNMENT. h, e: (1, T, H). Returns [out_k (1, n_k, H)]."""
    T, hid, outs = h.shape[1], h, []
    with autocast_for(h.device):
        for k in range(nsteps):
            n = n_slots(T, k)
            if n < 1:
                break
            emb = e[:, k + 1:k + 1 + n]           # x_{i+k+1}
            hin = hid[:, :n]                      # h_i (k = 0) or out_{k-1}[i]
            pk = slice_pe(pe, n)
            if grad_ckpt and torch.is_grad_enabled():   # no randomness inside: no RNG state to preserve
                out = checkpoint(mtp, emb, hin, pk, use_reentrant=False, preserve_rng_state=False)
            else:
                out = mtp(emb, hin, pk)
            outs.append(out)
            hid = out
    return outs


@torch.no_grad()
def main_pass(ctx, ids, chunk=LOGIT_CHUNK):
    x = torch.tensor([ids], device=ctx.device, dtype=torch.long)
    h = ctx.lm(input_ids=x, use_cache=False).last_hidden_state   # post final norm, bf16
    T = h.shape[1]
    P, I, A = [], [], []
    for c in range(0, T, chunk):
        p, i, a = teacher_from_logits(F.linear(h[0, c:c + chunk], ctx.lm_head_w), ctx.topk)
        P.append(p); I.append(i); A.append(a)
    tp, ti, targ = torch.cat(P), torch.cat(I), torch.cat(A)
    q, di, valid, mass = restrict_to_vocab(tp, ti, ctx.t2d)
    return {"x": x, "h": h, "e": ctx.embed(x), "pe": rope_for(ctx.lm, h),
            "q": q, "di": di, "valid": valid, "mass": mass, "targ": targ}


def mtp_losses(outs, w_c, d2t, tq, tdi, tvalid, ttarg, resp, step_w, chunk=LOGIT_CHUNK):
    """Training loss: sum_k step_w[k] * mean_{slots} KL(teacher restricted to the draft vocab at i+k+1 ||
    softmax(draft head(out_k[i]))) over slots that are real decode positions with in-vocab teacher mass.
    w_c: draft head weight in compute dtype (keeps the graph to the fp32 master). Teacher tensors are per position.
    Logits are computed in checkpointed chunks, so only the chunk inputs are kept for backward."""
    total, st = None, {"kl": [], "agree": [], "count": [], "matches": []}
    for k, (out, w) in enumerate(zip(outs, step_w)):
        n = out.shape[1]
        sl = slice(k + 1, k + 1 + n)                               # teacher_index(i, k) for i = 0..n-1
        m = step_mask(resp, k, n) & tvalid[sl]
        idx = m.nonzero(as_tuple=True)[0]
        match = torch.zeros(n, dtype=torch.bool, device=out.device)
        if idx.numel() == 0:
            st["kl"].append(float("nan")); st["agree"].append(float("nan")); st["count"].append(0)
            st["matches"].append(match)
            continue
        tix = teacher_index(idx, k)
        x, q, di, tg = out[0].index_select(0, idx), tq[tix], tdi[tix].long(), ttarg[tix].long()
        kl, ams = None, []
        for c in range(0, idx.numel(), chunk):
            args = (x[c:c + chunk], w_c, q[c:c + chunk], di[c:c + chunk])
            if torch.is_grad_enabled():
                kl_c, am_c = checkpoint(_kl_chunk, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                kl_c, am_c = _kl_chunk(*args)
            kl = kl_c if kl is None else kl + kl_c
            ams.append(am_c)
        hit = d2t[torch.cat(ams)] == tg
        match[idx] = hit
        L = kl / idx.numel()
        total = w * L if total is None else total + w * L
        st["kl"].append(float(L.detach())); st["agree"].append(float(hit.float().mean()))
        st["count"].append(int(idx.numel())); st["matches"].append(match)
    return total, st


def train_chain_counts(matches, resp, nsteps):
    """Chain acceptance on a training batch, from mtp_losses' per-slot matches."""
    if len(matches) < nsteps:
        return new_counts(nsteps)
    n_last = matches[nsteps - 1].shape[0]
    ridx = root_mask(resp, nsteps, n_last).nonzero(as_tuple=True)[0]
    return chain_counts(torch.stack([m[ridx] for m in matches[:nsteps]]))


@torch.no_grad()
def cache_eval_docs(ctx, docs):
    """Main-model pass once per held-out doc; keeps h (bf16), greedy targets and the restricted teacher on the CPU."""
    cache, ds = [], {"n": 0, "oov": 0, "greedy": 0, "mass": 0.0, "nomass": 0}
    for d in docs:
        mp = main_pass(ctx, d.ids)
        resp = torch.tensor(d.resp, dtype=torch.bool, device=ctx.device)
        c = data_stat_counts(mp["x"][0], resp, mp["targ"], ctx.t2d, mp["mass"], mp["valid"])
        for k in ds:
            ds[k] += c[k]
        cache.append({"kind": d.kind, "ids": mp["x"][0].to("cpu", torch.int32), "resp": resp.cpu(),
                      "h": mp["h"][0].cpu(), "targ": mp["targ"].to("cpu", torch.int32), "q": mp["q"].cpu(),
                      "di": mp["di"].to("cpu", torch.int32), "valid": mp["valid"].cpu()})
        del mp
    return cache, finalize_data_stats(ds)


@torch.no_grad()
def evaluate(ctx, cache, mode, nsteps, chunk=LOGIT_CHUNK):
    """Exact greedy chain acceptance on the cached held-out docs.
    mode 'full': original-style head = full 248k lm_head, ids are full ids.
    mode 'draft': draft head over the draft vocabulary, argmax mapped through d2t (an out-of-vocab teacher token can
    never be matched, exactly as in vLLM)."""
    was_training = ctx.mtp.training
    ctx.mtp.eval()
    dev = ctx.device
    W = ctx.lm_head_w if mode == "full" else ctx.head_w.detach().to(torch.bfloat16)
    tot, by_kind = new_counts(nsteps), {}
    kl_sum, kl_n = [0.0] * nsteps, [0] * nsteps
    for d in cache:
        ids = d["ids"].to(dev, torch.long)
        T = ids.numel()
        n_last = n_slots(T, nsteps - 1)
        if n_last < 1:
            continue
        ridx = root_mask(d["resp"].to(dev), nsteps, n_last).nonzero(as_tuple=True)[0]
        if ridx.numel() == 0:
            continue
        h = d["h"].to(dev)[None]
        outs = run_chain(ctx.mtp, h, ctx.embed(ids[None]), rope_for(ctx.lm, h), nsteps)
        targ = d["targ"].to(dev, torch.long)
        if mode == "draft":
            dq, ddi, dvalid = d["q"].to(dev), d["di"].to(dev, torch.long), d["valid"].to(dev)
        rows = []
        for k, out in enumerate(outs):
            x, tix = out[0].index_select(0, ridx), teacher_index(ridx, k)
            preds = []
            for c in range(0, x.shape[0], chunk):
                lg = F.linear(x[c:c + chunk].to(W.dtype), W).float()
                preds.append(lg.argmax(-1))
                if mode == "draft":
                    tc = tix[c:c + chunk]
                    vm = dvalid[tc]
                    if bool(vm.any()):
                        kl_sum[k] += float(kl_rows(lg[vm], dq[tc][vm], ddi[tc][vm]))
                        kl_n[k] += int(vm.sum())
            pred = torch.cat(preds)
            if mode == "draft":
                pred = ctx.d2t[pred]
            rows.append(pred == targ[tix])
        c = chain_counts(torch.stack(rows))
        add_counts(tot, c)
        add_counts(by_kind.setdefault(d["kind"], new_counts(nsteps)), c)
    if was_training:
        ctx.mtp.train()
    res = finalize_counts(tot, nsteps)
    if len(by_kind) > 1:
        res["by_kind"] = {k: {"n_roots": v["n"], "acc_len": finalize_counts(v, nsteps)["acc_len"]}
                          for k, v in sorted(by_kind.items())}
    if mode == "draft":
        res["kl"] = [round(s / n, 4) if n else float("nan") for s, n in zip(kl_sum, kl_n)]
    return res


def fmt_eval(tag, ev, nsteps):
    s = (f"{tag}: acc_len@1..{nsteps} {[ev['acc_len'][str(m)] for m in range(1, nsteps + 1)]} "
         f"per_pos {ev['per_pos']} agree {ev['agree']}")
    if "kl" in ev:
        s += f" kl {ev['kl']}"
    s += f" roots {ev['n_roots']}"
    for k, v in ev.get("by_kind", {}).items():
        s += f" | {k}: @3 {v['acc_len'].get(str(min(3, nsteps)))} @{nsteps} {v['acc_len'][str(nsteps)]}"
    return s


def snapshot(ctx):
    return {"mtp": {k: v.detach().to("cpu", torch.bfloat16).clone() for k, v in ctx.mtp.state_dict().items()},
            "head": ctx.head_w.detach().to("cpu", torch.bfloat16).clone()}


def build_save_dicts(snap, orig_meta):
    """model_mtp dict with exactly the original key set, shapes and dtypes; draft head dict {'weight': bf16 V x H}."""
    new = {"mtp." + k: v for k, v in snap["mtp"].items()}
    if set(new) != set(orig_meta):
        raise RuntimeError(f"MTP key set differs from the original: {sorted(set(new) ^ set(orig_meta))}")
    out = {}
    for k, (dt, shape) in orig_meta.items():
        if tuple(new[k].shape) != tuple(shape):
            raise RuntimeError(f"{k}: shape {tuple(new[k].shape)} != original {tuple(shape)}")
        out[k] = new[k].to(dt).contiguous()
    return out, {"weight": snap["head"].to(torch.bfloat16).contiguous()}


def save_outputs(outdir, model_dir, mtp_dict, head_dict, d2t, meta):
    from safetensors.torch import save_file

    os.makedirs(outdir, exist_ok=True)
    original = os.path.realpath(os.path.join(model_dir, "model_mtp.safetensors"))
    paths = {n: os.path.join(outdir, n) for n in
             ("model_mtp.safetensors", "draft_head.safetensors", "draft_vocab.pt", "meta.json")}
    for p in paths.values():
        if os.path.realpath(p) == original:
            raise RuntimeError(f"refusing to overwrite the original {original}")

    def atomic(path, write):
        tmp = path + ".tmp"
        write(tmp)
        os.replace(tmp, path)

    atomic(paths["model_mtp.safetensors"], lambda p: save_file(mtp_dict, p, metadata={"format": "pt"}))
    atomic(paths["draft_head.safetensors"], lambda p: save_file(head_dict, p, metadata={"format": "pt"}))
    atomic(paths["draft_vocab.pt"], lambda p: torch.save(d2t.detach().cpu().long().clone(), p))

    def wj(p):
        with open(p, "w") as f:
            json.dump(meta, f, indent=1)

    atomic(paths["meta.json"], wj)                      # written last: meta.json present = complete set
    return paths


# =====================================================================================================================
# Main
# =====================================================================================================================

def parse_args(argv=None):
    e = os.environ.get
    ap = argparse.ArgumentParser(description="Retrain the Qwen3.8 MTP head with a trimmed draft vocabulary.")
    ap.add_argument("--model", default=e("MODEL", MODEL_DEFAULT))
    ap.add_argument("--data", default=e("DATA", DATA_DEFAULT), help="gen.jsonl / gen_v2.jsonl from mtp_gen")
    ap.add_argument("--eval-data", default=e("EVAL_DATA", ""),
                    help="optional separate held-out jsonl (ideally generated at temperature 0: exact greedy metric)")
    ap.add_argument("--vocab", default=e("VOCAB", VOCAB_DEFAULT), help="torch file, 1-D int64 draft-vocab token ids")
    ap.add_argument("--nsteps", type=int, default=int(e("NSTEPS", "5")))
    ap.add_argument("--step-w", default=e("STEP_W", "1.0,0.8,0.6,0.45,0.35"))
    ap.add_argument("--lr", type=float, default=float(e("LR", "5e-5")), help="MTP layer learning rate")
    ap.add_argument("--lr-head", type=float, default=float(e("LR_HEAD")) if e("LR_HEAD") else None,
                    help="draft head learning rate (default 0.1*LR)")
    ap.add_argument("--epochs", type=int, default=int(e("EPOCHS", "1")))
    ap.add_argument("--maxlen", type=int, default=int(e("MAXLEN", "4096")))
    ap.add_argument("--topk", type=int, default=int(e("TOPK", "64")))
    ap.add_argument("--outdir", default=e("OUTDIR", OUTDIR_DEFAULT))
    ap.add_argument("--heldout", type=int, default=int(e("HELDOUT", "120")))
    ap.add_argument("--eval-every", type=int, default=int(e("EVAL_EVERY", "300")))
    ap.add_argument("--log-every", type=int, default=int(e("LOG_EVERY", "25")))
    ap.add_argument("--chat-template", default=e("CHAT_TEMPLATE", TEMPLATE_DEFAULT))
    ap.add_argument("--min-avail-gb", type=float, default=float(e("MIN_AVAIL_GB", "4")))
    ap.add_argument("--load-rss-gb", type=float, default=float(e("LOAD_RSS_GB", "30")))
    ap.add_argument("--grad-ckpt", action="store_true", default=e("GRAD_CKPT", "0") == "1",
                    help="recompute the MTP layer in backward (saves ~1.3 GB per draft step)")
    ap.add_argument("--smoke", action="store_true", help="load, 2 train steps, 1 tiny eval, save nothing")
    ap.add_argument("--eval-only", action="store_true", help="baselines (a) and (b) only")
    a = ap.parse_args(argv)
    w = [float(v) for v in a.step_w.split(",") if v.strip()]
    if a.nsteps < 1 or len(w) < a.nsteps:
        ap.error(f"STEP_W has {len(w)} weights, NSTEPS={a.nsteps} needs at least that many")
    a.step_w = w[:a.nsteps]
    if a.lr_head is None:
        a.lr_head = 0.1 * a.lr
    if a.smoke and a.eval_only:
        ap.error("--smoke and --eval-only are exclusive")
    return a


def load_rows(path):
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if (r.get("prompt") and r.get("content")) or (r.get("prompt_text") and r.get("gen_text")):
                    rows.append(r)
                else:
                    bad += 1
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def load_main_model(model_dir):
    from transformers import CompressedTensorsConfig, Qwen3_5ForConditionalGeneration

    try:
        qcfg = CompressedTensorsConfig(dequantize=True)           # transformers >= 5: run_compressed is deprecated
    except TypeError:
        qcfg = CompressedTensorsConfig(run_compressed=False)
    # device_map="cuda": the packed NVFP4 weights go straight to the GPU and compressed-tensors decompresses them in
    # place there. The watchdog aborts if host RSS passes LOAD_RSS_GB meanwhile (a CPU-side BF16 copy is 54 GB).
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda", quantization_config=qcfg, attn_implementation="sdpa")
    model.eval().requires_grad_(False)
    n_q, freed = disable_qdq(model)
    return model, n_q, freed


def disable_qdq(model):
    """compressed-tensors wraps every quantized Linear with an on-the-fly QDQ forward. After decompression the weight
    QDQ is skipped, but the scheme's NVFP4 input_activations are still fake-quantized (W4A4 emulation). vLLM serves
    this checkpoint with Marlin (W4A16: BF16 activations), so the teacher must run the plain BF16 forward.
    The decompressed per-group weight_scale (1/16 of each weight, ~3 GiB in total) is then unused: free it."""
    n, freed = 0, 0
    for m in model.modules():
        if getattr(m, "quantization_scheme", None) is not None:
            m.quantization_enabled = False
            n += 1
            ws = m._parameters.get("weight_scale")
            if ws is not None:
                freed += ws.numel() * ws.element_size()
                del m._parameters["weight_scale"]
    return n, freed


def run(args):
    t_run = time.time()
    mode = "smoke" if args.smoke else "eval-only" if args.eval_only else "train"
    log(f"mtp_train_v2 start, mode {mode}, pid {os.getpid()}")
    log("config", json.dumps({k: v for k, v in vars(args).items()}))
    torch.manual_seed(0)
    random.seed(0)

    # ---- 0. everything cheap first, while vLLM keeps serving (fail fast)
    with open(os.path.join(args.model, "config.json")) as f:
        cfg = json.load(f)
    vocab_size = cfg.get("text_config", cfg)["vocab_size"]
    d2t, t2d = build_vocab_maps(torch.load(args.vocab, map_location="cpu", weights_only=True), vocab_size)
    log(f"draft vocab {args.vocab}: {d2t.numel()} ids of {vocab_size}")
    if os.path.realpath(args.outdir) == os.path.realpath(args.model):
        raise RuntimeError("OUTDIR must not be the model directory (the original model_mtp.safetensors stays untouched)")
    if not (args.smoke or args.eval_only):
        os.makedirs(args.outdir, exist_ok=True)
        probe = os.path.join(args.outdir, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        free_gb = os.statvfs(args.outdir).f_bavail * os.statvfs(args.outdir).f_frsize / GIB
        if free_gb < 3:
            raise RuntimeError(f"only {free_gb:.1f} GiB free under {args.outdir}; the outputs need about 1.3 GiB")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    for name in ("<|im_end|>", "</think>", "<think>"):
        tid = tok.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0 and t2d[tid] < 0:
            log(f"WARNING: {name} (id {tid}) is not in the draft vocabulary: it can never be drafted")
    template = None
    if args.chat_template and os.path.exists(args.chat_template):
        with open(args.chat_template) as f:
            template = f.read()
    else:
        log(f"WARNING: chat template {args.chat_template!r} not found; using v1's literal format (no system prompt)")

    rows, bad = load_rows(args.data)
    random.Random(0).shuffle(rows)                  # same shuffle as build_draft_vocab.py (vocab from the first 90%)
    eval_build = None
    if args.eval_data:
        eval_rows, bad_e = load_rows(args.eval_data)
        eval_docs, st_e = build_docs(eval_rows, tok, template, args.maxlen)
        eval_docs = eval_docs[:args.heldout]
        eval_build = {"bad_lines": bad_e, **st_e}
        train_docs, st = build_docs(rows, tok, template, args.maxlen)
    else:
        docs, st = build_docs(rows, tok, template, args.maxlen)
        if len(docs) <= args.heldout + 10:
            raise RuntimeError(f"only {len(docs)} usable docs")
        eval_docs, train_docs = docs[-args.heldout:], docs[:-args.heldout]   # tail: disjoint from the vocab split
    packs = pack_docs(train_docs, args.maxlen)
    n_tok = sum(len(p[0]) for p in packs)
    n_resp = sum(sum(p[1]) for p in packs)
    data_info = {"data": args.data, "bad_lines": bad, "build": st, "train_docs": len(train_docs),
                 "train_packs": len(packs), "train_tokens": n_tok, "train_generated_tokens": n_resp,
                 "heldout_docs": len(eval_docs), "eval_data": args.eval_data or None, "eval_build": eval_build,
                 "kinds": {k: sum(1 for d in train_docs if d.kind == k) for k in sorted({d.kind for d in train_docs})}}
    log("data", json.dumps(data_info))
    if args.smoke:
        eval_docs = eval_docs[:4]
        packs = sorted(packs, key=lambda p: -len(p[0]))[:2]          # the longest packs: representative peak memory
    total_steps = 2 if args.smoke else len(packs) * args.epochs

    # ---- 1. stop vLLM, load the main model (NVFP4 decompressed to BF16 on the GPU)
    s = mem_snapshot()
    log("before stop:", fmt_mem(s))
    stop_vllm()
    s = mem_snapshot()
    log("after stop:", fmt_mem(s))
    if s["avail_gb"] < 2 * args.min_avail_gb:
        raise RuntimeError(f"only {s['avail_gb']:.1f} GiB host RAM available after stopping vLLM")
    t0 = time.time()
    WATCHDOG.phase = "load"
    model, n_qdq, freed = load_main_model(args.model)
    WATCHDOG.phase = "run"
    t_load = time.time() - t0
    log(f"activation QDQ disabled on {n_qdq} compressed-tensors modules (teacher = W4A16 like vLLM's Marlin path), "
        f"{freed / GIB:.1f} GiB of unused weight_scale freed")
    if n_qdq == 0:
        log("WARNING: no compressed-tensors modules found; check that the model really is the NVFP4 checkpoint")
    lm = model.model.language_model
    try:
        model.model.visual = None                 # vision tower is never used here
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    torch.cuda.empty_cache()
    bad_dtypes = {str(p.dtype) for p in lm.parameters() if p.dtype not in (torch.bfloat16, torch.float32)}
    if bad_dtypes:
        raise RuntimeError(f"language model still has non-float params {bad_dtypes}: decompression incomplete")
    log(f"main model loaded in {t_load / 60:.1f} min | GPU {torch.cuda.memory_allocated() / GIB:.1f} GiB | "
        + fmt_mem(mem_snapshot()))
    tcfg = model.config.text_config
    lm_head_w = model.lm_head.weight
    assert lm_head_w.shape == (vocab_size, tcfg.hidden_size) and lm_head_w.dtype == torch.bfloat16, lm_head_w.shape

    # ---- 2. MTP (original Qwen weights) + draft head from lm_head rows
    from safetensors.torch import load_file

    orig = load_file(os.path.join(args.model, "model_mtp.safetensors"))
    assert all(k.startswith("mtp.") for k in orig), sorted(orig)[:3]
    orig_meta = {k: (v.dtype, tuple(v.shape)) for k, v in orig.items()}
    with torch.device("cuda"):
        mtp = make_mtp(tcfg).float()
    mtp.load_state_dict({k[4:]: v.to("cuda", torch.float32) for k, v in orig.items()}, strict=True)
    del orig
    mtp.train()
    d2t_g = d2t.cuda()
    head_w = nn.Parameter(lm_head_w.detach()[d2t_g].float().clone())       # fp32 master
    ctx = Ctx(lm, lm_head_w, mtp, head_w, d2t, t2d, args.topk, args.grad_ckpt)
    log(f"MTP {sum(p.numel() for p in mtp.parameters()) / 1e6:.1f}M params, draft head {tuple(head_w.shape)} | "
        f"GPU {torch.cuda.memory_allocated() / GIB:.1f} GiB")

    # ---- 3. held-out cache and baselines
    t0 = time.time()
    cache, dstats = cache_eval_docs(ctx, eval_docs)
    t_cache = time.time() - t0
    log(f"held-out cache: {len(cache)} docs in {t_cache:.0f}s | data stats {json.dumps(dstats)}")
    if dstats["data_greedy_frac"] < 0.5:
        raise RuntimeError(f"the loaded model reproduces only {dstats['data_greedy_frac']:.1%} of its own generated "
                           "tokens greedily: loading/decompression looks broken, aborting before training")
    t0 = time.time()
    ev_a = evaluate(ctx, cache, "full", args.nsteps)
    t_eval = time.time() - t0
    log(fmt_eval("(a) original MTP, FULL-vocab head", ev_a, args.nsteps))
    ev_b = evaluate(ctx, cache, "draft", args.nsteps)
    log(fmt_eval("(b) original MTP, draft-vocab head (lm_head rows)", ev_b, args.nsteps))
    result = {"data_stats": dstats, "baseline_a_full_vocab": ev_a, "baseline_b_draft_vocab": ev_b}
    if args.eval_only:
        log("RESULT", json.dumps(result))
        return

    # ---- 4. training
    params = list(mtp.parameters()) + [head_w]
    opt = torch.optim.AdamW([{"params": list(mtp.parameters()), "lr": args.lr, "base_lr": args.lr},
                             {"params": [head_w], "lr": args.lr_head, "base_lr": args.lr_head}],
                            betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    warm = min(50, total_steps // 10)
    gates = gate_steps(args.nsteps)
    best, best_ev, best_step, history = None, None, None, []
    win = {"t": 0.0, "tok": 0, "n": 0, "loss": 0.0, "kl": [[] for _ in range(args.nsteps)],
           "agree": [[] for _ in range(args.nsteps)], "chain": new_counts(args.nsteps), "gn": 0.0, "skipped": 0}
    step, step_times = 0, []
    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    log(f"training: {total_steps} steps, warmup {warm}, LR {args.lr} (MTP) / {args.lr_head} (head), "
        f"STEP_W {args.step_w}, gate at {gates} drafts")
    for ep in range(args.epochs if not args.smoke else 1):
        order = list(range(len(packs)))
        if not args.smoke:
            random.Random(ep).shuffle(order)
        for pi in order:
            if step >= total_steps:
                break
            ids, resp_l = packs[pi]
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale(step, total_steps, warm)
            t0 = time.time()
            mp = main_pass(ctx, ids)
            resp = torch.tensor(resp_l, dtype=torch.bool, device="cuda")
            outs = run_chain(mtp, mp["h"], mp["e"], mp["pe"], args.nsteps, args.grad_ckpt)
            loss, st_k = mtp_losses(outs, head_w.to(torch.bfloat16), ctx.d2t, mp["q"], mp["di"], mp["valid"],
                                    mp["targ"], resp, args.step_w)
            opt.zero_grad(set_to_none=True)
            if loss is None or not torch.isfinite(loss):
                win["skipped"] += 1
                log(f"step {step + 1}: skipped (loss {loss})")
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if torch.isfinite(gn):
                    opt.step()
                    win["gn"] += float(gn)
                else:
                    win["skipped"] += 1
                    log(f"step {step + 1}: non-finite grad norm, update skipped")
                win["loss"] += float(loss.detach())
            torch.cuda.synchronize()
            dt = time.time() - t0
            step += 1
            step_times.append(dt)
            win["t"] += dt; win["tok"] += len(ids); win["n"] += 1
            for k in range(args.nsteps):
                if k < len(st_k["kl"]) and st_k["count"][k]:
                    win["kl"][k].append(st_k["kl"][k]); win["agree"][k].append(st_k["agree"][k])
            add_counts(win["chain"], train_chain_counts(st_k["matches"], resp, args.nsteps))
            del mp, outs, loss, st_k
            if step % args.log_every == 0 or args.smoke or step == total_steps:
                n = max(1, win["n"])
                ch = finalize_counts(win["chain"], args.nsteps)
                mean = lambda v: round(sum(v) / len(v), 4) if v else float("nan")  # noqa: E731
                eta = (total_steps - step) * win["t"] / n / 60
                log(f"step {step}/{total_steps} | {win['t'] / n:.2f} s/step {win['tok'] / max(1e-9, win['t']):.0f} tok/s"
                    f" | loss {win['loss'] / n:.4f} | kl {[mean(v) for v in win['kl']]}"
                    f" | agree {[mean(v) for v in win['agree']]} | train acc_len@{gates} "
                    f"{[ch['acc_len'][str(m)] for m in gates]} per_pos {ch['per_pos']}"
                    f" | lr {opt.param_groups[0]['lr']:.2e} | gnorm {win['gn'] / n:.3f} | skipped {win['skipped']}"
                    f" | GPU {torch.cuda.memory_allocated() / GIB:.1f} (peak {torch.cuda.max_memory_allocated() / GIB:.1f})"
                    f" GiB | RAM avail {mem_snapshot()['avail_gb']:.1f} GiB | eta {eta:.0f} min")
                win = {"t": 0.0, "tok": 0, "n": 0, "loss": 0.0, "kl": [[] for _ in range(args.nsteps)],
                       "agree": [[] for _ in range(args.nsteps)], "chain": new_counts(args.nsteps), "gn": 0.0,
                       "skipped": 0}
            if (not args.smoke) and (step % args.eval_every == 0 or step == total_steps):
                ev = evaluate(ctx, cache, "draft", args.nsteps)
                ok = gate_passes(ev, ev_b, gates)
                history.append({"step": step, "eval": ev, "beats_b": ok})
                log(fmt_eval(f"EVAL step {step} (c)", ev, args.nsteps) + f" | beats (b) at {gates}: {ok}")
                if ok and better(ev, best_ev, args.nsteps):
                    best, best_ev, best_step = snapshot(ctx), ev, step
                    log(f"new best at step {step}: acc_len@{args.nsteps} {ev['acc_len'][str(args.nsteps)]}")
    t_train = time.time() - t_train

    if args.smoke:
        t0 = time.time()
        ev_c = evaluate(ctx, cache, "draft", args.nsteps)
        log(fmt_eval("(c) after 2 smoke steps", ev_c, args.nsteps))
        mtp_dict, head_dict = build_save_dicts(snapshot(ctx), orig_meta)   # dry run: key/shape/dtype checks only
        log(f"save dry run OK: {len(mtp_dict)} MTP tensors (keys == original), draft head "
            f"{tuple(head_dict['weight'].shape)} {head_dict['weight'].dtype}; nothing written")
        full_steps = data_info["train_packs"] * args.epochs
        s_step = step_times[-1]
        n_evals = full_steps // args.eval_every + 1
        per_doc = args.heldout / max(1, len(cache))
        eval_full = t_eval * per_doc                     # (a) is the slowest eval (full 248k head)
        log(f"SMOKE OK | step times {[round(t, 2) for t in step_times]} s | peak GPU "
            f"{torch.cuda.max_memory_allocated() / GIB:.1f} GiB | min MemAvailable seen "
            f"{WATCHDOG.min_avail_seen:.1f} GiB | projected full run: load {t_load / 60:.0f} min + cache/baselines "
            f"~{(t_cache * per_doc + 2 * eval_full) / 60:.0f} min + {full_steps} steps x {s_step:.2f} s = "
            f"{full_steps * s_step / 60:.0f} min + {n_evals} evals ~{n_evals * eval_full / 60:.0f} min")
        return

    # ---- 5. gate and save
    result.update({"history": history, "best_step": best_step, "trained_c": best_ev, "gate_steps": gates})
    if best is None:
        log(f"NO IMPROVEMENT: no evaluation beat (b) at {gates} drafts. Nothing saved.")
        log("RESULT", json.dumps(result))
        return
    mtp_dict, head_dict = build_save_dicts(best, orig_meta)
    meta = {"date": now("%d-%m-%Y %H:%M"), "script": os.path.abspath(__file__), "config": vars(args),
            "model": args.model, "vocab": {"path": args.vocab, "size": int(d2t.numel()),
                                           "note": "draft_vocab.pt holds absolute full-vocab ids (draft index -> id). "
                                                   "vLLM's LocalArgmaxMixin / EAGLE3 d2t is an OFFSET table: "
                                                   "d2t_offsets = ids - arange(V)."},
            "data": data_info, "timing_min": {"load": round(t_load / 60, 1), "train": round(t_train / 60, 1),
                                               "total": round((time.time() - t_run) / 60, 1)},
            "mtp_keys": sorted(mtp_dict), **result}
    paths = save_outputs(args.outdir, args.model, mtp_dict, head_dict, d2t, meta)
    log(f"SAVED (best step {best_step}): acc_len@{gates} (b) {[ev_b['acc_len'][str(m)] for m in gates]} -> (c) "
        f"{[best_ev['acc_len'][str(m)] for m in gates]} | files {json.dumps(paths)}")
    log("The original model_mtp.safetensors is untouched; deployment is a separate manual step.")
    log("RESULT", json.dumps(result))


def main(argv=None):
    global WATCHDOG
    args = parse_args(argv)
    signal.signal(signal.SIGUSR1, _on_usr1)
    signal.signal(signal.SIGTERM, _on_term)
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:       # keep nohup's "ignore SIGHUP"
        signal.signal(signal.SIGHUP, _on_term)
    WATCHDOG = RamWatchdog(args.min_avail_gb, args.load_rss_gb)
    WATCHDOG.start()
    code, why = 1, "error"
    try:
        run(args)
        code, why = 0, "finished"
    except RamAbort as ex:
        STATE["shutdown"] = True
        code, why = 3, "RAM watchdog abort"
        log("ABORTED by the RAM watchdog:", ex)
    except KeyboardInterrupt:
        STATE["shutdown"] = True
        code, why = 130, "interrupted"
        log("interrupted")
    except SystemExit as ex:
        STATE["shutdown"] = True
        code, why = 143, "terminated"
        log("terminated:", ex)
    except BaseException:  # noqa: BLE001
        STATE["shutdown"] = True
        log("FAILED")
        traceback.print_exc()
    finally:
        STATE["shutdown"] = True
        WATCHDOG.stop()
        if STATE["vllm_stopped"]:
            schedule_vllm_restart(why)
        log(f"exit {code} ({why})")
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)                                   # fast exit: the driver frees the GPU, then vLLM starts


if __name__ == "__main__":
    main()
