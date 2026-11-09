#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""test_lmhead_screen_cpu.py (Horizon Tech, 30-09-2026): CPU tests of the exact lm_head screen (no CUDA context).

Run with the vLLM 0.30 venv python and the candidate plugin package first on PYTHONPATH:
  PYTHONPATH=<pkg>:<tl-pkgs> $HK_VENV/bin/python test_lmhead_screen_cpu.py [test ...]
Each test runs in a fresh subprocess (switches are read at import / install time).

  bound_and_ties        synthetic rows with exact BF16 ties at the max and at the 20th value: every tied token selected,
                        |screen - fp32| <= B everywhere, sparse first-index argmax == full
  kernel_emulation      the kernels' algorithm in torch (tile threshold, compaction with capacity, overflow rows in
                        full, sparse out) on synthetic data, including forced overflow
  vllm_masks_equal      vLLM's own apply_top_k_top_p_pytorch / apply_top_k_only: top-k masks bit-identical on the
                        sparse and full logits; top-k + top-p 0.95 identical up to swaps between exactly tied tokens at
                        the top-p cut (vLLM's unstable sort; same in production), temperature 0.04 / 1.0
  eligibility           the per-step rule on fake sampler state: every stock reason and both screen modes
  real_class_hooks      install() on vLLM 0.30's REAL LogitsProcessor and V2 GPUModelRunner: wrappers installed,
                        signatures unchanged, dispatch (ctx greedy / topk / none, draft argmax, capture fallback)
                        with a stub screen; the real Sampler source still has every attribute the rule reads
  extra_compare         lmscreen_extra.py compare on synthetic run data: every gate passes on good data and fails on
                        a missing self-test, a non-bitwise M, an unused screen, low memory, a greedy or acceptance drop
  real_weights          (needs the checkpoint + captured hidden states) 192 real rows through the kernel emulation:
                        greedy and top-20 sets exact against the fp32 reference on the real BF16 head
"""
import os
import subprocess
import sys

TESTS = {}
HERE = os.path.dirname(os.path.abspath(__file__))


def test(fn):
    TESTS[fn.__name__] = fn
    return fn


PRE = r'''
import os, sys, types, math, torch
torch.cuda.is_available = lambda: False
torch.manual_seed(0)
from horizon_kernels import lmhead_screen as ls
'''

EMU = r'''
def emulate(x, W, q, sc, P, k=None, kmax=64, tile=128):
    """The kernels' algorithm (tilelang.contrib.sm120_fp8.lmhead_screen.Screen.logits) in torch on CPU: screen,
    tile threshold, compaction with capacity kmax, recheck of the selected tokens, full recompute of overflow rows.
    Recheck / full values come from ref (the fp32 reference rounded to BF16), standing in for the GPU GEMM."""
    ref = (x.float() @ W.float().t()).to(torch.bfloat16)
    s = ls.screen_fp8_reference(x, q, sc)
    xn = ls.row_norms(x)
    lower, upper = ls.bounds(s, xn, P)
    M, V = s.shape
    nt = (V + tile - 1) // tile
    TM = torch.nn.functional.pad(lower, (0, nt * tile - V), value=-math.inf).view(M, nt, tile).amax(-1)
    if k is None:
        thr = TM.amax(-1)
    else:
        vals = TM.topk(int(k.max()), dim=-1).values
        thr = vals.gather(1, (k - 1).view(-1, 1)).view(-1)
    keep = ls.up(upper) >= ls.dn(thr)[:, None]
    cnt = keep.sum(-1)
    out = torch.full_like(ref, -math.inf)
    for r in range(M):
        if cnt[r] > kmax:
            out[r] = ref[r]
        else:
            idx = keep[r].nonzero().flatten()
            out[r, idx] = ref[r, idx]
    return out, ref, cnt
'''


@test
def bound_and_ties():
    return PRE + r'''
V, K, M = 4096, 256, 24
W = (torch.randn(V, K) * 0.05).to(torch.bfloat16)
# make exact BF16 ties: duplicate some weight rows (identical dot products in any order)
W[7] = W[3]; W[100] = W[3]; W[2000] = W[1500]
x = (torch.randn(M, K) * 2).to(torch.bfloat16)
x[:8] = (W[3].float() * 40).to(torch.bfloat16)          # rows whose max is the tied group {3, 7, 100}
q, sc = ls.quant_fp8_rows(W)
P = ls.token_terms(W, ls.dequant_fp8_rows(q, sc))
s = ls.screen_fp8_reference(x, q, sc)
ref32 = x.float() @ W.float().t()
ref = ref32.to(torch.bfloat16)
xn = ls.row_norms(x)
lo, hi = ls.bounds(s, xn, P)
assert bool(((ref32 >= lo) & (ref32 <= hi)).all()), "bound violated"
for k in (1, 5, 20):
    sel, T = ls.select(s, xn, P, k)
    exact = ls.topk_set_mask(ref, k)
    assert bool((exact <= sel).all()), ("top-k set not inside the selection", k)
    sp = ls.sparse_logits(ref, sel)
    assert bool((ls.topk_set_mask(sp, k) == exact).all()), k
assert bool((ls.first_argmax(ls.sparse_logits(ref, ls.select(s, xn, P, 1)[0])) == ls.first_argmax(ref)).all())
am = ls.first_argmax(ref[:8])
assert bool((am == 3).all()), am                         # first index among the tied maxima {3, 7, 100}
sel1 = ls.select(s[:8], xn[:8], P, 1)[0]
assert bool(sel1[:, [3, 7, 100]].all())
t = ls.token_terms(W, ls.dequant_fp8_rows(q, sc))
t64 = (W.double() - ls.dequant_fp8_rows(q, sc).double()).norm(dim=1)
assert bool((t.double() >= t64).all())                    # P rounded up, never below the float64 error norm
print("ok")
'''


@test
def kernel_emulation():
    return PRE + EMU + r'''
V, K, M = 8192, 256, 64
W = (torch.randn(V, K) * 0.05).to(torch.bfloat16)
x = (torch.randn(M, K) * 3).to(torch.bfloat16)
x[:4] *= 0.05                                            # flat rows: many candidates -> overflow at kmax 64
q, sc = ls.quant_fp8_rows(W)
P = ls.token_terms(W, ls.dequant_fp8_rows(q, sc))
for k in (None, torch.full((M,), 20, dtype=torch.int64)):
    for kmax in (64, 100000):
        out, ref, cnt = emulate(x, W, q, sc, P, k=k, kmax=kmax)
        kk = 1 if k is None else 20
        assert bool((ls.topk_set_mask(out, kk) == ls.topk_set_mask(ref, kk)).all()), (kk, kmax)
        assert bool((ls.first_argmax(out) == ls.first_argmax(ref)).all())
        if kmax == 64:
            assert int((cnt > kmax).sum()) > 0, "expected forced overflow rows"
# tile threshold never above the exact k-th lower bound (so the selection is a superset)
s = ls.screen_fp8_reference(x, q, sc); lo, _ = ls.bounds(s, ls.row_norms(x), P)
for k in (1, 5, 20):
    for tile in (16, 128):
        assert bool((ls.tile_threshold(lo, k, tile) <= ls.threshold(lo, k)).all())
    assert bool((ls.tile_threshold(lo, 1, 128) == ls.threshold(lo, 1)).all())
print("ok")
'''


@test
def vllm_masks_equal():
    return PRE + EMU + r'''
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch, apply_top_k_only
V, K, M = 8192, 256, 32
W = (torch.randn(V, K) * 0.05).to(torch.bfloat16)
x = (torch.randn(M, K) * 3).to(torch.bfloat16)
q, sc = ls.quant_fp8_rows(W)
P = ls.token_terms(W, ls.dequant_fp8_rows(q, sc))
k = torch.full((M,), 20, dtype=torch.int64)
out, ref, cnt = emulate(x, W, q, sc, P, k=k, kmax=100000)
def same_up_to_ties(ma, mb):
    """Equal kept-value multisets per row, and every differing position is a swap between EQUAL values (vLLM's
    top-p cut sorts with an unstable sort: which of two exactly tied tokens survives the cut depends on the rest of
    the row, in production too; the sampled distribution is the same)."""
    fa, fb = torch.isfinite(ma), torch.isfinite(mb)
    if not torch.equal(fa.sum(-1), fb.sum(-1)):
        return False
    for r in range(ma.shape[0]):
        if not torch.equal(ma[r][fa[r]].sort().values, mb[r][fb[r]].sort().values):
            return False
    return True
ties = 0
for T in (1.0, 0.04):
    a = ref.float() / T
    b = out.float() / T
    ma = apply_top_k_top_p_pytorch(a.clone(), k, None)
    mb = apply_top_k_top_p_pytorch(b.clone(), k, None)
    assert torch.equal(ma, mb), ("top-k masked logits differ", T)
    ma = apply_top_k_top_p_pytorch(a.clone(), k, torch.full((M,), 0.95))
    mb = apply_top_k_top_p_pytorch(b.clone(), k, torch.full((M,), 0.95))
    ties += int((~torch.eq(ma, mb) & ~(torch.isinf(ma) & torch.isinf(mb))).any(-1).sum())
    assert same_up_to_ties(ma, mb), ("top-k + top-p differ beyond tie swaps", T)
    assert torch.equal(apply_top_k_only(a.clone(), k), apply_top_k_only(b.clone(), k))
print("ok (rows with a tie swap at the top-p cut: %d)" % ties)
'''


FAKE = r'''
import numpy as np
NO = -1
def U(a): return types.SimpleNamespace(np=np.asarray(a))
def make(n=3, temp=0.0, top_k=248320, logprobs=None, bias=False, pen=False, bad=0, min_p=0.0, tb=False):
    ss = types.SimpleNamespace(temperature=U([temp] * n), top_k=U([top_k] * n), min_p=U([min_p] * n))
    class Sampler:
        compute_nans = False; trace_replay_state = None; return_sampling_mask = False
        def get_logprobs_dims(self, idx): return None if logprobs is None else (logprobs, 0)
    Sampler.__name__ = "Sampler"
    s = Sampler()
    s.sampling_states = ss
    s.logit_bias_state = types.SimpleNamespace(use_logit_bias=np.array([bias] * n))
    s.penalties_state = types.SimpleNamespace(use_penalty=np.array([pen] * n))
    s.bad_words_state = types.SimpleNamespace(num_bad_words=U([bad] * n))
    s.thinking_budget_state = types.SimpleNamespace(enabled=True, use_thinking_budget=np.array([tb] * n))
    runner = types.SimpleNamespace(sampler=s, batch_sharder=None,
                                   rejection_sampler=types.SimpleNamespace(synthetic_conditional_rates=None, watermark_key=None))
    ib = types.SimpleNamespace(idx_mapping_np=np.arange(n), num_reqs=n, cu_num_logits_np=np.arange(0, 4 * n + 1, 4))
    return runner, ib
'''


@test
def eligibility():
    return PRE + FAKE + r'''
E = ls.eligibility
r, ib = make(); assert E(r, ib, None)[0] == "greedy"
assert E(r, ib, object())[3] == "grammar"
r, ib = make(logprobs=5); assert E(r, ib, None)[3] == "logprobs"
for kw in (dict(bias=True), dict(pen=True), dict(bad=2), dict(tb=True)):
    r, ib = make(**kw); assert E(r, ib, None)[3] == "logits_processing", kw
r, ib = make(min_p=0.1); assert E(r, ib, None)[3] == "min_p"
r, ib = make(temp=1.0, top_k=20); assert E(r, ib, None)[3] == "sampling"          # top-k mode off by default
os.environ["HORIZON_LMHEAD_SCREEN_TOPK"] = "32"
mode, k_rows, kmax, why = E(r, ib, None); assert mode == "topk" and kmax == 20 and len(k_rows) == 12, (mode, why)
r, ib = make(temp=1.0, top_k=248320); assert E(r, ib, None)[3] == "sampling"      # no top_k: full distribution
r, ib = make(n=9, temp=1.0, top_k=20); assert E(r, ib, None)[3] == "sampling"     # 36 rows > TOPK_ROWS 32
r, ib = make(n=17); assert E(r, ib, None)[3] == "rows"                            # 68 rows > MAXM 64 (default)
r, ib = make(); r.rejection_sampler.synthetic_conditional_rates = object(); assert E(r, ib, None)[3] == "rejection_method"
r, ib = make(); r.batch_sharder = object(); assert E(r, ib, None)[3] == "batch_sharder"
r, ib = make(); ls._ST["ok_m"] = {4, 8}; assert E(r, ib, None)[3] == "rows"         # 12 rows not bit-verified at load
r, ib = make(n=2); assert E(r, ib, None)[0] == "greedy"                           # 8 rows verified
print("ok")
'''


@test
def real_class_hooks():
    return "import os; os.environ['HORIZON_LMHEAD_SCREEN']='1'\n" + PRE + r'''
import inspect, types
from vllm.model_executor.layers.logits_processor import LogitsProcessor
import vllm.v1.worker.gpu.model_runner as mr
from vllm.v1.worker.gpu.sample import sampler as smp_mod, states as st_mod
o_apply, o_top, o_sample = LogitsProcessor._apply_head, LogitsProcessor.get_top_tokens, mr.GPUModelRunner.sample
ls._log = lambda: types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
ls.install()
for new, old in ((LogitsProcessor._apply_head, o_apply), (LogitsProcessor.get_top_tokens, o_top),
                 (mr.GPUModelRunner.sample, o_sample)):
    assert getattr(new, "_horizon_screen", False) and new is not old
    assert list(inspect.signature(new).parameters) == list(inspect.signature(old).parameters), (new, old)
ls.install()                                                            # idempotent
src = inspect.getsource(smp_mod.Sampler) + inspect.getsource(st_mod.SamplingStates)
for attr in ("compute_nans", "trace_replay_state", "return_sampling_mask", "get_logprobs_dims", "logit_bias_state",
             "penalties_state", "bad_words_state", "thinking_budget_state", "sampling_states", "min_p", "top_k",
             "temperature"):
    assert attr in src, attr
import vllm.v1.worker.gpu.sample.logit_bias as lb, vllm.v1.worker.gpu.sample.penalties as pe
import vllm.v1.worker.gpu.sample.bad_words as bw, vllm.v1.worker.gpu.sample.thinking_budget as tbm
for mod, attr in ((lb, "use_logit_bias"), (pe, "use_penalty"), (bw, "num_bad_words"), (tbm, "use_thinking_budget")):
    assert attr in inspect.getsource(mod), attr
import vllm.v1.worker.gpu.input_batch as ibm
for attr in ("idx_mapping_np", "cu_num_logits_np", "num_reqs"):
    assert attr in inspect.getsource(ibm.InputBatch), attr
import vllm.v1.worker.gpu.spec_decode.rejection_sampler as rsm
assert "synthetic_conditional_rates" in inspect.getsource(rsm.RejectionSampler)

# dispatch with a stub screen on CPU
V, K = 512, 64
calls = []
class StubScreen:
    V = 512
    def logits(self, x, k=None, k_host_max=None):
        calls.append(("logits", x.shape[0], None if k is None else int(k_host_max)))
        o = torch.full((x.shape[0], V), -math.inf, dtype=torch.bfloat16); o[:, 5] = 1.0; return o
    def full(self, x):
        calls.append(("full", x.shape[0])); return torch.zeros(x.shape[0], V, dtype=torch.bfloat16)
head = types.SimpleNamespace(tp_size=1, weight=torch.zeros(V, K, dtype=torch.bfloat16),
                             shard_indices=types.SimpleNamespace(num_org_vocab_padding=0, org_vocab_start_index=0))
head.quant_method = types.SimpleNamespace(apply=lambda l, x, bias=None: (calls.append(("stock", x.shape[0])),
                                          torch.zeros(x.shape[0], V, dtype=torch.bfloat16))[1])
lp = LogitsProcessor.__new__(LogitsProcessor)
torch.nn.Module.__init__(lp)
lp.scale, lp.soft_cap, lp.head_dtype, lp.logits_as_input, lp.org_vocab_size = 1.0, None, None, False, V
ls._ST.update(screen=StubScreen(), tried=True, head=head, ok_m=None)
h = torch.zeros(3, K, dtype=torch.bfloat16)
lp._apply_head(head, h, None); assert calls[-1] == ("stock", 3), calls                # no ctx: stock
ls._ST["ctx"] = ("greedy", None, 1); lp._apply_head(head, h, None); assert calls[-1] == ("logits", 3, None)
assert ls._ST["ctx"] is None                                                            # consumed once
ls._ST["ctx"] = ("topk", [20, 20, 1], 20); lp._apply_head(head, h, None); assert calls[-1] == ("logits", 3, 20)
ls._ST["ctx"] = ("own", None, 0); lp._apply_head(head, h, None); assert calls[-1] == ("full", 3)
other = types.SimpleNamespace(**vars(head))
ls._ST["ctx"] = ("greedy", None, 1); lp._apply_head(other, h, None); assert calls[-1] == ("stock", 3)  # other head
ids = lp.get_top_tokens(head, h); assert calls[-1] == ("logits", 3, None) and ids.tolist() == [5, 5, 5]
os.environ["HORIZON_LMHEAD_SCREEN_DRAFT"] = "0"
lp.get_top_tokens(head, h); assert calls[-1] == ("stock", 3)
os.environ["HORIZON_LMHEAD_SCREEN_DRAFT"] = "1"
ls._ST["ok_m"] = {1, 2}; lp.get_top_tokens(head, h); assert calls[-1] == ("stock", 3)   # M 3 not verified
ls._ST["ok_m"] = None
# first use during capture never builds the screen (compiling inside a graph capture would break it)
ls._ST.update(screen=None, tried=False, head=None)
torch.cuda.is_available = lambda: True
torch.cuda.is_current_stream_capturing = lambda: True
built = []
ls._attach = lambda lm, dev: built.append(1)
assert ls._screen_for(head, h) is None and not built and ls._ST["tried"] is False
# the sample() wrapper sets the ctx for eligible steps and clears it afterwards
torch.cuda.is_current_stream_capturing = lambda: False
seen = {}
def fake_orig(self, hs, ib, go):
    seen["ctx"] = ls._ST["ctx"]; return "out"
os.environ["HORIZON_LMHEAD_SCREEN_LOG_EVERY"] = "0"
''' + FAKE + r'''
mr.GPUModelRunner.sample = fake_orig
LogitsProcessor._apply_head, LogitsProcessor.get_top_tokens = o_apply, o_top      # re-install over fake_orig
ls.install()
r, ib = make()
assert mr.GPUModelRunner.sample(r, torch.zeros(12, K), ib, None) == "out" and seen["ctx"] == ("greedy", None, 1)
assert ls._ST["ctx"] is None
r, ib = make(logprobs=3)
mr.GPUModelRunner.sample(r, torch.zeros(12, K), ib, None); assert seen["ctx"] is None
assert ls._ST["n"]["stock:logprobs"] == 1 and ls._ST["n"]["mode:greedy"] == 1, ls._ST["n"]
print("ok")
'''


@test
def real_weights():
    return PRE + EMU + r'''
import glob
from safetensors import safe_open
files = sorted(glob.glob("$HK_WORK/formats/s1/mainh_*.pt"))
if not files or not os.path.exists("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors"):
    print("skip (no captured hidden states or checkpoint)"); raise SystemExit(0)
torch.set_num_threads(int(os.environ.get("THREADS", "32")))
x = torch.load(files[3], map_location="cpu").reshape(-1, 5120).to(torch.bfloat16)
x = x[x.float().abs().amax(-1) > 0][::21][:192]
with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cpu") as f:
    W = f.get_tensor("lm_head.weight")
q = torch.empty(W.shape, dtype=torch.float8_e4m3fn); sc = torch.empty(W.shape[0]); P = torch.empty(W.shape[0])
for r0 in range(0, W.shape[0], 16384):
    qq, ss = ls.quant_fp8_rows(W[r0:r0 + 16384]); q[r0:r0 + 16384], sc[r0:r0 + 16384] = qq, ss
    P[r0:r0 + 16384] = ls.token_terms(W[r0:r0 + 16384], ls.dequant_fp8_rows(qq, ss))
out, ref, cnt = emulate(x, W, q, sc, P, k=None, kmax=4096)
assert bool((ls.first_argmax(out) == ls.first_argmax(ref)).all())
g = int((cnt > 4096).sum()); mean1 = float(cnt.float().mean())
k = torch.full((x.shape[0],), 20, dtype=torch.int64)
out20, _, cnt20 = emulate(x, W, q, sc, P, k=k, kmax=16384, tile=16)
assert bool((ls.topk_set_mask(out20, 20) == ls.topk_set_mask(ref, 20)).all())
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
pp = torch.full((x.shape[0],), 0.95)
assert torch.equal(apply_top_k_top_p_pytorch(ref.float(), k, None), apply_top_k_top_p_pytorch(out20.float(), k, None))
ma, mb = apply_top_k_top_p_pytorch(ref.float(), k, pp), apply_top_k_top_p_pytorch(out20.float(), k, pp)
fa, fb = torch.isfinite(ma), torch.isfinite(mb)
assert torch.equal(fa.sum(-1), fb.sum(-1))
assert all(torch.equal(ma[r][fa[r]].sort().values, mb[r][fb[r]].sort().values) for r in range(ma.shape[0]))
print("ok rows %d greedy mean %.1f overflow %d | top20 mean %.1f overflow %d" % (
    x.shape[0], mean1, g, float(cnt20.float().mean()), int((cnt20 > 16384).sum())))
'''


@test
def extra_compare():
    return "import os, sys, json, tempfile, subprocess\nHERE = %r\n" % HERE + r"""
X = os.path.join(HERE, "lmscreen_extra.py")
def run(files):
    d = tempfile.mkdtemp()
    for n, v in files.items():
        json.dump(v, open(os.path.join(d, n), "w"))
    subprocess.run([sys.executable, X, "compare", d, d + "/c.json"], check=True)
    return {k: v["result"] for k, v in json.load(open(d + "/c.json"))["gates"].items()}
seq = [{"toks": [i, i + 1]} for i in range(10)]
good = {"selftest-X.json": {"summary": {"M_tested": 256, "M_ok": 256, "M_not_ok": [], "rows_used": 9, "hidden": "h",
                                         "timing_us": {"4": {"stock_us": 1600, "screen_greedy_us": 850, "screen_top20_us": 1000}}},
                             "per_M": {"4": {"overflow_rows": 0}}},
        "counters-X.json": {"counters": {"calls": 400, "mode:greedy": 380, "stock:sampling": 20}},
        "mem-X.json": {"real_free_mib": 2200}, "mem-P.json": {"real_free_mib": 3400},
        "greedy-X.json": {"seq": seq, "conc": seq}, "greedy-P.json": {"seq": seq, "conc": seq},
        "acc-X.json": {d: {"mean_accept_len": 2.5} for d in ("prose", "maths", "code")},
        "acc-P.json": {d: {"mean_accept_len": 2.5} for d in ("prose", "maths", "code")}}
r = run(good); assert all(v == "pass" for v in r.values()) and len(r) == 6, r
def bad(**kw):
    f = json.loads(json.dumps(good)); f.update(kw); return run(f)
b = dict(good); del b["selftest-X.json"]; r = run(b); assert r["selftest_ok"] == "fail" and r["h1_bitwise"] == "fail"
r = bad(**{"selftest-X.json": {"summary": dict(good["selftest-X.json"]["summary"], M_ok=250, M_not_ok=[9, 17]), "per_M": {}}})
assert r["h1_bitwise"] == "fail" and r["selftest_ok"] == "pass"
assert bad(**{"counters-X.json": {"counters": {"calls": 9, "stock:logprobs": 9}}})["screen_used"] == "fail"
assert bad(**{"mem-X.json": {"real_free_mib": 900}})["mem_free"] == "fail"
assert bad(**{"greedy-X.json": {"seq": seq[:9] + [{"toks": [0]}], "conc": seq}})["greedy_seq"] == "fail"
assert bad(**{"acc-X.json": {"prose": {"mean_accept_len": 2.47}, "maths": {"mean_accept_len": 2.5}, "code": {"mean_accept_len": 2.5}}})["acceptance"] == "fail"
# 02-10-2026: a screen with a control arm writes *-C.json (the 02-10 run reported greedy/acceptance "missing")
ctl = {k.replace("-P.json", "-C.json"): v for k, v in good.items()}
r = run(ctl); assert all(v == "pass" for v in r.values()) and len(r) == 6, r
ctl2 = dict(ctl); ctl2["greedy-C.json"] = {"seq": seq[:9] + [{"toks": [0]}], "conc": seq}
assert run(ctl2)["greedy_seq"] == "fail"
# a gate: P1 X1 X2 P2, every X arm gated (a missing X2 self-test fails)
gt = {}
for k, v in good.items():
    for a, b in (("X", "X1"), ("X", "X2"), ("P", "P1"), ("P", "P2")):
        if k.endswith(f"-{a}.json"):
            gt[k[:-(len(a) + 6)] + f"-{b}.json"] = v
r = run(gt); assert all(v == "pass" for v in r.values()) and len(r) == 6, r
gt2 = dict(gt); del gt2["selftest-X2.json"]; r = run(gt2); assert r["selftest_ok"] == "fail", r
gt3 = dict(gt); gt3["mem-X2.json"] = {"real_free_mib": 1000}; assert run(gt3)["mem_free"] == "fail"
print("ok")
"""


if __name__ == "__main__":
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    bad = 0
    only = set(sys.argv[1:])
    for name, fn in TESTS.items():
        if only and name not in only:
            continue
        p = subprocess.run([sys.executable, "-c", fn()], env=env, capture_output=True, text=True, timeout=1800)
        ok = p.returncode == 0 and p.stdout.strip().startswith(("ok", "skip"))
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {p.stdout.strip()[-300:]}")
        if not ok:
            print(p.stderr[-3000:])
    print(f"{len(TESTS) - bad if not only else len(only) - bad}/{len(only) or len(TESTS)} passed")
    sys.exit(1 if bad else 0)
