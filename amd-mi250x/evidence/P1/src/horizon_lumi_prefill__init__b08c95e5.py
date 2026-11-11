"""LUMI research only (track P1). vLLM general plugin: runs tilelang/gqa_fa.py (dense causal GQA attention) for fresh prefill steps of a D 256 model on MI250X.
Research overlay only; it does nothing unless HLP_ENABLE=1. Anything it cannot run goes to vLLM's own kernel, so a miss costs speed, not correctness.
Hook: RocmAttentionImpl.forward and TritonAttentionImpl.forward, outside horizon_lumi_decode's hook if both are on (the two never take the same step: decode needs max_query_len 1, this needs >= HLP_MIN).
Prefill attention runs eager in both graph modes, so the hook sees every prefill step.
Taken: every sequence of the step is fresh (seq_len == query_len, so all keys are in the `key`/`value` arguments and the cache is not read) and HLP_MIN <= query_len <= the largest bucket.
A step with any decode row, any cached prefix (prefix-cache hit, or the 2nd chunk of a prompt longer than max_num_batched_tokens) or any shorter prompt stays stock as a whole.
The kernel is compiled per length and takes fp16, the model runs bf16:
  length: the prompt is padded up to the next bucket. Causal masking keeps rows < n independent of the padding (rows >= n are computed and dropped), so the cost is time only: (L / n)^2.
  dtype: bf16 -> fp16 on the copy into the padded buffers (exact below 65504), fp16 -> bf16 on the copy out (rounds 10 mantissa bits to 7). HLP_GUARD=1 (default) checks the result is finite and
  reruns the step on vLLM's kernel if not, which is what an out-of-range input gives.
Kernel choice per length: gqa_fa.d256_tile (the landed dispatch; raced at Hkv 4, B 1, fp16 inputs on whole prompts. Hkv 2 under TP 2 and padded lengths are NOT raced: p1_check.py measures them).
Env: HLP_ENABLE=1; HLP_SCRIPTS dir of gqa_fa.py; HLP_MIN (1024); HLP_BUCKETS "1024,1088,..." or HLP_LADDER "lo:hi:step" (default 1024:16384:0.06, multiples of 32);
HLP_PRELOAD=1 load every bucket at the first taken step; HLP_LAZY=1 compile a missed bucket in a background process; HLP_VERIFY n: first n steps per bucket also run vLLM's kernel and log the difference;
HLP_GUARD (1); HLP_COUNTER_DIR: counters rewritten to <dir>/hlp-<pid>.json at most every 5 s."""
import atexit, json, os, subprocess, sys, time

D_EXPECT = 256
_fa = None
_kern, _buf, _pending = {}, {}, set()
_ct = {"hit": 0, "seqs": 0, "tokens": 0, "pad_tokens": 0, "stock_shape": 0, "stock_mixed": 0, "stock_context": 0, "stock_short": 0, "stock_long": 0, "stock_miss": 0,
       "stock_capture": 0, "stock_guard": 0, "error": 0}
_verify_left, _miss, _used = {}, {}, {}
_flush_t = 0.0
_preloaded = False
_last = (None, None)
_buckets = None


def _log(**k):
    print("HLP " + json.dumps({"pid": os.getpid(), "t": round(time.time(), 2), **k}), flush=True)


def ladder(lo, hi, step):
    """Lengths from lo to hi, each a multiple of 32, each at most (1 + step) times the one before; hi is always the last."""
    out, L = [], -(-lo // 32) * 32
    while L < hi:
        out.append(L)
        L = max(L + 32, int(L * (1 + step)) // 32 * 32)
    return out + [hi // 32 * 32]


def buckets():
    global _buckets
    if _buckets is None:
        b = os.environ.get("HLP_BUCKETS")
        if b:
            _buckets = sorted({int(x) for x in b.split(",") if x})
        else:
            lo, hi, st = os.environ.get("HLP_LADDER", "1024:16384:0.06").split(":")
            _buckets = ladder(int(lo), int(hi), float(st))
        if any(L % 2 for L in _buckets):
            raise ValueError("HLP buckets must be even (gqa_flash_fast)")
    return _buckets


def bucket(n):
    """Smallest bucket >= n, or None."""
    return next((L for L in buckets() if L >= n), None)


def _flush(force=False):
    global _flush_t
    d = os.environ.get("HLP_COUNTER_DIR")
    now = time.monotonic()
    if not d or (not force and now - _flush_t < 5.0):
        return
    _flush_t = now
    try:
        f = os.path.join(d, f"hlp-{os.getpid()}.json")
        with open(f + ".tmp", "w") as o:
            json.dump({"pid": os.getpid(), "t": round(time.time(), 2), **_ct, "miss_keys": _miss, "bucket_use": _used}, o)
        os.replace(f + ".tmp", f)
    except OSError:
        pass


def _mod():
    global _fa
    if _fa is None:
        import importlib.util
        p = os.path.join(os.environ["HLP_SCRIPTS"], "gqa_fa.py")
        s = importlib.util.spec_from_file_location("hlp_gqa_fa", p)
        m = importlib.util.module_from_spec(s)
        t = time.time(); s.loader.exec_module(m); _fa = m
        _log(ev="import", file=p, s=round(time.time() - t, 2))
    return _fa


def build(L, Hkv, G, D):
    """The landed d256 kernel at length L, B 1: (Q [1,L,Hkv*G,D], K [1,L,Hkv,D], V [1,L,Hkv,D]) fp16 -> O [1,L,Hkv*G,D] fp16. Compiles on a TileLang cache miss."""
    if D != D_EXPECT:
        raise ValueError(f"horizon_lumi_prefill covers D {D_EXPECT} only (the D 128 forms are fixed to Hkv 4), got {D}")
    m = _mod()
    return m.gqa_flash_fast(1, L, Hkv, G, D, True, **m.d256_tile(L, 1))


def _load(key):
    t = time.time()
    _kern[key] = build(*key)
    _verify_left[key] = int(os.environ.get("HLP_VERIFY", "0"))
    _log(ev="load", key=key, s=round(time.time() - t, 3))


def _lazy(key):
    if key in _pending:
        return
    _pending.add(key)
    if os.environ.get("HLP_LAZY") != "1":
        return
    mark = os.path.join(os.environ["TILELANG_CACHE_DIR"], "hlp-" + "-".join(map(str, key)))
    try:
        os.mkdir(mark)
    except FileExistsError:
        return
    code = "import horizon_lumi_prefill as h,sys,json;h.build(*json.loads(sys.argv[1]));open(sys.argv[2]+'/done','w').close()"
    subprocess.Popen([sys.executable, "-c", code, json.dumps(list(key)), mark], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    _log(ev="lazy_compile_start", key=key)


def _kernel(key):
    k = _kern.get(key)
    if k is not None:
        return k
    if key in _pending and os.path.exists(os.path.join(os.environ["TILELANG_CACHE_DIR"], "hlp-" + "-".join(map(str, key)), "done")):
        _load(key); _pending.discard(key)
        return _kern[key]
    _lazy(key)
    return None


def _bufs(L, H, Hkv, D, dev):
    b = _buf.get((L, H, Hkv, D))
    if b is None:
        import torch
        z = lambda h: torch.zeros((1, L, h, D), dtype=torch.float16, device=dev)
        b = _buf[(L, H, Hkv, D)] = (z(H), z(Hkv), z(Hkv))
    return b


def dense(kern, L, query, key, value, out):
    """One fresh sequence through the length-L kernel. query [n,H,D], key/value [n,Hkv,D], out [n,H,D], any float dtype; n <= L. Used by the hook and by p1_check.py, so the checked path is the served path.
    Rows >= n of K and V are zeroed every call: a masked key has weight exactly 0, but 0 * inf is NaN in the PV product, so a non-finite value left there by an earlier, longer call
    would reach rows < n. Rows >= n of Q keep old values; they only feed output rows that are dropped."""
    n, H, D = query.shape
    Q, K, V = _bufs(L, H, key.shape[1], D, query.device)
    Q[0, :n].copy_(query); K[0, :n].copy_(key); V[0, :n].copy_(value)
    if n < L:
        K[0, n:].zero_(); V[0, n:].zero_()
    O = kern(Q, K, V)
    out.copy_(O[0, :n])


def _plan(m):
    """[(start, end, L)] for a step this plugin takes, or the name of the counter that says why not. One host read of the step's metadata, cached for the 16 layers of the step."""
    global _last
    if _last[0] is m:
        return _last[1]
    qs = m.query_start_loc.tolist()
    ns = len(qs) - 1
    sl = m.seq_lens[:ns].tolist()
    lo = int(os.environ.get("HLP_MIN", "1024"))
    plan = []
    for i in range(ns):
        n = qs[i + 1] - qs[i]
        if n == 1 and sl[i] > 1:
            plan = "stock_mixed"; break
        if sl[i] != n:
            plan = "stock_context"; break
        if n < lo:
            plan = "stock_short"; break
        L = bucket(n)
        if L is None:
            plan = "stock_long"; break
        plan.append((qs[i], qs[i + 1], L))
    if plan == [] or (isinstance(plan, list) and plan[-1][1] != m.num_actual_tokens):
        plan = "stock_shape"
    _last = (m, plan)
    return plan


def _preload(Hkv, G, D):
    global _preloaded
    _preloaded = True
    if os.environ.get("HLP_PRELOAD") == "1":
        for L in buckets():
            _load((L, Hkv, G, D))


def _run(impl, query, key, value, m, output, stock):
    import torch
    plan = _plan(m)
    if isinstance(plan, str):
        _ct[plan] += 1
        return None
    Hkv, H, D = impl.num_kv_heads, query.shape[1], query.shape[2]
    if not _preloaded:
        _preload(Hkv, H // Hkv, D)
    ks = []
    for s, e, L in plan:
        kk = _kernel((L, Hkv, H // Hkv, D))
        if kk is None:
            _ct["stock_miss"] += 1
            _miss[str(L)] = _miss.get(str(L), 0) + 1
            return None
        ks.append(kk)
    nt = m.num_actual_tokens
    ver = [(L, Hkv, H // Hkv, D) for _, _, L in plan if _verify_left.get((L, Hkv, H // Hkv, D), 0) > 0]
    ref = None
    if ver:
        for k in set(ver):
            _verify_left[k] -= 1
        stock()                                              # vLLM's own kernel on the same inputs, into output
        ref = output[:nt].clone()
    for (s, e, L), kk in zip(plan, ks):
        dense(kk, L, query[s:e], key[s:e], value[s:e], output[s:e])
    if os.environ.get("HLP_GUARD", "1") == "1" and not bool(torch.isfinite(output[:nt]).all()):
        _ct["stock_guard"] += 1
        return None                                          # the caller reruns the step on vLLM's kernel, which overwrites output
    if ref is not None:
        a, r = output[:nt].float(), ref.float()
        _log(ev="verify", lens=[e - s for s, e, _ in plan], buckets=[L for _, _, L in plan], max_abs=float((a - r).abs().max()), rel=float((a - r).norm() / r.norm().clamp_min(1e-30)),
             rmax=float(r.abs().max()), nan=int(torch.isnan(a).sum()))
    _ct["seqs"] += len(plan); _ct["tokens"] += nt; _ct["pad_tokens"] += sum(L - (e - s) for s, e, L in plan)
    for _, _, L in plan:
        _used[str(L)] = _used.get(str(L), 0) + 1
    return output


def _no_window(sw):
    if sw is None:
        return True
    if isinstance(sw, (tuple, list)):
        return all(x is None or x <= 0 for x in sw)
    return sw <= 0


def _wrap(cls):
    orig = cls.forward

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None):
        import torch
        m = attn_metadata
        call = lambda: orig(self, layer, query, key, value, kv_cache, m, output, output_scale, output_block_scale)
        if m is None or m.max_query_len < int(os.environ.get("HLP_MIN", "1024")):   # decode and short steps: nothing of ours on the path but this compare
            return call()
        _flush()
        if output_scale is not None or output_block_scale is not None or m.use_cascade or not getattr(m, "causal", True) or key is None or value is None \
                or query.dtype != torch.bfloat16 or query.dim() != 3 or query.shape[2] != D_EXPECT or query.shape[1] % self.num_kv_heads \
                or getattr(self, "sinks", None) is not None or getattr(self, "alibi_slopes", None) is not None or not _no_window(getattr(self, "sliding_window", None)) \
                or str(getattr(self, "kv_cache_dtype", "auto")).startswith("fp8") or "ENCODER" in str(getattr(self, "attn_type", "")).upper():
            _ct["stock_shape"] += 1
            return call()
        if torch.cuda.is_current_stream_capturing():
            _ct["stock_capture"] += 1
            return call()
        try:
            r = _run(self, query, key, value, m, output, call)
        except Exception as e:
            _ct["error"] += 1
            if _ct["error"] <= 5:
                _log(ev="error", err=repr(e)[:400])
            return call()
        if r is None:
            return call()
        _ct["hit"] += 1
        return r

    forward.__wrapped__ = orig
    cls.forward = forward


_done = False


def register():
    global _done
    if _done or os.environ.get("HLP_ENABLE") != "1":
        return
    _done = True
    from vllm.v1.attention.backends import rocm_attn, triton_attn
    _wrap(rocm_attn.RocmAttentionImpl)
    _wrap(triton_attn.TritonAttentionImpl)
    atexit.register(lambda: (_flush(True), _log(ev="counters", **_ct)))
    _log(ev="registered", buckets=len(buckets()), lo=buckets()[0], hi=buckets()[-1])


def counters():
    return dict(_ct)
