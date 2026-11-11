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
Version 2 (11 Nov 2026, HLP_CTX=1, default off): also takes a prefill chunk that has cached context, on RocmAttentionImpl, when the step holds one sequence. The square causal kernel runs at
Lk = context + chunk with the chunk's queries in rows context..Lk: causal masking makes those rows exactly the chunk's attention over cached prefix + chunk; the rows before are computed and dropped.
The prefix K and V are gathered from the paged cache through the step's block table (measured 0.1 to 0.2 ms), the chunk's K and V come from the call's arguments. Cost grows with Lk squared while vLLM's
kernel grows with chunk x Lk, so it is taken only up to HLP_CTX_MAX total tokens (default 9408: one-GCD job 23626788 against the per-chunk times of vLLM's kernel in the profile). Add the chunk
boundaries to the buckets (HLP_BUCKETS_784=1: every multiple of 784 up to the ladder's end) so that a chunk step is not padded.
Version 4 (11 Nov 2026, HLP_QW=1, default off; implies the weight scaling of HLP_PSHIFT): chunks with cached context run on gqa_fa_qw.py, the same kernel launched for the chunk's query
blocks only (cost chunk x total instead of total^2), at the exact total length. Taken: one sequence in the step, total length even and at most the largest bucket, and a kernel for (total, chunk)
precompiled: HLP_QW_CHUNKS (default "1568") x every total = k x chunk up to the ladder's end. Everything else as HLP_CTX. Fresh steps use the same module's square form.
Env: HLP_ENABLE=1; HLP_SCRIPTS dir of gqa_fa.py; HLP_MIN (1024); HLP_BUCKETS "1024,1088,..." or HLP_LADDER "lo:hi:step" (default 1024:16384:0.06, multiples of 32);
HLP_PRELOAD=1 load every bucket at the first taken step; HLP_LAZY=1 compile a missed bucket in a background process; HLP_VERIFY n: first n steps per bucket also run vLLM's kernel and log the difference;
HLP_GUARD (1); HLP_COUNTER_DIR: counters rewritten to <dir>/hlp-<pid>.json at most every 5 s."""
import atexit, json, os, subprocess, sys, time

D_EXPECT = 256
_fa = None
_kv = None
_kern, _buf, _pending = {}, {}, set()
_ct = {"hit": 0, "qw_hit": 0, "ctx_hit": 0, "ctx_tokens": 0, "stock_ctx_long": 0, "seqs": 0, "tokens": 0, "pad_tokens": 0, "stock_shape": 0, "stock_mixed": 0, "stock_context": 0, "stock_short": 0, "stock_long": 0, "stock_miss": 0,
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
        if os.environ.get("HLP_BUCKETS_784") == "1":
            _buckets = sorted(set(_buckets) | set(range(1568, _buckets[-1] + 1, 784)))
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
        t = time.time()
        if os.environ.get("HLP_QW") == "1":           # version 4: gqa_fa_qw.py (query window + weight scaling); its square form serves the fresh steps
            s = importlib.util.spec_from_file_location("hlp_gqa_fa_qw", os.path.join(os.environ["HLP_SCRIPTS"], "gqa_fa_qw.py"))
            w = importlib.util.module_from_spec(s); s.loader.exec_module(w); m = w.load(p)
        elif os.environ.get("HLP_PSHIFT") == "1":       # version 3: gqa_fa_ps.py, the same kernel with its weights scaled by 2^10 before the fp16 cast (the fp16 MFMA drops subnormal operands)
            s = importlib.util.spec_from_file_location("hlp_gqa_fa_ps", os.path.join(os.environ["HLP_SCRIPTS"], "gqa_fa_ps.py"))
            w = importlib.util.module_from_spec(s); s.loader.exec_module(w); m = w.load(p)
        else:
            s = importlib.util.spec_from_file_location("hlp_gqa_fa", p)
            m = importlib.util.module_from_spec(s)
            s.loader.exec_module(m)
        _fa = m
        _log(ev="import", file=p, pshift=os.environ.get("HLP_PSHIFT") == "1", s=round(time.time() - t, 2))
    return _fa


def build(L, Hkv, G, D):
    """The landed d256 kernel at length L, B 1: (Q [1,L,Hkv*G,D], K [1,L,Hkv,D], V [1,L,Hkv,D]) fp16 -> O [1,L,Hkv*G,D] fp16. Compiles on a TileLang cache miss."""
    if D != D_EXPECT:
        raise ValueError(f"horizon_lumi_prefill covers D {D_EXPECT} only (the D 128 forms are fixed to Hkv 4), got {D}")
    m = _mod()
    return m.gqa_flash_fast(1, L, Hkv, G, D, True, **m.d256_tile(L, 1))


def qw_keys():
    """[(total, chunk)] precompiled for HLP_QW: every chunk size of HLP_QW_CHUNKS at every total k x chunk, k >= 2, up to the largest bucket."""
    hi = buckets()[-1]
    return [(L, n) for n in (int(x) for x in os.environ.get("HLP_QW_CHUNKS", "1568").replace(":", ",").split(",") if x) for L in range(2 * n, hi + 1, n) if L % 2 == 0]


def build_qw(L, n, Hkv, G, D):
    """gqa_fa_qw's kernel: (Q [1,L,Hkv*G,D], K, V [1,L,Hkv,D]) fp16 -> O [1,L,Hkv*G,D] fp16 with rows L-n.. written (the last n query positions over all L keys)."""
    m = _mod()
    return m.gqa_flash_fast(1, L, Hkv, G, D, True, qw=n, **m.d256_tile(L, 1))


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


def dense_ctx(kern, L, query, key, value, key_cache, value_cache, blocks, ctx, out):
    """One chunk of n rows with ctx cached tokens before it through the square length-L kernel (L >= ctx + n). key_cache [blocks, Hkv, D/x, block_size, x] and value_cache [blocks, Hkv, D, block_size]
    as PagedAttention.split_kv_cache returns them; blocks: the sequence's block ids covering at least ctx tokens. Rows >= ctx + n of K and V are zeroed (see dense); Q rows before ctx keep old values:
    they only feed output rows that are dropped."""
    n, H, D = query.shape
    Hkv = key.shape[1]
    Q, K, V = _bufs(L, H, Hkv, D, query.device)
    kg = key_cache.index_select(0, blocks)
    vg = value_cache.index_select(0, blocks)
    nt = kg.shape[0] * vg.shape[3]
    K[0, :ctx].copy_(kg.permute(0, 3, 1, 2, 4).reshape(nt, Hkv, D)[:ctx])
    V[0, :ctx].copy_(vg.permute(0, 3, 1, 2).reshape(nt, Hkv, D)[:ctx])
    e = ctx + n
    K[0, ctx:e].copy_(key); V[0, ctx:e].copy_(value); Q[0, ctx:e].copy_(query)
    if e < L:
        K[0, e:].zero_(); V[0, e:].zero_()
    O = kern(Q, K, V)
    out.copy_(O[0, ctx:e])


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
            if os.environ.get("HLP_QW") == "1" and ns == 1 and n >= lo and sl[i] > n:
                if sl[i] % 2 or sl[i] > buckets()[-1]:
                    plan = "stock_ctx_long"; break
                plan = ("qw", n, sl[i], sl[i]); break
            if os.environ.get("HLP_CTX") == "1" and ns == 1 and n >= lo and sl[i] > n:
                if sl[i] > int(os.environ.get("HLP_CTX_MAX", "9408")) or bucket(sl[i]) is None:
                    plan = "stock_ctx_long"; break
                plan = ("ctx", n, sl[i], bucket(sl[i])); break
            plan = "stock_context"; break
        if n < lo:
            plan = "stock_short"; break
        L = bucket(n)
        if L is None:
            plan = "stock_long"; break
        plan.append((qs[i], qs[i + 1], L))
    if plan == [] or (isinstance(plan, list) and plan[-1][1] != m.num_actual_tokens) or (isinstance(plan, tuple) and plan[1] != m.num_actual_tokens):
        plan = "stock_shape"
    _last = (m, plan)
    return plan


def _preload(Hkv, G, D):
    global _preloaded
    _preloaded = True
    if os.environ.get("HLP_PRELOAD") == "1":
        for L in buckets():
            _load((L, Hkv, G, D))
        if os.environ.get("HLP_QW") == "1":
            for L, n in qw_keys():
                t = time.time()
                _kern[("qw", L, n, Hkv, G, D)] = build_qw(L, n, Hkv, G, D)
                _verify_left[("qw", L, n, Hkv, G, D)] = int(os.environ.get("HLP_VERIFY", "0"))
                _log(ev="load", key=("qw", L, n), s=round(time.time() - t, 3))


def _run(impl, query, key, value, m, output, stock):
    import torch
    plan = _plan(m)
    if isinstance(plan, str):
        _ct[plan] += 1
        return None
    Hkv, H, D = impl.num_kv_heads, query.shape[1], query.shape[2]
    if not _preloaded:
        _preload(Hkv, H // Hkv, D)
    if isinstance(plan, tuple):
        return _run_ctx(impl, query, key, value, m, output, stock, plan, Hkv, H, D)
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
        if len(plan) == 1:
            _dump(f"fresh{nt}", float((a - r).norm() / r.norm().clamp_min(1e-30)), q=query[:nt], k=key[:nt], v=value[:nt], ours=output[:nt], stock=ref, ctx=0)
    _ct["seqs"] += len(plan); _ct["tokens"] += nt; _ct["pad_tokens"] += sum(L - (e - s) for s, e, L in plan)
    for _, _, L in plan:
        _used[str(L)] = _used.get(str(L), 0) + 1
    return output


def _run_ctx(impl, query, key, value, m, output, stock, plan, Hkv, H, D):
    """A chunk with cached context (HLP_CTX=1). Returns output, or None for vLLM's kernel."""
    import torch
    _, n, sl, L = plan
    if type(impl).__name__ != "RocmAttentionImpl" or _kv is None:
        _ct["stock_context"] += 1
        return None
    qw = plan[0] == "qw"
    key4 = ("qw", L, n, Hkv, H // Hkv, D) if qw else (L, Hkv, H // Hkv, D)
    kk = _kern.get(key4) if qw else _kernel(key4)
    if kk is None:
        _ct["stock_miss"] += 1
        _miss[f"qw{L}:{n}" if qw else str(L)] = _miss.get(f"qw{L}:{n}" if qw else str(L), 0) + 1
        return None
    from vllm.v1.attention.backends import rocm_attn
    kc, vc = rocm_attn.PagedAttention.split_kv_cache(_kv.transpose(0, 1), Hkv, D)
    ctx = sl - n
    bs = vc.shape[3]
    blocks = m.block_table[0, :-(-ctx // bs)].long()
    ref = None
    if _verify_left.get(key4, 0) > 0:
        _verify_left[key4] -= 1
        stock()
        ref = output[:n].clone()
    dense_ctx(kk, L, query[:n], key[:n], value[:n], kc, vc, blocks, ctx, output[:n])
    if os.environ.get("HLP_GUARD", "1") == "1" and not bool(torch.isfinite(output[:n]).all()):
        _ct["stock_guard"] += 1
        return None
    if ref is not None:
        a, r = output[:n].float(), ref.float()
        _log(ev="verify", ctx=ctx, qw=qw, lens=[n], buckets=[L], max_abs=float((a - r).abs().max()), rel=float((a - r).norm() / r.norm().clamp_min(1e-30)), rmax=float(r.abs().max()), nan=int(torch.isnan(a).sum()))
        Qb, Kb, Vb = _bufs(L, H, Hkv, D, query.device)
        _dump(f"ctx{ctx}", float((a - r).norm() / r.norm().clamp_min(1e-30)), q=query[:n], k=Kb[0, :sl], v=Vb[0, :sl], ours=output[:n], stock=ref, ctx=ctx)
    _ct["ctx_hit"] += 1; _ct["ctx_tokens"] += n; _ct["pad_tokens"] += L - sl
    if qw:
        _ct["qw_hit"] += 1
    _used[f"ctx{L}"] = _used.get(f"ctx{L}", 0) + 1
    return output


_dumped = 0


def _dump(tag, rel, **t):
    """HLP_DUMP=<dir>: in a verify step whose difference from vLLM's kernel exceeds HLP_DUMP_REL (3e-3), save the step's tensors (at most HLP_DUMP_N files per process, default 3) so that both
    results can be compared with a float64 reference offline (p1_dump_ref.py)."""
    global _dumped
    d = os.environ.get("HLP_DUMP")
    if not d or _dumped >= int(os.environ.get("HLP_DUMP_N", "3")) or rel < float(os.environ.get("HLP_DUMP_REL", "3e-3")):
        return
    import torch
    _dumped += 1
    f = os.path.join(d, f"hlp-dump-{os.getpid()}-{_dumped}-{tag}.pt")
    torch.save({k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in t.items()}, f)
    _log(ev="dump", file=f, rel=rel)


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
        global _kv
        _kv = kv_cache
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
