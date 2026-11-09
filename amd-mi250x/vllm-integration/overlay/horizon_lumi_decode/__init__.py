"""LUMI research only (track V1). vLLM general plugin: runs tilelang/gqa_decode.py for decode-only attention steps of a hybrid model on MI250X.
Research overlay only; it does nothing unless HLD_ENABLE=1. Anything it cannot run goes to vLLM's own kernel, so a miss costs speed, not correctness.
Hook: RocmAttentionImpl.forward (layout rocm_hybrid) and TritonAttentionImpl.forward (layout lbnhc). Attention runs eager under PIECEWISE graphs (vLLM's own comment in both forwards), so there the hook sees every step. vLLM's default FULL_AND_PIECEWISE replays a FULL graph on decode-only steps and the hook is not called at all: HLD_CAPTURE=1 records our kernels into that graph at capture (max_seq_len = max_model_len there, so the 65536 bucket; verify cannot run in a graph). Without it those steps stay stock.
B and MAXP are compile-time, so (B, L) buckets: B pads with rows of length 0, L rounds up. NBLK only sets the outermost shape, which the kernel never uses for addressing (bounds checks are off), so kernels are compiled at a fixed NBLK and fed a view of that shape.
Env: HLD_ENABLE=1; HLD_SCRIPTS dir of gqa_decode.py; HLD_PRELOAD "B:L,..." loaded at the first decode; HLD_VERIFY n: first n calls per kernel also run vLLM's kernel and log the difference; HLD_SKIP_S n: at L 65536 use the uniform dead="skip" form with n splits; HLD_LAZY=1 compile missed buckets in a background process; HLD_VERIFY_PAD n: first n padded calls (rows < B) per kernel also verified; HLD_COUNTER_DIR: counters rewritten to <dir>/<pid>.json at most every 5 s, so they survive SIGKILL (atexit alone does not)."""
import atexit, json, os, subprocess, sys, time

BUCK_B = (1, 2, 4, 8, 16, 32, 64)
BUCK_L = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
FAKE_NBLK = 512
PAGE, D_EXPECT = 16, 256
_gq = None
_kern, _buf, _view, _pending = {}, {}, {}, set()
_ct = {"hit": 0, "stock_shape": 0, "stock_miss": 0, "stock_capture": 0, "capture_hit": 0, "error": 0, "pad_rows": 0}
_verify_left = {}
_verify_pad_left = {}
_miss, _by_d, _lay_seen, _lay_hit = {}, {}, set(), set()
_flush_t = 0.0
_preloaded = False


def _log(**k):
    print("HLD " + json.dumps({"pid": os.getpid(), "t": round(time.time(), 2), **k}), flush=True)


def _flush(force=False):
    """Atomic rewrite of this process's counters. Called from the hook; at most every 5 s."""
    global _flush_t
    d = os.environ.get("HLD_COUNTER_DIR")
    now = time.monotonic()
    if not d or (not force and now - _flush_t < 5.0):
        return
    _flush_t = now
    try:
        f = os.path.join(d, f"{os.getpid()}.json")
        with open(f + ".tmp", "w") as o:
            json.dump({"pid": os.getpid(), "t": round(time.time(), 2), **_ct, "layers_seen": len(_lay_seen), "layers_hit": len(_lay_hit), "calls_by_D": _by_d, "miss_keys": _miss}, o)
        os.replace(f + ".tmp", f)
    except OSError:
        pass


def _mod():
    global _gq
    if _gq is None:
        import importlib.util
        p = os.path.join(os.environ["HLD_SCRIPTS"], "gqa_decode.py")
        s = importlib.util.spec_from_file_location("hld_gqa_decode", p)
        m = importlib.util.module_from_spec(s)
        t = time.time(); s.loader.exec_module(m); _gq = m
        _log(ev="import", file=p, s=round(time.time() - t, 2))
    return _gq


def build(layout, B, L, Hkv, G, D, Nb):
    """(k1, k2) for the bucket, through the landed picker for shape; compiles on a TileLang cache miss."""
    S = int(os.environ.get("HLD_SKIP_S", "0"))
    if S and L == 65536:                                    # the FULL-graph bucket: most splits hold no page of a short row, so skip them (N3 round 5)
        return _mod().paged_uniform(B, FAKE_NBLK, L // PAGE, Hkv, G, D, S, PAGE, layout, None, None, Nb, "bf16", "bits", "skip")
    return _mod().paged_kernels(B, FAKE_NBLK, L // PAGE, Hkv, G, D, PAGE, layout, None, None, Nb, "bf16", "bits")


def _load(key):
    t = time.time()
    _kern[key] = build(*key)
    _verify_left[key] = int(os.environ.get("HLD_VERIFY", "0"))
    _verify_pad_left[key] = int(os.environ.get("HLD_VERIFY_PAD", "0"))
    _log(ev="load", key=key, s=round(time.time() - t, 3))


def _lazy(key):
    """Compile in a separate process so the serving step never waits on hipcc. The first rank to take the marker compiles."""
    if key in _pending:
        return
    _pending.add(key)
    if os.environ.get("HLD_LAZY") != "1":
        return
    d = os.environ["TILELANG_CACHE_DIR"]
    mark = os.path.join(d, "hld-" + "-".join(map(str, key)))
    try:
        os.mkdir(mark)
    except FileExistsError:
        return
    code = "import horizon_lumi_decode as h,sys,json,os;h.build(*json.loads(sys.argv[1]));open(sys.argv[2]+'/done','w').close()"
    subprocess.Popen([sys.executable, "-c", code, json.dumps(list(key)), mark], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    _log(ev="lazy_compile_start", key=key)


def _kernel(key):
    k = _kern.get(key)
    if k is not None:
        return k
    if key in _pending and os.path.exists(os.path.join(os.environ["TILELANG_CACHE_DIR"], "hld-" + "-".join(map(str, key)), "done")):
        _load(key); _pending.discard(key)
        return _kern[key]
    _lazy(key)
    return None


def _preload(layout, Hkv, G, D, Nb):
    global _preloaded
    _preloaded = True
    for it in filter(None, os.environ.get("HLD_PRELOAD", "").split(",")):
        B, L = map(int, it.split(":"))
        _load((layout, B, L, Hkv, G, D, Nb))


def _views(layout, kv, Hkv, D, Nb):
    """Shape the layer's cache into the compiled form without moving it. None if the live strides are not the contiguous form this plugin was written for."""
    key = (kv.data_ptr(), kv.shape)
    v = _view.get(key)
    if v is not None:
        return v
    import torch
    x = 2 * D if layout == "lbnhc" else 0
    if layout == "lbnhc":                                   # logical [NBLK, Hkv, Nb, 2D], physical [NBLK, Nb, Hkv, 2D]
        if tuple(kv.stride()) != (Nb * Hkv * x, x, Hkv * x, 1) or kv.shape[0] < FAKE_NBLK:
            return None
        k = torch.as_strided(kv, (FAKE_NBLK, Nb, Hkv, x), (Nb * Hkv * x, Hkv * x, x, 1), kv.storage_offset())
        v = (k, k)
    else:                                                   # logical [NBLK, 2, Nb, Hkv*D]
        hd = Hkv * D
        if tuple(kv.stride()) != (2 * Nb * hd, Nb * hd, hd, 1) or kv.shape[0] < FAKE_NBLK:
            return None
        o = kv.storage_offset()
        v = (torch.as_strided(kv, (FAKE_NBLK, 2, Hkv, D // 8, Nb, 8), (2 * Nb * hd, Nb * hd, Nb * D, 8 * Nb, 8, 1), o),
             torch.as_strided(kv, (FAKE_NBLK, 2, Hkv, D, Nb), (2 * Nb * hd, Nb * hd, Nb * D, Nb, 1), o))
    _view[key] = v
    return v


def _run(impl, layout, layer, query, kv, m, output, stock, capturing=False):
    import torch
    n = m.num_actual_tokens
    Hkv, H, D = impl.num_kv_heads, query.shape[1], query.shape[2]
    Nb = kv.shape[2]
    if (not _preloaded):
        _preload(layout, Hkv, H // Hkv, D, Nb)
    B = next((b for b in BUCK_B if b >= n), None)
    L = next((l for l in BUCK_L if l >= m.max_seq_len), None)
    if B is None or L is None:
        _ct["stock_shape"] += 1
        return None
    key = (layout, B, L, Hkv, H // Hkv, D, Nb)
    kk = _kern.get(key) if capturing else _kernel(key)     # a capture never starts a compile; no kernel means stock in the graph
    if kk is None:
        _ct["stock_miss"] += 1
        mk = f"{B}:{L}"
        _miss[mk] = _miss.get(mk, 0) + 1
        return None
    v = _views(layout, kv, Hkv, D, Nb)
    if v is None:
        _ct["stock_shape"] += 1
        return None
    bw = -(-(L // PAGE) // (Nb // PAGE))
    bt = m.block_table
    if B == n and bt.shape == (B, bw) and bt.dtype == torch.int32 and bt.stride(0) == bw:   # the kernel addresses BT[i, j] as i * bw + j
        BT = bt
    else:
        b = _buf.get(("bt", B, bw))
        if b is None:
            b = _buf[("bt", B, bw)] = torch.zeros((B, bw), dtype=torch.int32, device=query.device)
        w = min(bw, bt.shape[1])
        b[:n, :w] = bt[:n, :w].to(torch.int32)
        BT = b
    sl = m.seq_lens
    if B == n and sl.dtype == torch.int32 and sl.shape[0] == B:
        Lens, Q = sl, query[:n]
    else:
        ln = _buf.get(("len", B))
        if ln is None:
            ln = _buf[("len", B)] = torch.zeros((B,), dtype=torch.int32, device=query.device)
            _buf[("q", B, H, D)] = torch.zeros((B, H, D), dtype=query.dtype, device=query.device)
        ln.zero_(); ln[:n] = sl[:n].to(torch.int32)
        Q = _buf[("q", B, H, D)]; Q[:n] = query[:n]
        Lens = ln
        _ct["pad_rows"] += B - n
    k1, k2 = kk
    O = k2(*k1(Q, v[0], v[1], BT, Lens), Lens) if L == 65536 and os.environ.get("HLD_SKIP_S", "0") != "0" else k2(*k1(Q, v[0], v[1], BT, Lens))
    if capturing:                                            # replay pads rows with seq_len 0, where this kernel gives NaN and vLLM's gives 0
        output[:n] = torch.nan_to_num(O[:n], nan=0.0)
        return output
    left = _verify_left.get(key, 0)
    padleft = _verify_pad_left.get(key, 0) if B != n else 0
    if left > 0 or padleft > 0:
        if padleft > 0:
            _verify_pad_left[key] = padleft - 1
        else:
            _verify_left[key] = left - 1
        stock()                                              # vLLM's own kernel on the same inputs, into output
        ref = output[:n].clone()
        a, r = O[:n].float(), ref.float()
        _log(ev="verify", key=key, n=int(n), padded=bool(B != n), max_seq_len=int(m.max_seq_len), max_abs=float((a - r).abs().max()), rel=float((a - r).norm() / r.norm().clamp_min(1e-30)), rmax=float(r.abs().max()), nan=int(torch.isnan(a).sum()))
    output[:n] = O[:n]
    return output


def _wrap(cls, layout):
    orig = cls.forward

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None):
        import torch
        m = attn_metadata
        _flush()
        d = str(query.shape[2]) if query.dim() == 3 else "?"
        _by_d[d] = _by_d.get(d, 0) + 1
        _lay_seen.add(id(layer))
        call = lambda: orig(self, layer, query, key, value, kv_cache, m, output, output_scale, output_block_scale)
        if m is None or output_scale is not None or output_block_scale is not None or m.max_query_len != 1 or m.use_cascade or not getattr(m, "causal", True) \
                or query.dtype != torch.bfloat16 or kv_cache.dim() != 4 or getattr(self, "sinks", None) is not None or getattr(self, "alibi_slopes", None) is not None \
                or m.seq_lens.shape[0] < m.num_actual_tokens or query.shape[2] != D_EXPECT or query.shape[1] % self.num_kv_heads:
            _ct["stock_shape"] += 1
            return call()
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and os.environ.get("HLD_CAPTURE") != "1":
            _ct["stock_capture"] += 1
            return call()
        try:
            r = _run(self, layout, layer, query, kv_cache, m, output, call, capturing)
        except Exception as e:
            _ct["error"] += 1
            if _ct["error"] <= 5:
                _log(ev="error", err=repr(e)[:400])
            return call()
        if r is None:
            return call()
        _ct["capture_hit" if capturing else "hit"] += 1
        _lay_hit.add(id(layer))
        return r

    forward.__wrapped__ = orig
    cls.forward = forward


_done = False


def register():
    global _done
    if _done or os.environ.get("HLD_ENABLE") != "1":
        return
    _done = True
    from vllm.v1.attention.backends import rocm_attn, triton_attn
    _wrap(rocm_attn.RocmAttentionImpl, "rocm_hybrid")
    _wrap(triton_attn.TritonAttentionImpl, "lbnhc")
    atexit.register(lambda: (_flush(True), _log(ev="counters", **_ct)))
    _log(ev="registered")


def counters():
    return dict(_ct)
