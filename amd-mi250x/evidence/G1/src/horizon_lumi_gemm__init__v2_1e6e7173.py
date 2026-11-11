"""LUMI research only (track PROFILE / G1). vLLM general plugin: runs tilelang/gemm_bf16.py (the bf16 MFMA form of gemm_prefetch's hand kernel) for the large dense linear layers of prefill steps on MI250X.
Research overlay only; it does nothing unless HLG_ENABLE=1. Anything it cannot run goes to vLLM's own path, so a miss costs speed, not correctness.
Hook: on ROCm every unquantized linear layer calls vllm.model_executor.layers.utils.rocm_unquantized_gemm, which calls the custom op vllm.rocm_unquantized_gemm (opaque to torch.compile, a Python call at run time).
register() replaces that function, before the model is built, by one that calls our custom op vllm.hlg_gemm with the same signature and the same fake implementation; its body takes the call or hands it to
vLLM's own implementation (rocm_unquantized_gemm_impl: wvSplitK for up to 5 rows, else torch F.linear = hipBLASLt).
Taken: x 2-D contiguous bf16 [M, K], weight bf16 [N, K] with (N, K) in HLG_SHAPES, no bias, not inside a graph capture, and M in HLG_MS (kernels are compiled per M) or at most HLG_MAXPAD below one of them
(then x is copied into an [Mb, K] buffer; the rows past M are computed and dropped, cost Mb / M).
Kernel: gemm_hand_rows on a 224x256 tile, XOR-swizzled LDS, partial waits (gemm_prefetch's HAND_FILL schedule for 16384x4096x4096). M need not be a tile multiple: rows past M read zeros through the buffer
range check. The server's prefill chunk at default settings is 1568 = 7 x 224 rows; with --max-num-batched-tokens 16384 the first step of a prompt is a multiple of 784 up to 15680.
No dtype change anywhere: activations, weights and result stay bf16 (the kernel is declared fp16 and gets the same 16-bit storage reinterpreted), accumulation is fp32 as in hipBLASLt.
HLG_FORMS (version 2, 11 Nov): per kernel, a tile and schedule other than the default, or "stock" to leave that (rows, shape) to vLLM: "NxK:M:bMxbN:sched;..." with sched one of xs4, plw, plw16, x8_42, x8_24
(gm_tiles.py's names), e.g. "17408x5120:1568:224x224:xs4;5120x8704:1568:stock". At the default chunk (1568 rows) the 224x256 tile leads at no shape; the tile sweep (job 23624491) found other tiles that do.
Env: HLG_ENABLE=1; HLG_SCRIPTS dir of gemm_bf16.py and gemm_prefetch.py; HLG_SHAPES "17408x5120,5120x8704,..."; HLG_MS "1568,..." ; HLG_MIN (1024) rows below which nothing of ours runs;
HLG_MAXPAD (0.0); HLG_VERIFY n: the first n taken calls per kernel also run vLLM's path and log the difference; HLG_COUNTER_DIR: counters rewritten to <dir>/hlg-<pid>.json at most every 5 s."""
import atexit, json, os, sys, time

_gb = None
_kern, _buf, _verify_left = {}, {}, {}
_ct = {"hit": 0, "rows": 0, "pad_rows": 0, "stock_small": 0, "stock_shape": 0, "stock_m": 0, "stock_capture": 0, "error": 0}
_miss, _used = {}, {}
_flush_t = 0.0
_shapes = _ms = None
_preloaded = False
TILE = (224, 256, 0)
SCHED = {"plw": 1, "xs": 1, "cn": 1}
SCHEDS = {"xs4": dict(pad=0, plw=1, xs=1, cn=1), "plw": dict(pad=32, plw=1, cn=1), "plw16": dict(pad=16, plw=1, cn=1), "x8_42": dict(pad=0, plw=1, nw=8, wg=(4, 2), xs=1, cn=1),
          "x8_24": dict(pad=0, plw=1, nw=8, wg=(2, 4), xs=1, cn=1)}
_forms = None


def forms():
    """{(M, N, K): None (stock) or (bM, bN, pad, schedule kwargs)} from HLG_FORMS; keys not listed use TILE and SCHED."""
    global _forms
    if _forms is None:
        _forms = {}
        for e in os.environ.get("HLG_FORMS", "").replace("\n", ";").split(";"):
            e = e.strip()
            if not e:
                continue
            nk, m, *rest = e.split(":")
            N, K = map(int, nk.split("x"))
            if rest[0] == "stock":
                _forms[(int(m), N, K)] = None
            else:
                bM, bN = map(int, rest[0].split("x"))
                kw = dict(SCHEDS[rest[1]])
                _forms[(int(m), N, K)] = (bM, bN, kw.pop("pad"), kw)
    return _forms


def _log(**k):
    print("HLG " + json.dumps({"pid": os.getpid(), "t": round(time.time(), 2), **k}), flush=True)


def shapes():
    global _shapes
    if _shapes is None:
        _shapes = {tuple(map(int, s.split("x"))) for s in os.environ.get("HLG_SHAPES", "17408x5120,5120x8704,8192x5120,7168x5120,5120x3072").split(",") if s}
    return _shapes


def ms():
    global _ms
    if _ms is None:
        _ms = sorted({int(x) for x in os.environ.get("HLG_MS", "1568").replace(":", ",").split(",") if x})
    return _ms


def bucket(n):
    """The compiled M that takes n rows: n itself, or the next one within HLG_MAXPAD; None otherwise."""
    pad = float(os.environ.get("HLG_MAXPAD", "0"))
    for M in ms():
        if M == n or (M > n and M - n <= pad * n):
            return M
    return None


def _flush(force=False):
    global _flush_t
    d = os.environ.get("HLG_COUNTER_DIR")
    now = time.monotonic()
    if not d or (not force and now - _flush_t < 5.0):
        return
    _flush_t = now
    try:
        f = os.path.join(d, f"hlg-{os.getpid()}.json")
        with open(f + ".tmp", "w") as o:
            json.dump({"pid": os.getpid(), "t": round(time.time(), 2), **_ct, "stock_m_rows": _miss, "kernel_use": _used}, o)
        os.replace(f + ".tmp", f)
    except OSError:
        pass


def _mod():
    global _gb
    if _gb is None:
        d = os.environ["HLG_SCRIPTS"]
        os.environ.setdefault("GEMM_PREFETCH", os.path.join(d, "gemm_prefetch.py"))
        import importlib.util
        s = importlib.util.spec_from_file_location("hlg_gemm_bf16", os.path.join(d, "gemm_bf16.py"))
        m = importlib.util.module_from_spec(s)
        t = time.time(); s.loader.exec_module(m); m.module(); _gb = m
        _log(ev="import", dir=d, s=round(time.time() - t, 2))
    return _gb


def build(M, N, K):
    """(a bf16 [M, K] contiguous, w bf16 [N, K]) -> bf16 [M, N]. Compiles on a TileLang cache miss. N must be a multiple of 256."""
    import torch
    f = forms().get((M, N, K), (*TILE, SCHED))
    if f is None:
        return None                                           # left to vLLM by HLG_FORMS
    bM, bN, pad, kw = f
    k = _mod().module().gemm_hand_rows(M, N, K, bM, bN, pad, **kw)   # cn 1: C has the true N columns, so N need not be a tile multiple
    f16, bf = torch.float16, torch.bfloat16
    if M % bM == 0:
        return lambda a, w: k(a.view(f16), w.view(f16)).view(bf)
    return lambda a, w: k(a.view(f16), w.view(f16))[:M].view(bf)


def _load(key):
    t = time.time()
    _kern[key] = build(*key)
    _verify_left[key] = int(os.environ.get("HLG_VERIFY", "0"))
    _log(ev="load", key=key, s=round(time.time() - t, 3))


def _preload():
    global _preloaded
    _preloaded = True
    for M in ms():
        for N, K in sorted(shapes()):
            try:
                _load((M, N, K))
            except Exception as e:
                _log(ev="load_error", key=(M, N, K), err=repr(e)[:300])


def run(x, weight, stock):
    """The taken path, also used by the standalone check so that the checked path is the served path. Returns None when the call is not ours (the counter says why)."""
    import torch
    n = x.shape[0]
    N, K = weight.shape
    if (N, K) not in shapes():
        _ct["stock_shape"] += 1
        return None
    Mb = bucket(n)
    if Mb is None:
        _ct["stock_m"] += 1
        if len(_miss) < 256 or str(n) in _miss:
            _miss[str(n)] = _miss.get(str(n), 0) + 1
        return None
    if torch.cuda.is_current_stream_capturing():
        _ct["stock_capture"] += 1
        return None
    if not _preloaded:
        _preload()
    key = (Mb, N, K)
    kern = _kern.get(key)
    if kern is None:
        _ct["stock_m"] += 1
        return None
    if Mb != n:
        b = _buf.get((Mb, K))
        if b is None:
            b = _buf[(Mb, K)] = torch.zeros((Mb, K), dtype=torch.bfloat16, device=x.device)
        b[:n].copy_(x)
        out = kern(b, weight)[:n]
        _ct["pad_rows"] += Mb - n
    else:
        out = kern(x if x.is_contiguous() else x.contiguous(), weight)
    if _verify_left.get(key, 0) > 0:
        _verify_left[key] -= 1
        ref = stock(x, weight, None).float()
        a = out.float()
        _log(ev="verify", key=key, rows=n, max_abs=float((a - ref).abs().max()), rel=float((a - ref).norm() / ref.norm().clamp_min(1e-30)), rmax=float(ref.abs().max()), nonfinite=int((~torch.isfinite(a)).sum()),
             ref_nonfinite=int((~torch.isfinite(ref)).sum()))
    _ct["hit"] += 1; _ct["rows"] += n
    _used[f"{Mb}x{N}x{K}"] = _used.get(f"{Mb}x{N}x{K}", 0) + 1
    return out


_done = False


def register():
    global _done
    if _done or os.environ.get("HLG_ENABLE") != "1":
        return
    _done = True
    import torch
    from vllm.model_executor.layers import utils as U
    from vllm.utils.torch_utils import direct_register_custom_op
    stock = U.rocm_unquantized_gemm_impl
    lo = int(os.environ.get("HLG_MIN", "1024"))
    bf = torch.bfloat16

    def hlg_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 2 or x.shape[0] < lo:                 # decode and short steps: nothing of ours on the path but this compare
            return stock(x, weight, bias)
        _flush()
        if bias is not None or x.dtype != bf or weight.dtype != bf or weight.dim() != 2 or not weight.is_contiguous():
            _ct["stock_shape"] += 1
            return stock(x, weight, bias)
        try:
            r = run(x, weight, stock)
        except Exception as e:
            _ct["error"] += 1
            if _ct["error"] <= 5:
                _log(ev="error", err=repr(e)[:400])
            r = None
        return stock(x, weight, bias) if r is None else r

    direct_register_custom_op(op_name="hlg_gemm", op_func=hlg_gemm, fake_impl=U.rocm_unquantized_gemm_fake)

    def rocm_unquantized_gemm(layer, x, weight, bias=None):
        return torch.ops.vllm.hlg_gemm(x, weight, bias)
    U.rocm_unquantized_gemm = rocm_unquantized_gemm
    atexit.register(lambda: (_flush(True), _log(ev="counters", **_ct)))
    _log(ev="registered", shapes=sorted(shapes()), ms=ms(), min=lo, forms={"x".join(map(str, k)): (v if v is None else [v[0], v[1], v[2]]) for k, v in forms().items()})


def counters():
    return dict(_ct)
