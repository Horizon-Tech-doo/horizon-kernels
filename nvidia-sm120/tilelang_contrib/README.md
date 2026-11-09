# tilelang_contrib

The TileLang kernels that `horizon_kernels`, `horizon_kvfp4` and the scripts in `scripts/kernels/` import as
`tilelang.contrib.<pkg>` and `tilelang.adaptive`. Packages here: `adaptive`, `horizon_fused`, `horizon_k4`, `horizon_moe`, `horizon_ops`, `sm120_fp8`, `sm120_nvfp4`, plus `adaptive` (`tilelang.adaptive`).

They are plain Python TileLang code. The CUDA they need is embedded as strings in the `.py` files. They run on an
unmodified TileLang install and need no TileLang fork or patch.

## Requirements

- TileLang v0.1.15 (recommended) or any upstream TileLang at or after commit `7a5f446f` (PR #3257, the SM120
  register-operand block-scaled GEMM used by `sm120_nvfp4.prefill_rs`, `prefill_rf`, `kernels` and `decode_fp4x3`).
- **Tested TileLang commit: none yet on stock TileLang.** The kernels ran and were measured on our own TileLang build
  (v0.1.15 plus these packages placed inside it). Importing and running this folder on an unmodified TileLang
  `>= 7a5f446f` has not been verified yet; the minimum version comes from reading the TileLang sources, not from a run.
- An NVIDIA SM120 GPU (RTX PRO 6000 Blackwell class) and a CUDA toolkit that TileLang supports for `sm_120`.

## Install

Importing `horizon_kernels` or `horizon_kvfp4` from this tree registers the hook automatically (their `__init__.py`
looks for this folder next to them and calls `install()`; `HK_NO_CONTRIB_BOOTSTRAP=1` turns that off). For anything
else, register it yourself as below. Nothing is copied into TileLang. Make the folder that contains `tilelang_contrib/` importable, then register the
import hook before the first `tilelang.contrib.<pkg>` import:

```python
import tilelang_contrib
tilelang_contrib.install()          # our copies win over same-named packages in the installed TileLang
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl
```

To have it in every Python process of an environment (for example a vLLM server that loads the `horizon_kernels`
plugin), write a `.pth` file once:

```bash
cd nvidia-sm120
python -m tilelang_contrib --pth > "$(python -c 'import site; print(site.getsitepackages()[0])')/horizon_tilelang_contrib.pth"
python -m tilelang_contrib --list   # shows which module names map to which folders
```

`tilelang_contrib.install(prefer_installed=True)` makes an installed `tilelang/contrib/<pkg>` win instead.
`install(third_party=True)` also puts the two upstream TileLang files in `third_party/tilelang/` on `sys.path`.
Two benchmark scripts import them as `sm120_nvfp4_blockscaled_gemm` and
`maint.gemm.gemm_sm120.tilelang_nvfp4_quantizer`.

Scripts that already put a `contrib/` folder on `tilelang.contrib.__path__` keep working if they point at this folder.

## Not included yet

The GDN decode path is held back until a GPU re-check has passed:

- `horizon_k4/segment.py`
- `horizon_ops/gdn_decode.py`
- `horizon_ops/gdn_g3v.py`
- `horizon_ops/gdn_recover.py`
- `horizon_ops/gdn_state.py`

Code that imports these modules (the GDN decode and GDN state paths in `horizon_kernels`, the scripts under
`scripts/kernels/gdnstate/` and the GDN parts of `scripts/kernels/k4p1/` and `scripts/kernels/ops/`) raises
`ImportError` until they are published.

## Licence

Apache-2.0, Copyright 2026 Horizon Tech d.o.o. `third_party/tilelang/` is TileLang's own code under its MIT licence.
See `NOTICE` here and in the repository root for third-party credits.
