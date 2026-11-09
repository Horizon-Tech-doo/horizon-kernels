# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""horizon_kvfp4: Horizon Tech's 4-bit (NVFP4) KV cache for vLLM 0.29 on SM120 (29-09-2026).

A separate vLLM general plugin (own dist-info entry point), so the production plugin package stays unchanged. With
HORIZON_KVFP4=1 it registers HorizonFp4Backend as vLLM's CUSTOM attention backend; select it with
`--attention-backend CUSTOM` and `--kv-cache-dtype nvfp4`. Without the switch it does nothing.
"""

def _hk_tilelang_contrib():
    """Public horizon-kernels tree: register the TileLang kernel packages in nvidia-sm120/tilelang_contrib/ on a stock
    TileLang, so that tilelang.contrib.<pkg> and tilelang.adaptive resolve (see tilelang_contrib/README.md). Does nothing
    when HK_NO_CONTRIB_BOOTSTRAP=1 or when no tilelang_contrib/ folder is found next to this package."""
    import os
    import sys
    if os.environ.get("HK_NO_CONTRIB_BOOTSTRAP") == "1":
        return
    try:
        import tilelang_contrib
    except ImportError:
        d = os.path.dirname(os.path.abspath(__file__))
        for _ in range(4):
            d = os.path.dirname(d)
            if os.path.isfile(os.path.join(d, "tilelang_contrib", "__init__.py")):
                sys.path.append(d)
                break
        else:
            return
        import tilelang_contrib
    tilelang_contrib.install()


_hk_tilelang_contrib()
del _hk_tilelang_contrib
import os


def register():
    if os.environ.get("HORIZON_KVFP4", "0") != "1":
        return
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.CUSTOM, "horizon_kvfp4.backend.HorizonFp4Backend")
