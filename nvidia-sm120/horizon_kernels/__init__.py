# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Horizon Tech custom TileLang kernels (horizon_kernels) for the RTX PRO 6000 (SM120). See docs/."""

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
