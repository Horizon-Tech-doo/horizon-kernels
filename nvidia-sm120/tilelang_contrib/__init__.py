# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Make the horizon-kernels TileLang kernels importable on a stock TileLang install, without copying anything into it.

The kernels and the vLLM plugin import their TileLang code as ``tilelang.contrib.<pkg>`` (sm120_nvfp4, sm120_fp8,
horizon_fused, horizon_ops, horizon_k4, horizon_moe) and ``tilelang.adaptive``. This folder holds those packages.
``install()`` puts a small import hook in front of Python's normal finders that maps exactly those names to the
sub-folders here; everything else (``tilelang`` itself, ``tilelang.contrib.nvcc``, ...) still comes from the installed
TileLang. Importing this module does not import TileLang, so it is cheap enough for a ``.pth`` line.

Use one of:

    import tilelang_contrib; tilelang_contrib.install()     # before the first tilelang.contrib.<pkg> import

    python -m tilelang_contrib --pth > <site-packages>/horizon_tilelang_contrib.pth    # once per environment

The ``horizon_kernels`` and ``horizon_kvfp4`` packages of this tree call ``install()`` themselves when they are imported
(set HK_NO_CONTRIB_BOOTSTRAP=1 to skip that). Built for TileLang v0.1.15; a run on an unmodified TileLang has not been
verified yet (see README.md in this folder). Nothing here needs a TileLang fork.
"""
import importlib.abc
import importlib.util
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
# Extra names some experiment scripts use for a second copy of a package (same files, separate module objects).
ALIASES = {"horizon_ops_wip": "horizon_ops"}
# Unmodified upstream TileLang files (MIT) that two benchmark scripts import as top-level modules.
THIRD_PARTY_PATHS = (
    os.path.join(HERE, "third_party", "tilelang"),                           # maint.gemm.gemm_sm120.*
    os.path.join(HERE, "third_party", "tilelang", "examples", "gemm_sm120"),  # sm120_nvfp4_blockscaled_gemm
)


def packages():
    """{module name: folder} for every package in this folder."""
    found = {}
    for name in sorted(os.listdir(HERE)):
        d = os.path.join(HERE, name)
        if name.startswith(("_", ".")) or name == "third_party" or not os.path.isfile(os.path.join(d, "__init__.py")):
            continue
        found["tilelang.adaptive" if name == "adaptive" else "tilelang.contrib." + name] = d
    for alias, real in ALIASES.items():
        if "tilelang.contrib." + real in found:
            found.setdefault("tilelang.contrib." + alias, found["tilelang.contrib." + real])
    return found


class ContribFinder(importlib.abc.MetaPathFinder):
    """Resolves only the top-level names in ``mapping``; their submodules are then found through ``__path__``."""

    horizon_tilelang_contrib = True   # marker that survives a reload of this module (isinstance would not)

    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def find_spec(self, fullname, path=None, target=None):
        d = self.mapping.get(fullname)
        if d is None:
            return None
        return importlib.util.spec_from_file_location(fullname, os.path.join(d, "__init__.py"),
                                                      submodule_search_locations=[d])


def install(prefer_installed=False, third_party=False):
    """Register the import hook (idempotent) and return it.

    prefer_installed=False (default): these copies win over same-named packages inside the installed TileLang.
    prefer_installed=True: an installed ``tilelang/contrib/<pkg>`` wins; this folder is only the fallback.
    third_party=True: also put the unmodified upstream TileLang example/maint files on sys.path.
    """
    finder = next((f for f in sys.meta_path if getattr(f, "horizon_tilelang_contrib", False)), None)
    if finder is None:
        finder = ContribFinder(packages())
        if prefer_installed:
            sys.meta_path.append(finder)
        else:
            sys.meta_path.insert(0, finder)
        for name, d in finder.mapping.items():
            mod = sys.modules.get(name)
            origin = getattr(mod, "__file__", None) or ""
            if mod is not None and not os.path.abspath(origin).startswith(d + os.sep):
                warnings.warn(f"{name} was already imported from {origin}; tilelang_contrib.install() does not "
                              f"replace modules that are already loaded", stacklevel=2)
    if third_party:
        for p in THIRD_PARTY_PATHS:
            if p not in sys.path:
                sys.path.append(p)
    return finder
