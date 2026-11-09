# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""python -m tilelang_contrib [--pth | --list]

--pth   print a .pth file that makes every Python start in this environment call tilelang_contrib.install()
--list  print which module names map to which folders (default)
"""
import os
import sys

from . import HERE, packages

if __name__ == "__main__":
    if "--pth" in sys.argv[1:]:
        print(os.path.dirname(HERE))
        print("import tilelang_contrib; tilelang_contrib.install()")
    else:
        for name, d in packages().items():
            print(f"{name}\t{d}")
