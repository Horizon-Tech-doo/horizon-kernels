# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GPU arithmetic screen only; no speed or integrated-serving claim."""
import json
from pathlib import Path
import sys
import tilelang.contrib

root,output=map(Path,sys.argv[1:3])
tilelang.contrib.__path__.insert(0,str(root/'contrib'))
import importlib.util
spec=importlib.util.spec_from_file_location('k4_lut_check',root/'check_silu_factor_table_05-10-2026.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
check_table=module.check_table

with output.open('x') as out:
    def emit(record):
        line=json.dumps(record)
        out.write(line+'\n');out.flush();print(line,flush=True)
    check_table(emit)
