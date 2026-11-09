#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""hz screen adapter: always run all GSM8K questions at 32 in flight."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("tag")
    p.add_argument("url")
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("horizon_gsm_screen", a.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = str(a.out.parent)
    module.URL = a.url.rstrip("/") + "/v1/chat/completions"
    module.run(a.tag, 32)
    written = a.out.parent / f"gsm-{a.tag}.json"
    data = json.loads(written.read_text())
    if len(data.get("correct", [])) != 1319:
        raise ValueError("incomplete GSM8K screen")
    written.replace(a.out)


if __name__ == "__main__":
    main()
