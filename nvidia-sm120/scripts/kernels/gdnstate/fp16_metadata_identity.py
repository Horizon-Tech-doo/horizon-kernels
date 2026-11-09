#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Require packed metadata to preserve all previously measured fp16 probabilities."""
import argparse
import json
from pathlib import Path


def check(candidate, reference):
    documents = {f'{kind}_{length}' for kind in ('prose', 'code') for length in (1024, 4096, 16384, 32768)}
    complete = set(candidate) == set(reference) == documents
    tokens = sum(len(value.get('lp', [])) for value in reference.values())
    complete = complete and tokens == 108536 and all(
        set(value) == {'lp', 'top1'} and len(value['lp']) == len(value['top1']) > 0
        for value in reference.values())
    return {'gates': {'unchanged_fp16_probabilities': {
        'result': 'pass' if complete and candidate == reference else 'fail',
        'observed': {'complete_reference': complete, 'tokens': tokens, 'exact': candidate == reference}}}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'reference', 'output'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(check(json.loads(a.candidate.read_text()),
                                         json.loads(a.reference.read_text())), indent=2)+'\n')
