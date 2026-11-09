# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Copy checkpoint conv history from pre-conv rows into reserved cache slots."""
from vllm.triton_utils import triton, tl


@triton.jit
def _export_conv(source, cache, slots, rows, N: tl.constexpr, D: tl.constexpr,
                 WIDTH: tl.constexpr, SS0: tl.constexpr, SS1: tl.constexpr,
                 CS0: tl.constexpr, CS1: tl.constexpr, CS2: tl.constexpr,
                 BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    active = index < N*D*WIDTH
    export = index // (D*WIDTH)
    dim = index // WIDTH % D
    history = index % WIDTH
    slot = tl.load(slots+export,active,0)
    row = tl.load(rows+export,active,0)-WIDTH+history
    value = tl.load(source+row*SS0+dim*SS1,active,0)
    tl.store(cache+slot*CS0+dim*CS1+history*CS2,value,active)


def export_conv(source,cache,slots,rows,width):
    if source.shape[1] != cache.shape[1] or width <= 0 or width > cache.shape[2]:
        raise ValueError('unsupported checkpoint convolution history layout')
    _export_conv[(triton.cdiv(slots.numel()*source.shape[1]*width,1024),)](
        source,cache,slots,rows,slots.numel(),source.shape[1],width,
        *source.stride(),*cache.stride(),1024)
