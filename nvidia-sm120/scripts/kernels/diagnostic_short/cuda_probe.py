# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Tiny driver-API kernel, independent of PyTorch/CUDA runtime. Keeps provider interposition intact."""
import ctypes as C
import json

cuda = C.CDLL('libcuda.so.1')
def check(name, *args):
    rc = getattr(cuda, name)(*args)
    if rc:
        err = C.c_char_p()
        cuda.cuGetErrorString(rc, C.byref(err))
        raise RuntimeError(f'{name}: {rc}: {err.value!r}')
check('cuInit', 0)
dev = C.c_int()
check('cuDeviceGet', C.byref(dev), 0)
ctx = C.c_void_p()
check('cuCtxCreate_v2', C.byref(ctx), 0, dev)
try:
    ptx = b'''\n.version 8.0
.target sm_80
.address_size 64
.visible .entry diagnostic_write(.param .u64 output) {
.reg .u64 %rd;
.reg .u32 %r;
ld.param.u64 %rd, [output];
mov.u32 %r, 42;
st.global.u32 [%rd], %r;
ret;
}\n'''
    module, fn = C.c_void_p(), C.c_void_p()
    check('cuModuleLoadData', C.byref(module), C.c_char_p(ptx))
    check('cuModuleGetFunction', C.byref(fn), module, b'diagnostic_write')
    ptr = C.c_uint64()
    check('cuMemAlloc_v2', C.byref(ptr), C.c_size_t(4))
    args = (C.c_void_p * 1)(C.cast(C.byref(ptr), C.c_void_p))
    check('cuLaunchKernel', fn, 1, 1, 1, 1, 1, 1, 0, C.c_void_p(), args, C.c_void_p())
    check('cuCtxSynchronize')
    result = C.c_uint32()
    check('cuMemcpyDtoH_v2', C.byref(result), ptr, C.c_size_t(4))
    check('cuMemFree_v2', ptr)
    assert result.value == 42, result.value
    print(json.dumps({'probe': 'driver_ptx', 'value': result.value, 'ok': True}), flush=True)
finally:
    check('cuCtxDestroy_v2', ctx)
