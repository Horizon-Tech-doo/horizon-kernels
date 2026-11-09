// SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
// SPDX-License-Identifier: Apache-2.0
extern "C" __global__ void hz_probe_add(float* x, unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]+=2.0f;
}
