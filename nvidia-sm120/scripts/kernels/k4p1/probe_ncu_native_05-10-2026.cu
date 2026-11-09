// SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
static void ck(cudaError_t rc) { if(rc!=cudaSuccess) { std::fprintf(stderr,"CUDA_ERROR %s\n",cudaGetErrorString(rc)); std::exit(70); } }
__global__ void hz_probe_add(const float* a,float* b,int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) b[i]=a[i]+2.0f; }
int main() {
 std::setvbuf(stdout,nullptr,_IONBF,0); std::puts("NATIVE_START");
 const int n=1<<20; float *a,*b; ck(cudaMalloc(&a,n*sizeof(float))); ck(cudaMalloc(&b,n*sizeof(float)));
 std::puts("NATIVE_ALLOCATED"); ck(cudaMemset(a,0,n*sizeof(float))); hz_probe_add<<<(n+255)/256,256>>>(a,b,n);
 ck(cudaGetLastError()); ck(cudaDeviceSynchronize()); float first,last;
 ck(cudaMemcpy(&first,b,sizeof(float),cudaMemcpyDeviceToHost)); ck(cudaMemcpy(&last,b+n-1,sizeof(float),cudaMemcpyDeviceToHost));
 if(first!=2.0f || last!=2.0f) return 71; ck(cudaFree(a));ck(cudaFree(b)); std::puts("NATIVE_CUDA_SMOKE_PASS"); return 0;
}
