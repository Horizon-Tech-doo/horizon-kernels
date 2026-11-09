// SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
// SPDX-License-Identifier: Apache-2.0
#include <cuda.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void mark(const char* s) { std::puts(s); std::fflush(stdout); }
static void check(CUresult r, const char* s) {
    if(r!=CUDA_SUCCESS) { std::fprintf(stderr,"DRIVER_ERROR %s code=%d\n",s,int(r)); std::exit(2); }
}
template<class T> T sym(void* h,const char* s) {
    auto p=dlsym(h,s); if(!p) { std::fprintf(stderr,"MISSING_SYMBOL %s\n",s);std::exit(3); }
    return reinterpret_cast<T>(p);
}
int main(int argc,char** argv) {
    if(argc!=2) return 4;
    mark("DRIVER_START");
    void* h=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
    if(!h) { std::fprintf(stderr,"DLOPEN_FAILED %s\n",dlerror()); return 5; }
    auto init=sym<CUresult(*)(unsigned)>(h,"cuInit");
    Dl_info info{}; dladdr(reinterpret_cast<void*>(init),&info);
    std::printf("DRIVER_LIBRARY %s\n",info.dli_fname);std::fflush(stdout);
    check(init(0),"cuInit");mark("DRIVER_INITIALIZED");
    int v=0;check(sym<CUresult(*)(int*)>(h,"cuDriverGetVersion")(&v),"driverVersion");
    std::printf("DRIVER_VERSION %d\n",v);std::fflush(stdout);
    CUdevice dev;check(sym<CUresult(*)(CUdevice*,int)>(h,"cuDeviceGet")(&dev,0),"deviceGet");
    CUcontext ctx;check(sym<CUresult(*)(CUcontext*,unsigned,CUdevice)>(h,"cuCtxCreate_v2")(&ctx,0,dev),"contextCreate");mark("DRIVER_CONTEXT");
    CUmodule mod;check(sym<CUresult(*)(CUmodule*,const char*)>(h,"cuModuleLoad")(&mod,argv[1]),"moduleLoad");
    CUfunction fn;check(sym<CUresult(*)(CUfunction*,CUmodule,const char*)>(h,"cuModuleGetFunction")(&fn,mod,"hz_probe_add"),"getFunction");
    unsigned n=1u<<20;CUdeviceptr x;
    check(sym<CUresult(*)(CUdeviceptr*,size_t)>(h,"cuMemAlloc_v2")(&x,size_t(n)*sizeof(float)),"alloc");mark("DRIVER_ALLOCATED");
    check(sym<CUresult(*)(CUdeviceptr,unsigned,size_t)>(h,"cuMemsetD32_v2")(x,0,n),"memset");
    void* args[]={&x,&n};
    check(sym<CUresult(*)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**)>(h,"cuLaunchKernel")(fn,(n+255)/256,1,1,256,1,1,0,nullptr,args,nullptr),"launch");
    check(sym<CUresult(*)()>(h,"cuCtxSynchronize")(),"synchronize");mark("DRIVER_KERNEL_COMPLETE");
    std::vector<float> out(n);
    check(sym<CUresult(*)(void*,CUdeviceptr,size_t)>(h,"cuMemcpyDtoH_v2")(out.data(),x,size_t(n)*sizeof(float)),"copy");
    for(float f:out) if(f!=2.0f) return 6;
    check(sym<CUresult(*)(CUdeviceptr)>(h,"cuMemFree_v2")(x),"free");
    check(sym<CUresult(*)(CUmodule)>(h,"cuModuleUnload")(mod),"unload");
    check(sym<CUresult(*)(CUcontext)>(h,"cuCtxDestroy_v2")(ctx),"destroy");
    mark("DRIVER_PASS");return 0;
}
