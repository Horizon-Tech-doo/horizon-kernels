// E1: hipBLASLt called directly from C++ (no torch), interleaved with solo TileLang kernels in one process and on the same data.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <map>
#include <fstream>
#include <cmath>
#include <cstring>
static hipStream_t st;
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("HIPERR %s %s\n",#x,hipGetErrorString(e_));fflush(stdout);exit(1);} }while(0)
#define CB(x) do{hipblasStatus_t e_=(x); if(e_!=HIPBLAS_STATUS_SUCCESS){printf("HBERR %s %d\n",#x,(int)e_);fflush(stdout);exit(1);} }while(0)
static uint64_t r_=88172645463325252ull; static inline uint64_t rn(){r_^=r_<<13;r_^=r_>>7;r_^=r_<<17;return r_;}
static uint16_t* mk16(size_t n){ uint16_t* p; CK(hipMalloc(&p,n*2)); std::vector<uint16_t> h(n);
  for(size_t i=0;i<n;i+=4){uint64_t v=rn(); for(int j=0;j<4&&i+j<n;j++){uint16_t m=(v>>(16*j))&0x83ff; h[i+j]=m|(14<<10);} }
  CK(hipMemcpy(p,h.data(),n*2,hipMemcpyHostToDevice)); return p;}
static std::map<std::string,hipModule_t> mods;
static hipModule_t mod(const std::string&fn){ auto it=mods.find(fn); if(it!=mods.end()) return it->second; std::ifstream in(fn,std::ios::binary); if(!in){printf("NOFILE %s\n",fn.c_str());exit(1);} std::vector<char> co((std::istreambuf_iterator<char>(in)),{}); hipModule_t m; CK(hipModuleLoadData(&m,co.data())); mods[fn]=m; return m; }
struct SH{const char*sn;int M,N,K,it;};
static double half2d(uint16_t v){ int s=v>>15,e=(v>>10)&31,m=v&1023; double x=(e==0)?m/1024.0*pow(2,-14):(1+m/1024.0)*pow(2,e-15); return s?-x:x; }
int main(int argc,char**argv){ int rounds=atoi(argv[1]); CK(hipStreamCreate(&st));
  hipblasLtHandle_t hl; CB(hipblasLtCreate(&hl));
  SH shs[]={{"4k",4096,4096,4096,40},{"8k",8192,8192,8192,20},{"32k",32768,1024,2048,40},{"17k",4096,17408,5120,20}};
  struct V{const char*n;int bM,bK;size_t dyn;}; V vs[]={{"b12864",128,64,32768},{"b25632w2",256,32,24576},{"rr12864w3",128,64,32768},{"rr25632w2",256,32,24576}};
  float a1=1.f,b0=0.f;
  for(auto&s:shs){ uint16_t*A=mk16(size_t(s.M)*s.K),*B=mk16(size_t(s.N)*s.K),*C=mk16(size_t(s.M)*s.N),*C2=mk16(size_t(s.M)*s.N);
    // same problem as torch's a @ b.T: column-major TN, first=B (K x N), second=A (K x M), result N x M
    hipblasLtMatrixLayout_t la,lb,lc; hipblasLtMatmulDesc_t d;
    CB(hipblasLtMatrixLayoutCreate(&la,HIP_R_16F,s.K,s.N,s.K)); CB(hipblasLtMatrixLayoutCreate(&lb,HIP_R_16F,s.K,s.M,s.K)); CB(hipblasLtMatrixLayoutCreate(&lc,HIP_R_16F,s.N,s.M,s.N));
    CB(hipblasLtMatmulDescCreate(&d,HIPBLAS_COMPUTE_32F,HIP_R_32F)); hipblasOperation_t ta=HIPBLAS_OP_T,tb=HIPBLAS_OP_N;
    CB(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))); CB(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    hipblasLtMatmulPreference_t pr; CB(hipblasLtMatmulPreferenceCreate(&pr)); size_t maxws=1<<20; CB(hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&maxws,sizeof(maxws)));
    hipblasLtMatmulHeuristicResult_t hr; int got=0; CB(hipblasLtMatmulAlgoGetHeuristic(hl,d,la,lb,lc,lc,pr,1,&hr,&got)); void* ws=nullptr; if(hr.workspaceSize) CK(hipMalloc(&ws,hr.workspaceSize));
    printf("{\"shape\":\"%s\",\"heuristic_got\":%d,\"ws\":%zu}\n",s.sn,got,(size_t)hr.workspaceSize);fflush(stdout);
    auto HB=[&](uint16_t*CC){ CB(hipblasLtMatmul(hl,d,&a1,B,la,A,lb,&b0,CC,lc,CC,lc,&hr.algo,ws,hr.workspaceSize,st)); };
    double fl=2.0*s.M*s.N*s.K;
    for(int r=0;r<rounds;r++){
      for(int i=0;i<5;i++) HB(C2); CK(hipStreamSynchronize(st));
      auto t0=std::chrono::steady_clock::now(); for(int i=0;i<s.it;i++) HB(C2); CK(hipStreamSynchronize(st)); double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/s.it;
      printf("{\"impl\":\"hipblaslt_direct\",\"round\":%d,\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ms\":%.4f,\"TF\":%.2f}\n",r,s.sn,s.M,s.N,s.K,ms,fl/ms*1e-9);fflush(stdout);
      for(auto&v:vs){ std::string kn=std::string(v.n)+"_"+s.sn; std::string co="solo/"+kn+".co"; hipFunction_t f; CK(hipModuleGetFunction(&f,mod(co),kn.c_str())); void* a[3]={&A,&B,&C}; int gx=s.N/128,gy=s.M/v.bM;
        for(int i=0;i<5;i++) CK(hipModuleLaunchKernel(f,gx,gy,1,256,1,1,v.dyn,st,a,nullptr)); CK(hipStreamSynchronize(st));
        t0=std::chrono::steady_clock::now(); for(int i=0;i<s.it;i++) CK(hipModuleLaunchKernel(f,gx,gy,1,256,1,1,v.dyn,st,a,nullptr)); CK(hipStreamSynchronize(st)); ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/s.it;
        printf("{\"impl\":\"%s\",\"round\":%d,\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ms\":%.4f,\"TF\":%.2f}\n",v.n,r,s.sn,s.M,s.N,s.K,ms,fl/ms*1e-9);fflush(stdout); }
      if(r==0){ // numeric agreement TileLang vs hipBLASLt: C row-major M x N, C2 column-major N x M = same memory layout
        size_t n=size_t(s.M)*s.N; std::vector<uint16_t> h1(n),h2(n); CK(hipMemcpy(h1.data(),C,n*2,hipMemcpyDeviceToHost)); CK(hipMemcpy(h2.data(),C2,n*2,hipMemcpyDeviceToHost)); double mx=0,mg=0; size_t bits=0; for(size_t i=0;i<n;i++){ double x=half2d(h1[i]),y=half2d(h2[i]); mx=std::max(mx,fabs(x-y)); mg=std::max(mg,fabs(x)); bits+=(h1[i]!=h2[i]); }
        printf("{\"check\":\"%s\",\"max_abs_diff\":%.6g,\"max_abs\":%.6g,\"rel\":%.3g,\"elements_differ\":%zu,\"n\":%zu}\n",s.sn,mx,mg,mx/mg,bits,n);fflush(stdout); }
    }
    CK(hipFree(A));CK(hipFree(B));CK(hipFree(C));CK(hipFree(C2)); if(ws) CK(hipFree(ws));
  }
  printf("DONE\n"); return 0; }
