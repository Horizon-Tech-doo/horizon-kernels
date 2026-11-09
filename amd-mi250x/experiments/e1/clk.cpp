// clock vs load: stamps for no-MFMA kernels and MFMA at fractions of the CUs.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
static hipStream_t st;
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("HIPERR %s %s\n",#x,hipGetErrorString(e_));fflush(stdout);exit(1);} }while(0)
struct Tr { unsigned long long re,r0,r1,me,m0,m1; unsigned hw; unsigned bk; };
static hipModule_t load(const char*fn){ std::ifstream in(fn,std::ios::binary); std::vector<char> co((std::istreambuf_iterator<char>(in)),{}); hipModule_t m; CK(hipModuleLoadData(&m,co.data())); return m; }
int main(int argc,char**argv){ int round=argc>1?atoi(argv[1]):0; CK(hipStreamCreate(&st));
  hipModule_t m1=load("micro.co"),m2=load("micro2.co"); size_t n4=4096ull*4096; uint16_t*A,*B; CK(hipMalloc(&A,n4*2)); CK(hipMalloc(&B,n4*2));
  { std::vector<uint16_t> h(n4); unsigned long long r_=88172645463325252ull; for(size_t i=0;i<n4;i+=4){ r_^=r_<<13;r_^=r_>>7;r_^=r_<<17; for(int j=0;j<4&&i+j<n4;j++) h[i+j]=((r_>>(16*j))&0x83ff)|(14<<10);} CK(hipMemcpy(A,h.data(),n4*2,hipMemcpyHostToDevice)); CK(hipMemcpy(B,h.data(),n4*2,hipMemcpyHostToDevice)); }
  float*C; CK(hipMalloc(&C,1<<20)); Tr* tr; size_t ntr=4*4096; CK(hipMalloc(&tr,ntr*sizeof(Tr))); std::vector<Tr> h(ntr);
  struct J{const char* name; int mod; int args; int nk; int B;};
  std::vector<J> js={{"nop_tr",2,0,20000,110},{"valu_tr",2,0,40000,110},{"mfma_w1_tr",2,1,640,110},{"mfma_tr",1,1,640,11},{"mfma_tr",1,1,640,28},{"mfma_tr",1,1,640,55},{"mfma_tr",1,1,640,83},{"mfma_tr",1,1,640,110},{"mfma_tr",1,1,640,220},{"mfma32_tr",1,1,640,28},{"mfma32_tr",1,1,640,55},{"mfma32_tr",1,1,640,110}};
  for(auto&j:js){ hipFunction_t f; CK(hipModuleGetFunction(&f,j.mod==1?m1:m2,j.name)); int gx=j.B,gy=1,nk=j.nk;
    std::vector<void*> a; if(j.args) a={&A,&B,&C,&tr,&nk,&gx}; else a={&C,&tr,&nk,&gx};
    for(int w=0;w<3;w++) CK(hipModuleLaunchKernel(f,gx,gy,1,256,1,1,0,st,a.data(),nullptr)); CK(hipStreamSynchronize(st));
    auto t0=std::chrono::steady_clock::now(); for(int i=0;i<6;i++) CK(hipModuleLaunchKernel(f,gx,gy,1,256,1,1,0,st,a.data(),nullptr)); CK(hipStreamSynchronize(st)); double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/6;
    CK(hipMemcpy(h.data(),tr,j.B*4*sizeof(Tr),hipMemcpyDeviceToHost));
    std::vector<double> mz,cy; for(int i=0;i<j.B*4;i++){ double dm=h[i].m1-h[i].m0,dr=h[i].r1-h[i].r0; if(dr>0){mz.push_back(dm/dr*25.0);cy.push_back(dm);} } std::sort(mz.begin(),mz.end()); std::sort(cy.begin(),cy.end());
    printf("{\"m\":\"clk\",\"round\":%d,\"k\":\"%s\",\"B\":%d,\"nk\":%d,\"ms\":%.4f,\"mhz_p5\":%.1f,\"mhz_med\":%.1f,\"mhz_p95\":%.1f,\"ticks_med\":%.0f}\n",round,j.name,j.B,j.nk,ms,mz[mz.size()/20],mz[mz.size()/2],mz[mz.size()*19/20],cy[cy.size()/2]); fflush(stdout); }
  return 0; }
