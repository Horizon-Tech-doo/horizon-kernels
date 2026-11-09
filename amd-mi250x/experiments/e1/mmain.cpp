// E1 micro harness: mfma_only / mfma32_only block-count scan + traced timestamps.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <cstring>
#include <algorithm>
static hipModule_t mod; static hipStream_t st;
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("HIPERR %s %s\n",#x,hipGetErrorString(e_));fflush(stdout);exit(1);} }while(0)
static uint64_t r_=88172645463325252ull; static inline uint64_t rn(){r_^=r_<<13;r_^=r_>>7;r_^=r_<<17;return r_;}
static uint16_t* mk16(size_t n){ uint16_t* p; CK(hipMalloc(&p,n*2)); std::vector<uint16_t> h(n);
  for(size_t i=0;i<n;i+=4){uint64_t v=rn(); for(int j=0;j<4&&i+j<n;j++){uint16_t m=(v>>(16*j))&0x83ff; h[i+j]=m|(14<<10);} }
  CK(hipMemcpy(p,h.data(),n*2,hipMemcpyHostToDevice)); return p;}
struct Tr { unsigned long long re,r0,r1,me,m0,m1; unsigned hw; unsigned bk; };
static double run(hipFunction_t f,int gx,int gy,size_t dyn,std::vector<void*> a,int warm,int iters){
  for(int i=0;i<warm;i++) CK(hipModuleLaunchKernel(f,gx,gy,1,256,1,1,dyn,st,a.data(),nullptr));
  CK(hipStreamSynchronize(st));
  auto t0=std::chrono::steady_clock::now();
  for(int i=0;i<iters;i++) CK(hipModuleLaunchKernel(f,gx,gy,1,256,1,1,dyn,st,a.data(),nullptr));
  CK(hipStreamSynchronize(st));
  return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/iters;
}
static hipFunction_t F(const char*n){hipFunction_t f; CK(hipModuleGetFunction(&f,mod,n)); return f;}
struct G{int gx,gy;};
int main(int argc,char**argv){
  std::string mode=argc>1?argv[1]:"scan"; int round=argc>2?atoi(argv[2]):0; double sus=argc>3?atof(argv[3]):20;
  std::ifstream in("micro.co",std::ios::binary); std::vector<char> co((std::istreambuf_iterator<char>(in)),{});
  CK(hipModuleLoadData(&mod,co.data())); CK(hipStreamCreate(&st));
  int cu=0,wc=0,mx=0; CK(hipDeviceGetAttribute(&cu,hipDeviceAttributeMultiprocessorCount,0)); CK(hipDeviceGetAttribute(&wc,hipDeviceAttributeWallClockRate,0)); CK(hipDeviceGetAttribute(&mx,hipDeviceAttributeClockRate,0));
  printf("{\"dev\":1,\"cu\":%d,\"wall_khz\":%d,\"max_clock_mhz\":%d}\n",cu,wc,mx);
  size_t n4=4096ull*4096; uint16_t*A=mk16(n4),*B=mk16(n4); float*C; CK(hipMalloc(&C,1<<20));
  const char* kn[2]={"mfma_only","mfma32_only"}; int zero=0;
  std::vector<std::pair<size_t,int>> Ps={{0,3},{21000,3},{32768,2},{65024,1}};
  if(mode=="scan"){
    std::vector<G> gs={{11,10},{22,10},{33,10},{44,10},{55,10},{66,10},{77,10},{40,22},{41,22},{45,22},{32,32},{50,20},{1024,1},{44,25},{40,30},{40,33},{1320,1},{1321,1},{44,35},{50,33},{66,30},{55,40},{60,40},{66,40}};
    // rotate order per round
    for(int kk=0;kk<2;kk++){ int k=(kk+round)&1; hipFunction_t f=F(kn[k]);
      for(size_t pi=0;pi<Ps.size();pi++){ auto p=Ps[(pi+round)%Ps.size()];
        for(size_t gi=0;gi<gs.size();gi++){ G g=gs[(gi+round*7)%gs.size()]; int nk=64; int blocks=g.gx*g.gy;
          double ms=run(f,g.gx,g.gy,p.first,{&A,&B,&C,&nk,&zero},5,40);
          printf("{\"m\":\"scan\",\"round\":%d,\"k\":\"%s\",\"P\":%d,\"dyn\":%zu,\"gx\":%d,\"gy\":%d,\"B\":%d,\"nk\":%d,\"ms\":%.4f,\"TF\":%.2f}\n",round,kn[k],p.second,p.first,g.gx,g.gy,blocks,nk,ms,blocks*nk*2097152.0/ms*1e-9); fflush(stdout);
    } } }
    // long block: launch overhead negligible
    std::vector<G> gl={{32,32},{40,33},{40,22},{44,25},{33,10},{22,10},{11,10}};
    for(int k=0;k<2;k++){ hipFunction_t f=F(kn[k]); for(auto p:Ps) for(auto g:gl){ int nk=640,blocks=g.gx*g.gy; double ms=run(f,g.gx,g.gy,p.first,{&A,&B,&C,&nk,&zero},1,8);
      printf("{\"m\":\"long\",\"round\":%d,\"k\":\"%s\",\"P\":%d,\"dyn\":%zu,\"gx\":%d,\"gy\":%d,\"B\":%d,\"nk\":%d,\"ms\":%.4f,\"TF\":%.2f}\n",round,kn[k],p.second,p.first,g.gx,g.gy,blocks,nk,ms,blocks*nk*2097152.0/ms*1e-9); fflush(stdout);} }
    { hipFunction_t f=F("empty_k"); for(G g:std::vector<G>{{1,1},{32,32},{40,33}}){ double ms=run(f,g.gx,g.gy,0,{&C},5,40); printf("{\"m\":\"empty\",\"round\":%d,\"B\":%d,\"ms\":%.4f}\n",round,g.gx*g.gy,ms);fflush(stdout);} }
  } else if(mode=="trace"){
    const char* tn[2]={"mfma_tr","mfma32_tr"};
    std::vector<G> gs={{11,10},{33,10},{40,22},{32,32},{44,25},{40,33},{1321,1}};
    Tr* tr; size_t ntr=4*4096; CK(hipMalloc(&tr,ntr*sizeof(Tr))); std::vector<Tr> h(ntr);
    for(int k=0;k<2;k++){ hipFunction_t f=F(tn[k]); for(auto p:Ps) for(auto g:gs){ for(int nk:{64,640}){ int blocks=g.gx*g.gy;
      double ms=run(f,g.gx,g.gy,p.first,{&A,&B,&C,&tr,&nk,&g.gx},2,3);
      CK(hipMemcpy(h.data(),tr,blocks*4*sizeof(Tr),hipMemcpyDeviceToHost));
      char fn[256]; snprintf(fn,256,"traces/%s_P%d_d%zu_B%d_g%dx%d_nk%d_r%d.bin",tn[k],p.second,p.first,blocks,g.gx,g.gy,nk,round);
      FILE*fo=fopen(fn,"wb"); fwrite(h.data(),sizeof(Tr),blocks*4,fo); fclose(fo);
      printf("{\"m\":\"trace\",\"round\":%d,\"k\":\"%s\",\"P\":%d,\"dyn\":%zu,\"gx\":%d,\"gy\":%d,\"B\":%d,\"nk\":%d,\"ms\":%.4f,\"TF\":%.2f,\"f\":\"%s\"}\n",round,tn[k],p.second,p.first,g.gx,g.gy,blocks,nk,ms,blocks*nk*2097152.0/ms*1e-9,fn); fflush(stdout);
    } } }
  } else if(mode=="sustain"){
    // sustained: repeat long dispatches for `sus` seconds, per kernel; caller polls rocm-smi meanwhile
    for(int k=0;k<2;k++){ hipFunction_t f=F(kn[k]); int nk=640; G g={40,33}; int blocks=g.gx*g.gy; auto t0=std::chrono::steady_clock::now(); int n=0;
      auto t1=t0; do{ for(int i=0;i<20;i++) CK(hipModuleLaunchKernel(f,g.gx,g.gy,1,256,1,1,0,st,std::vector<void*>{&A,&B,&C,&nk,&zero}.data(),nullptr)); CK(hipStreamSynchronize(st)); n+=20; t1=std::chrono::steady_clock::now(); }while(std::chrono::duration<double>(t1-t0).count()<sus);
      double s=std::chrono::duration<double>(t1-t0).count(); printf("{\"m\":\"sustain\",\"k\":\"%s\",\"B\":%d,\"nk\":%d,\"n\":%d,\"s\":%.3f,\"TF\":%.2f}\n",kn[k],blocks,nk,n,s,n*blocks*nk*2097152.0/s*1e-12); fflush(stdout); }
  }
  printf("DONE\n"); return 0;
}
