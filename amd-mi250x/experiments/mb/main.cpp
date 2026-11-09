#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <map>
#include <fstream>
#include <cstring>
static hipModule_t mod; static hipStream_t st;
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("HIPERR %s %s\n",#x,hipGetErrorString(e_));fflush(stdout);exit(1);} }while(0)
static uint64_t r_=88172645463325252ull; static inline uint64_t rn(){r_^=r_<<13;r_^=r_>>7;r_^=r_<<17;return r_;}
static uint16_t* mk16(size_t n){ uint16_t* p; CK(hipMalloc(&p,n*2)); std::vector<uint16_t> h(n);
  for(size_t i=0;i<n;i+=4){uint64_t v=rn(); for(int j=0;j<4&&i+j<n;j++){uint16_t m=(v>>(16*j))&0x83ff; h[i+j]=m|(14<<10);} }
  CK(hipMemcpy(p,h.data(),n*2,hipMemcpyHostToDevice)); return p;}
struct Arg{void* p;};
static double run(const char* name,int gx,int gy,int gz,size_t dyn,std::vector<void*> args,int iters=40){
  hipFunction_t f; hipError_t e=hipModuleGetFunction(&f,mod,name); if(e!=hipSuccess){printf("{\"k\":\"%s\",\"err\":\"nofunc\"}\n",name);return -1;}
  std::vector<void*> ptr; for(auto&a:args) ptr.push_back(a);
  for(int i=0;i<5;i++) CK(hipModuleLaunchKernel(f,gx,gy,gz,256,1,1,dyn,st,ptr.data(),nullptr));
  CK(hipStreamSynchronize(st));
  auto t0=std::chrono::steady_clock::now();
  for(int i=0;i<iters;i++) CK(hipModuleLaunchKernel(f,gx,gy,gz,256,1,1,dyn,st,ptr.data(),nullptr));
  CK(hipStreamSynchronize(st));
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/iters;
  return ms;
}
static void G(const char* k,int M,int N,int K,int bM,int bN,int bK,size_t dyn,uint16_t*A,uint16_t*B,uint16_t*C){
  void* a[3]={&A,&B,&C}; double ms=run(k,N/bN,M/bM,1,dyn,{a[0],a[1],a[2]});
  std::vector<uint16_t> hc(size_t(M)*N); CK(hipMemcpy(hc.data(),C,hc.size()*2,hipMemcpyDeviceToHost)); uint64_t cs=1469598103934665603ull; for(auto v:hc){cs=(cs^v)*1099511628211ull;}
  double fl=2.0*M*N*K; printf("{\"cs\":\"%016llx\",\"k\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"bM\":%d,\"bN\":%d,\"bK\":%d,\"dyn\":%zu,\"ms\":%.4f,\"TF\":%.2f}\n",(unsigned long long)cs,k,M,N,K,bM,bN,bK,dyn,ms,fl/ms*1e-9); fflush(stdout);}
int main(){
  std::ifstream in("kernels.co",std::ios::binary); std::vector<char> co((std::istreambuf_iterator<char>(in)),{});
  CK(hipModuleLoadData(&mod,co.data())); CK(hipStreamCreate(&st));
  size_t n4=4096ull*4096, n8=8192ull*8192;
  uint16_t*A=mk16(n4),*B=mk16(n4),*C=mk16(n4),*A8=mk16(n8),*B8=mk16(n8),*C8=mk16(n8);
  size_t full=65024;
  auto g4=[&](const char*k,int bM,int bK,std::vector<size_t> ds){for(size_t d:ds) G(k,4096,4096,4096,bM,128,bK,d,A,B,C);};
  // micro
  { float* F; CK(hipMalloc(&F,1<<20)); int nk=64; void*a[5];
    for(size_t d:{(size_t)0,(size_t)65024}){ int zero=0,one=1; double ms=run("mfma_only",32,32,1,d,{&A,&B,&F,&nk,&zero}); printf("{\"k\":\"mfma_only\",\"dyn\":%zu,\"ms\":%.4f,\"TF\":%.2f}\n",d,ms,1024*64*2.0*128*128*64/ms*1e-9); ms=run("mfma_only",32,32,1,d,{&A,&B,&F,&nk,&one}); printf("{\"k\":\"mfma_2bar\",\"dyn\":%zu,\"ms\":%.4f,\"TF\":%.2f}\n",d,ms,1024*64*2.0*128*128*64/ms*1e-9);fflush(stdout);}
    for(size_t d:{(size_t)65024,(size_t)32768}){ double ms=run("bar_only",32,32,1,d,{&F,&nk}); printf("{\"k\":\"bar_only\",\"dyn\":%zu,\"ms\":%.4f}\n",d,ms);
      ms=run("lds_r",32,32,1,d,{&F,&nk}); printf("{\"k\":\"lds_r\",\"dyn\":%zu,\"ms\":%.4f}\n",d,ms);
      ms=run("lds_w",32,32,1,d,{&F,&nk}); printf("{\"k\":\"lds_w\",\"dyn\":%zu,\"ms\":%.4f}\n",d,ms);fflush(stdout);}
    for(size_t d:{(size_t)65024,(size_t)32768,(size_t)16384}){ int K=4096; double ms=run("gl8",32,32,1,d,{&A,&B,&F,&K,&nk}); printf("{\"k\":\"gl8_4k\",\"dyn\":%zu,\"ms\":%.4f,\"GBps\":%.1f}\n",d,ms,1024.0*64*32768/ms*1e-6);
      K=8192; int nk8=128; ms=run("gl8",64,64,1,d,{&A8,&B8,&F,&K,&nk8}); printf("{\"k\":\"gl8_8k\",\"dyn\":%zu,\"ms\":%.4f,\"GBps\":%.1f}\n",d,ms,4096.0*128*32768/ms*1e-6);fflush(stdout);}
    for(size_t d:{(size_t)0,(size_t)65024,(size_t)32768}){ int s=0; double ms=run("scr8",32,32,1,d,{&F,&nk,&s}); printf("{\"k\":\"scr8\",\"dyn\":%zu,\"ms\":%.4f}\n",d,ms);fflush(stdout);}
  }
  g4("b12832",128,32,{16384,21000,32768,full}); g4("b12832w4",128,32,{16384,21000,32768,full});
  g4("b12864",128,64,{32768,full}); g4("b12864w4",128,64,{32768,full});
  g4("b25632",256,32,{24576,full}); g4("b25632w2",256,32,{24576,full});
  g4("r12832",128,32,{16384,32768,full}); g4("r12864",128,64,{32768,full}); g4("r25632",256,32,{24576,full});
  g4("rr12832",128,32,{16384,32768,full}); g4("rr12832w4",128,32,{16384,21000,32768,full});
  g4("rr12864",128,64,{32768,full}); g4("rr12864w3",128,64,{32768,full});
  g4("rr25632",256,32,{24576,full}); g4("rr25632w2",256,32,{24576,full});
  G("b12864_8k",8192,8192,8192,128,128,64,32768,A8,B8,C8); G("b25632_8k",8192,8192,8192,256,128,32,24576,A8,B8,C8); G("b25632w2_8k",8192,8192,8192,256,128,32,24576,A8,B8,C8);
  G("rr12864_8kw3",8192,8192,8192,128,128,64,32768,A8,B8,C8); G("rr25632_8kw2",8192,8192,8192,256,128,32,24576,A8,B8,C8);
  // 32768x1024x2048
  { size_t na=32768ull*2048,nb=1024ull*2048,nc=32768ull*1024; uint16_t*Aa=mk16(na),*Bb=mk16(nb),*Cc=mk16(nc);
    G("b12864_32k",32768,1024,2048,128,128,64,32768,Aa,Bb,Cc); G("b25632_32k",32768,1024,2048,256,128,32,24576,Aa,Bb,Cc); G("b25632w2_32k",32768,1024,2048,256,128,32,24576,Aa,Bb,Cc);
    CK(hipFree(Aa));CK(hipFree(Bb));CK(hipFree(Cc)); }
  // MoE synthetic: P=48896, NB=764, E=256
  { int P=48896,NB=764,E=256,I=512,H=2048,Tn=4096,Pairs=32768; uint16_t*X=mk16(size_t(Tn)*H),*W13=mk16(size_t(E)*2*I*H),*W2=mk16(size_t(E)*H*I),*Inter=mk16(size_t(P)*I),*Y=mk16(size_t(Pairs)*H);
    int*Bexp,*Sid; float*Wt; CK(hipMalloc(&Bexp,NB*4)); CK(hipMalloc(&Sid,P*4)); CK(hipMalloc(&Wt,Pairs*4));
    std::vector<int> be(NB),si(P); std::vector<float> wt(Pairs,0.125f);
    for(int b=0;b<NB;b++) be[b]=(long)b*E/NB;
    for(int i=0;i<P;i++) si[i]= i<Pairs? i : Pairs;   // padding at the end, not spread over blocks (synthetic)
    CK(hipMemcpy(Bexp,be.data(),NB*4,hipMemcpyHostToDevice)); CK(hipMemcpy(Sid,si.data(),P*4,hipMemcpyHostToDevice)); CK(hipMemcpy(Wt,wt.data(),Pairs*4,hipMemcpyHostToDevice));
    auto gu=[&](const char*k,size_t d){double ms=run(k,NB,I/64,1,d,{&Bexp,&Inter,&Sid,&W13,&X}); printf("{\"k\":\"%s\",\"dyn\":%zu,\"ms\":%.4f}\n",k,d,ms);fflush(stdout);};
    auto dn=[&](const char*k,int bN,size_t d){double ms=run(k,NB,H/bN,1,d,{&Bexp,&Inter,&Sid,&W2,&Wt,&Y}); printf("{\"k\":\"%s\",\"dyn\":%zu,\"ms\":%.4f}\n",k,d,ms);fflush(stdout);};
    gu("gu",24576); gu("gu_ng",24576); gu("gu_ne",24576);
    dn("dn",128,24576); dn("dn_ng",128,24576); dn("dn_ne",128,24576); dn("dn_ngne",128,24576); dn("dn6",64,16384);
    { size_t np=size_t(P)*H; uint16_t*Xs=mk16(np); for(const char* k:{"gu_pre","gu_prenng"}){ double ms=run(k,NB,I/64,1,24576,{&Bexp,&Inter,&W13,&Xs}); printf("{\"k\":\"%s\",\"dyn\":24576,\"ms\":%.4f}\n",k,ms);fflush(stdout);} }
  }
  printf("DONE\n"); return 0;
}
