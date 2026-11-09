// E1 GEMM harness: table-driven, one hipModule per .co, shapes prebuilt, interleaved in a round.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
static hipStream_t st;
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("HIPERR %s %s\n",#x,hipGetErrorString(e_));fflush(stdout);exit(1);} }while(0)
static uint64_t r_=88172645463325252ull; static inline uint64_t rn(){r_^=r_<<13;r_^=r_>>7;r_^=r_<<17;return r_;}
static uint16_t* mk16(size_t n){ uint16_t* p; CK(hipMalloc(&p,n*2)); std::vector<uint16_t> h(n);
  for(size_t i=0;i<n;i+=4){uint64_t v=rn(); for(int j=0;j<4&&i+j<n;j++){uint16_t m=(v>>(16*j))&0x83ff; h[i+j]=m|(14<<10);} }
  CK(hipMemcpy(p,h.data(),n*2,hipMemcpyHostToDevice)); return p;}
struct Tr { unsigned long long re,r0,r1,me,m0,m1; unsigned hw; unsigned bk; };
struct E{std::string label,co,kn; int M,N,K,bM,bN,bK; size_t dyn; int trace; int iters;};
static std::map<std::string,hipModule_t> mods;
static hipModule_t mod(const std::string&fn){ auto it=mods.find(fn); if(it!=mods.end()) return it->second; std::ifstream in(fn,std::ios::binary); if(!in){printf("NOFILE %s\n",fn.c_str());exit(1);} std::vector<char> co((std::istreambuf_iterator<char>(in)),{}); hipModule_t m; CK(hipModuleLoadData(&m,co.data())); mods[fn]=m; return m; }
struct S{uint16_t*A,*B,*C;};
int main(int argc,char**argv){ std::string tab=argv[1]; int round=atoi(argv[2]);
  CK(hipStreamCreate(&st)); std::ifstream in(tab); std::vector<E> es; std::string line; while(std::getline(in,line)){ if(line.empty())continue; std::stringstream s(line); E e; s>>e.label>>e.co>>e.kn>>e.M>>e.N>>e.K>>e.bM>>e.bN>>e.bK>>e.dyn>>e.trace>>e.iters; es.push_back(e);} 
  std::map<std::string,S> shapes; for(auto&e:es){ std::string k=std::to_string(e.M)+"x"+std::to_string(e.N)+"x"+std::to_string(e.K); if(!shapes.count(k)){ S s; s.A=mk16(size_t(e.M)*e.K); s.B=mk16(size_t(e.N)*e.K); s.C=mk16(size_t(e.M)*e.N); shapes[k]=s; } }
  Tr* tr; size_t ntr=4*8192; CK(hipMalloc(&tr,ntr*sizeof(Tr))); std::vector<Tr> ht(ntr);
  std::rotate(es.begin(),es.begin()+((round*13)%es.size()),es.end());
  for(auto&e:es){ std::string k=std::to_string(e.M)+"x"+std::to_string(e.N)+"x"+std::to_string(e.K); S s=shapes[k]; hipFunction_t f; CK(hipModuleGetFunction(&f,mod(e.co),e.kn.c_str()));
    void* a[4]={&s.A,&s.B,&s.C,&tr}; int gx=e.N/e.bN,gy=e.M/e.bM;
    for(int i=0;i<5;i++) CK(hipModuleLaunchKernel(f,gx,gy,1,256,1,1,e.dyn,st,a,nullptr)); CK(hipStreamSynchronize(st));
    auto t0=std::chrono::steady_clock::now(); for(int i=0;i<e.iters;i++) CK(hipModuleLaunchKernel(f,gx,gy,1,256,1,1,e.dyn,st,a,nullptr)); CK(hipStreamSynchronize(st));
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/e.iters;
    std::vector<uint16_t> hc(size_t(e.M)*e.N); CK(hipMemcpy(hc.data(),s.C,hc.size()*2,hipMemcpyDeviceToHost)); uint64_t cs=1469598103934665603ull; for(auto v:hc){cs=(cs^v)*1099511628211ull;}
    double fl=2.0*e.M*e.N*e.K; printf("{\"cs\":\"%016llx\",\"round\":%d,\"label\":\"%s\",\"k\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"bM\":%d,\"bK\":%d,\"B\":%d,\"dyn\":%zu,\"iters\":%d,\"ms\":%.4f,\"TF\":%.2f}\n",(unsigned long long)cs,round,e.label.c_str(),e.kn.c_str(),e.M,e.N,e.K,e.bM,e.bK,gx*gy,e.dyn,e.iters,ms,fl/ms*1e-9); fflush(stdout);
    if(e.trace){ int nw=gx*gy*4; CK(hipMemcpy(ht.data(),tr,nw*sizeof(Tr),hipMemcpyDeviceToHost)); char fn[256]; snprintf(fn,256,"traces2/%s_r%d.bin",e.kn.c_str(),round); FILE*fo=fopen(fn,"wb"); fwrite(ht.data(),sizeof(Tr),nw,fo); fclose(fo); }
  }
  printf("DONE\n"); return 0; }
