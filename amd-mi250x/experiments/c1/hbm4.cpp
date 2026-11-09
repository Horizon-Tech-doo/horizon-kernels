// C1 s12: table-driven comparison of code objects with hipBLASLt called directly, in one process and on the same A, B, C buffers.
// hipBLASLt is one more arm in the per-round shuffled order, so it is not always first. Every arm's output is checked after a poisoned launch.
// C1 s19: hbm3 plus an allocation mode for A, B, C (argv 4: plain | contig | block), wall and monotonic time on every row.
// plain: hipMalloc. contig: hipExtMallocWithFlags(hipDeviceMallocContiguous). block: A, B, C carved from one contiguous allocation, 2 MiB aligned.
// argv: table rounds seed mode. Table line: label co kernel M N K bM bN bK dyn iters ("hipblaslt_direct - - M N K 0 0 0 0 iters" is the vendor arm).
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
static hipStream_t st;
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("HIPERR %s %s\n",#x,hipGetErrorString(e_));fflush(stdout);exit(1);} }while(0)
#define CB(x) do{hipblasStatus_t e_=(x); if(e_!=HIPBLAS_STATUS_SUCCESS){printf("HBERR %s %d\n",#x,(int)e_);fflush(stdout);exit(1);} }while(0)
static uint64_t r_=88172645463325252ull; static inline uint64_t rn(){r_^=r_<<13;r_^=r_>>7;r_^=r_<<17;return r_;}
static int amode=0; static char* blk=nullptr; static size_t blkoff=0,blksz=0;
static void* dalloc(size_t bytes){ void* p=nullptr; if(amode==0){ CK(hipMalloc(&p,bytes)); } else if(amode==1){ CK(hipExtMallocWithFlags(&p,bytes,hipDeviceMallocContiguous)); } else { size_t a=(blkoff+(2u<<20)-1)&~size_t((2u<<20)-1); if(a+bytes>blksz){printf("BLOCKFULL\n");fflush(stdout);exit(1);} p=blk+a; blkoff=a+bytes; } return p; }
static void dfree(void* p){ if(amode!=2) CK(hipFree(p)); }
static double wall(){ return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
static double mono(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static uint16_t* mk16(size_t n){ uint16_t* p=(uint16_t*)dalloc(n*2); std::vector<uint16_t> h(n);
  for(size_t i=0;i<n;i+=4){uint64_t v=rn(); for(int j=0;j<4&&i+j<n;j++){uint16_t m=(v>>(16*j))&0x83ff; h[i+j]=m|(14<<10);} }
  CK(hipMemcpy(p,h.data(),n*2,hipMemcpyHostToDevice)); return p;}
static std::map<std::string,hipModule_t> mods;
static hipModule_t mod(const std::string&fn){ auto it=mods.find(fn); if(it!=mods.end()) return it->second; std::ifstream in(fn,std::ios::binary); if(!in){printf("NOFILE %s\n",fn.c_str());fflush(stdout);exit(1);} std::vector<char> co((std::istreambuf_iterator<char>(in)),{}); char* keep=new char[co.size()]; std::copy(co.begin(),co.end(),keep); hipModule_t m; CK(hipModuleLoadData(&m,keep)); mods[fn]=m; return m; }
static double half2d(uint16_t v){ int s=v>>15,e=(v>>10)&31,m=v&1023; double x=(e==0)?m/1024.0*pow(2,-14):(1+m/1024.0)*pow(2,e-15); return s?-x:x; }
struct E{std::string label,co,kn; int M,N,K,bM,bN,bK; size_t dyn; int iters; hipFunction_t f=nullptr;};
struct Shape{int M,N,K; std::vector<E> arms;};
int main(int argc,char**argv){ std::string tab=argv[1]; int rounds=atoi(argv[2]); uint64_t seed=strtoull(argv[3],nullptr,10); std::string ms=argc>4?argv[4]:"plain"; amode=ms=="contig"?1:ms=="block"?2:0;
  CK(hipStreamCreate(&st)); hipblasLtHandle_t hl; CB(hipblasLtCreate(&hl));
  std::ifstream in(tab); std::string line; std::vector<std::string> keys; std::map<std::string,Shape> shapes;
  while(std::getline(in,line)){ if(line.empty())continue; std::stringstream s(line); E e; s>>e.label>>e.co>>e.kn>>e.M>>e.N>>e.K>>e.bM>>e.bN>>e.bK>>e.dyn>>e.iters;
    std::string k=std::to_string(e.M)+"x"+std::to_string(e.N)+"x"+std::to_string(e.K); if(!shapes.count(k)){keys.push_back(k); shapes[k]=Shape{e.M,e.N,e.K,{}};} shapes[k].arms.push_back(e); }
  float a1=1.f,b0=0.f;
  for(auto&key:keys){ Shape&S=shapes[key]; int M=S.M,N=S.N,K=S.K;
    if(amode==2){ blkoff=0; blksz=(size_t(M)*K+size_t(N)*K+size_t(M)*N)*2+(8u<<20); CK(hipExtMallocWithFlags((void**)&blk,blksz,hipDeviceMallocContiguous)); }
    uint16_t*A=mk16(size_t(M)*K),*B=mk16(size_t(N)*K),*C=mk16(size_t(M)*N);
    printf("{\"shape\":\"%s\",\"mode\":\"%s\",\"A\":\"%p\",\"B\":\"%p\",\"C\":\"%p\"}\n",key.c_str(),ms.c_str(),(void*)A,(void*)B,(void*)C);fflush(stdout);
    // same problem as torch's a @ b.T: column-major TN, first=B (K x N), second=A (K x M), result N x M, which is the same memory as row-major M x N
    hipblasLtMatrixLayout_t la,lb,lc; hipblasLtMatmulDesc_t d;
    CB(hipblasLtMatrixLayoutCreate(&la,HIP_R_16F,K,N,K)); CB(hipblasLtMatrixLayoutCreate(&lb,HIP_R_16F,K,M,K)); CB(hipblasLtMatrixLayoutCreate(&lc,HIP_R_16F,N,M,N));
    CB(hipblasLtMatmulDescCreate(&d,HIPBLAS_COMPUTE_32F,HIP_R_32F)); hipblasOperation_t ta=HIPBLAS_OP_T,tb=HIPBLAS_OP_N;
    CB(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))); CB(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    hipblasLtMatmulPreference_t pr; CB(hipblasLtMatmulPreferenceCreate(&pr)); size_t maxws=1<<20; CB(hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&maxws,sizeof(maxws)));
    hipblasLtMatmulHeuristicResult_t hr; int got=0; CB(hipblasLtMatmulAlgoGetHeuristic(hl,d,la,lb,lc,lc,pr,1,&hr,&got)); void* ws=nullptr; if(hr.workspaceSize) CK(hipMalloc(&ws,hr.workspaceSize));
    auto launch=[&](E&e){ if(e.co=="-"){ CB(hipblasLtMatmul(hl,d,&a1,B,la,A,lb,&b0,C,lc,C,lc,&hr.algo,ws,hr.workspaceSize,st)); } else { void* a[3]={&A,&B,&C}; CK(hipModuleLaunchKernel(e.f,N/e.bN,M/e.bM,1,256,1,1,e.dyn,st,a,nullptr)); } };
    for(auto&e:S.arms) if(e.co!="-") CK(hipModuleGetFunction(&e.f,mod(e.co),e.kn.c_str()));
    double fl=2.0*M*N*K; size_t n=size_t(M)*N; std::vector<uint16_t> h(n); std::map<uint64_t,std::vector<uint16_t>> uniq; std::map<uint64_t,std::string> uname; uint64_t hbcs=0;
    std::vector<int> ord(S.arms.size()); for(size_t i=0;i<ord.size();i++) ord[i]=i;
    for(int r=0;r<rounds;r++){
      uint64_t x=(seed*1000003ull+r*7919ull+0x9E3779B97F4A7C15ull)|1; auto rnd=[&](){x^=x<<13;x^=x>>7;x^=x<<17;return x;};
      for(int i=(int)ord.size()-1;i>0;i--) std::swap(ord[i],ord[rnd()%(i+1)]);
      int pos=0;
      for(int ai:ord){ E&e=S.arms[ai];
        for(int i=0;i<5;i++) launch(e); CK(hipStreamSynchronize(st));
        auto t0=std::chrono::steady_clock::now(); for(int i=0;i<e.iters;i++) launch(e); CK(hipStreamSynchronize(st));
        double ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/e.iters;
        printf("{\"round\":%d,\"pos\":%d,\"mode\":\"%s\",\"label\":\"%s\",\"shape\":\"%s\",\"iters\":%d,\"ms\":%.4f,\"TF\":%.2f,\"wall\":%.3f,\"mono\":%.3f}\n",r,pos,ms.c_str(),e.label.c_str(),key.c_str(),e.iters,ms_,fl/ms_*1e-9,wall(),mono());fflush(stdout); pos++;
        if(r==0||r==rounds-1){ CK(hipMemset(C,0xFF,n*2)); launch(e); CK(hipStreamSynchronize(st)); CK(hipMemcpy(h.data(),C,n*2,hipMemcpyDeviceToHost)); uint64_t cs=1469598103934665603ull; for(auto v:h) cs=(cs^v)*1099511628211ull;
          printf("{\"check\":\"cs\",\"round\":%d,\"label\":\"%s\",\"shape\":\"%s\",\"cs\":\"%016llx\"}\n",r,e.label.c_str(),key.c_str(),(unsigned long long)cs);fflush(stdout);
          if(!uniq.count(cs)){ uniq[cs]=h; uname[cs]=e.label; } if(e.co=="-") hbcs=cs; }
      }
    }
    // numeric agreement with hipBLASLt, once per distinct output
    auto&hb=uniq[hbcs];
    for(auto&kv:uniq){ double mx=0,mg=0; size_t bits=0; for(size_t i=0;i<n;i++){ double x=half2d(kv.second[i]),y=half2d(hb[i]); mx=std::max(mx,fabs(x-y)); mg=std::max(mg,fabs(y)); bits+=(kv.second[i]!=hb[i]); }
      printf("{\"check\":\"vs_hipblaslt\",\"shape\":\"%s\",\"cs\":\"%016llx\",\"first_label\":\"%s\",\"max_abs_diff\":%.6g,\"max_abs\":%.6g,\"rel\":%.3g,\"elements_differ\":%zu,\"n\":%zu}\n",key.c_str(),(unsigned long long)kv.first,uname[kv.first].c_str(),mx,mg,mx/mg,bits,n);fflush(stdout); }
    dfree(A);dfree(B);dfree(C); if(amode==2){ CK(hipFree(blk)); blk=nullptr; } if(ws) CK(hipFree(ws));
  }
  printf("DONE\n"); return 0; }
