// N4 round 3 harness. Reallocation steps in one process, all arms read at every step. hipBLASLt direct and solo code objects on the same data.
// Usage: hbr M N K mode seq iters warm arm...
//   mode: plain | contig     seq: one letter per step: N nothing, A/B/C reallocate that buffer, X all of them (hb's output and kernels' output are separate buffers; C means both).
//        a/b/c shift A, B or both by the next offset of DELTA inside the same allocation, with the same data rewritten at the new address (A's other offset returns to 0). The list restarts after X.
//   arm : hb | name,co,bM,bN,threads,dyn. Arm order rotates by one position per step. Each reallocation frees, holds a dummy of a different size, copies the same data back, frees the dummy.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <sstream>
static hipStream_t st;
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("HIPERR %s %s\n",#x,hipGetErrorString(e_));fflush(stdout);exit(1);} }while(0)
#define CB(x) do{hipblasStatus_t e_=(x); if(e_!=HIPBLAS_STATUS_SUCCESS){printf("HBERR %s %d\n",#x,(int)e_);fflush(stdout);exit(1);} }while(0)
static uint64_t r_=88172645463325252ull; static inline uint64_t rn(){r_^=r_<<13;r_^=r_>>7;r_^=r_<<17;return r_;}
static uint16_t f2h(float f){ uint32_t x; memcpy(&x,&f,4); uint32_t s=(x>>16)&0x8000; int e=(int)((x>>23)&255)-127+15; uint32_t m=x&0x7fffff;
  if(e>=31) return s|0x7c00; if(e<=0){ if(e<-10) return s; m|=0x800000; int sh=14-e; uint32_t h=m>>sh; if((m>>(sh-1))&1) h++; return s|h; }
  uint32_t h=(e<<10)|(m>>13); if(m&0x1000) h++; return s|h; }
static int amode=0;
static void* dal(size_t b){ void* p=nullptr; if(amode==1) CK(hipExtMallocWithFlags(&p,b,hipDeviceMallocContiguous)); else CK(hipMalloc(&p,b)); return p; }
static std::vector<uint16_t> gen(size_t n){ std::vector<uint16_t> h(n); for(size_t i=0;i<n;i+=2){ double u=(rn()>>11)*(1.0/9007199254740992.0),v=(rn()>>11)*(1.0/9007199254740992.0); if(u<1e-300)u=1e-300; double R=sqrt(-2*log(u)),t=6.283185307179586*v; h[i]=f2h((float)(R*cos(t))); if(i+1<n) h[i+1]=f2h((float)(R*sin(t))); } return h; }
static hipModule_t loadco(const std::string&fn){ std::ifstream in(fn,std::ios::binary); if(!in){printf("{\"err\":\"noco %s\"}\n",fn.c_str());fflush(stdout);exit(1);} std::vector<char> co((std::istreambuf_iterator<char>(in)),{}); char* keep=new char[co.size()]; std::copy(co.begin(),co.end(),keep); hipModule_t m; CK(hipModuleLoadData(&m,keep)); return m; }
static double nowe(){ return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
struct Arm{std::string name,co; int bM,bN,th; size_t dyn; hipFunction_t f;};
static std::vector<std::string> split(const std::string&s,char c){ std::vector<std::string> o; std::stringstream ss(s); std::string t; while(std::getline(ss,t,c)) o.push_back(t); return o; }
int main(int argc,char**argv){ if(argc<8) return 2; int M=atoi(argv[1]),N=atoi(argv[2]),K=atoi(argv[3]); amode=!strcmp(argv[4],"contig"); std::string seq=argv[5]; int it=atoi(argv[6]),warm=atoi(argv[7]);
  CK(hipStreamCreate(&st)); hipblasLtHandle_t hl; CB(hipblasLtCreate(&hl));
  std::vector<Arm> arms; for(int i=8;i<argc;i++){ if(!strcmp(argv[i],"hb")){ Arm a; a.name="hb"; arms.push_back(a); continue;} auto p=split(argv[i],','); Arm a; a.name=p[0];a.co=p[1];a.bM=atoi(p[2].c_str());a.bN=atoi(p[3].c_str());a.th=atoi(p[4].c_str());a.dyn=atol(p[5].c_str()); CK(hipModuleGetFunction(&a.f,loadco(a.co),a.name.c_str())); arms.push_back(a);}
  size_t na=size_t(M)*K,nb=size_t(N)*K,nc=size_t(M)*N; auto ha=gen(na),hb=gen(nb);
  const size_t PAD=size_t(32)<<20; std::vector<size_t> DELTA={0,256,1024,4096,16384,65536,1u<<20,2u<<20,6u<<20}; if(getenv("HBR_DELTAS")){ DELTA.clear(); for(auto&t:split(getenv("HBR_DELTAS"),',')) DELTA.push_back(strtoull(t.c_str(),nullptr,10)); } size_t di=0,aoff=0,boff=0;
  uint16_t*A=(uint16_t*)dal(na*2+PAD),*B=(uint16_t*)dal(nb*2+PAD),*C=(uint16_t*)dal(nc*2),*C2=(uint16_t*)dal(nc*2);
  CK(hipMemcpy(A,ha.data(),na*2,hipMemcpyHostToDevice)); CK(hipMemcpy(B,hb.data(),nb*2,hipMemcpyHostToDevice)); CK(hipMemset(C,0,nc*2)); CK(hipMemset(C2,0,nc*2));
  hipblasLtMatrixLayout_t la,lb,lc; hipblasLtMatmulDesc_t d;
  CB(hipblasLtMatrixLayoutCreate(&la,HIP_R_16F,K,N,K)); CB(hipblasLtMatrixLayoutCreate(&lb,HIP_R_16F,K,M,K)); CB(hipblasLtMatrixLayoutCreate(&lc,HIP_R_16F,N,M,N));
  CB(hipblasLtMatmulDescCreate(&d,HIPBLAS_COMPUTE_32F,HIP_R_32F)); hipblasOperation_t ta=HIPBLAS_OP_T,tb=HIPBLAS_OP_N;
  CB(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))); CB(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
  hipblasLtMatmulPreference_t pr; CB(hipblasLtMatmulPreferenceCreate(&pr)); size_t maxws=1<<20; CB(hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&maxws,sizeof(maxws)));
  hipblasLtMatmulHeuristicResult_t hr; int got=0; CB(hipblasLtMatmulAlgoGetHeuristic(hl,d,la,lb,lc,lc,pr,1,&hr,&got)); void* ws=nullptr; if(hr.workspaceSize) CK(hipMalloc(&ws,hr.workspaceSize));
  float a1=1.f,b0=0.f; double fl=2.0*M*N*K; int n=arms.size(); size_t dm[]={64,300,900,2048,128,1500,5120,512};
  auto launch=[&](Arm&a){ if(a.name=="hb") CB(hipblasLtMatmul(hl,d,&a1,B,la,A,lb,&b0,C2,lc,C2,lc,&hr.algo,ws,hr.workspaceSize,st));
    else { void* args[3]={&A,&B,&C}; int gx=(N+a.bN-1)/a.bN,gy=(M+a.bM-1)/a.bM; CK(hipModuleLaunchKernel(a.f,gx,gy,1,a.th,1,1,a.dyn,st,args,nullptr)); } };
  for(size_t s=0;s<seq.size();s++){ char q=seq[s]; bool ra=q=='A'||q=='X',rb=q=='B'||q=='X',rc=q=='C'||q=='X';
    if(ra||rb||rc){ CK(hipStreamSynchronize(st)); if(ra){ A=(uint16_t*)((char*)A-aoff); aoff=0; CK(hipFree(A)); } if(rb){ B=(uint16_t*)((char*)B-boff); boff=0; CK(hipFree(B)); } if(q=='X') di=0; if(rc){ CK(hipFree(C)); CK(hipFree(C2)); }
      void* dummy=dal(dm[s%8]<<20);
      if(s&1){ if(rb) B=(uint16_t*)dal(nb*2+PAD); if(ra) A=(uint16_t*)dal(na*2+PAD); } else { if(ra) A=(uint16_t*)dal(na*2+PAD); if(rb) B=(uint16_t*)dal(nb*2+PAD); }
      if(rc){ C=(uint16_t*)dal(nc*2); C2=(uint16_t*)dal(nc*2); }
      CK(hipFree(dummy));
      if(ra) CK(hipMemcpy(A,ha.data(),na*2,hipMemcpyHostToDevice)); if(rb) CK(hipMemcpy(B,hb.data(),nb*2,hipMemcpyHostToDevice)); if(rc){ CK(hipMemset(C,0,nc*2)); CK(hipMemset(C2,0,nc*2)); } }
    if(q=='a'||q=='b'||q=='c'){ CK(hipStreamSynchronize(st)); size_t dl=DELTA[di%DELTA.size()]; di++; size_t na_=q=='b'?0:dl,nb_=q=='a'?0:dl; A=(uint16_t*)((char*)A-aoff+na_); B=(uint16_t*)((char*)B-boff+nb_); aoff=na_; boff=nb_; CK(hipMemcpy(A,ha.data(),na*2,hipMemcpyHostToDevice)); CK(hipMemcpy(B,hb.data(),nb*2,hipMemcpyHostToDevice)); }
    printf("{\"aoff\":%zu,\"boff\":%zu,\"step\":%zu,\"q\":\"%c\",\"A\":\"%p\",\"B\":\"%p\",\"C\":\"%p\",\"C2\":\"%p\"}\n",aoff,boff,s,q,(void*)A,(void*)B,(void*)C,(void*)C2);
    for(int ii=0;ii<n;ii++){ Arm&a=arms[(ii+s)%n];
      for(int i=0;i<warm;i++) launch(a); CK(hipStreamSynchronize(st));
      double e0=nowe(); auto t0=std::chrono::steady_clock::now(); for(int i=0;i<it;i++) launch(a); CK(hipStreamSynchronize(st));
      double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/it;
      printf("{\"impl\":\"%s\",\"step\":%zu,\"q\":\"%c\",\"pos\":%d,\"ms\":%.4f,\"TF\":%.2f,\"t0\":%.3f,\"t1\":%.3f}\n",a.name.c_str(),s,q,ii,ms,fl/ms*1e-9,e0,nowe());fflush(stdout); } }
  // output of every arm once: FNV, elements different from hb
  std::vector<uint16_t> h1(nc),h2(nc); for(auto&a:arms) if(a.name=="hb"){ launch(a); CK(hipStreamSynchronize(st)); CK(hipMemcpy(h2.data(),C2,nc*2,hipMemcpyDeviceToHost)); }
  for(auto&a:arms) if(a.name!="hb"){ CK(hipMemset(C,0xFF,nc*2)); launch(a); CK(hipStreamSynchronize(st)); CK(hipMemcpy(h1.data(),C,nc*2,hipMemcpyDeviceToHost)); size_t bits=0; uint64_t cs=1469598103934665603ull; for(size_t i=0;i<nc;i++){ bits+=(h1[i]!=h2[i]); cs=(cs^h1[i])*1099511628211ull; } printf("{\"check\":\"%s\",\"fnv\":\"%016llx\",\"differ_from_hb\":%zu,\"n\":%zu}\n",a.name.c_str(),(unsigned long long)cs,bits,nc);fflush(stdout); }
  printf("{\"done\":1}\n"); return 0; }
