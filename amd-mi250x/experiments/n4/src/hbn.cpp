// N4 harness. One process per shape. Direct hipBLASLt and solo code objects on the same data, arm order rotated per round.
// Usage: hbn M N K data rounds iters warm probe arm...
//   data: randn | fixed       probe: 0 or spin ticks of the clock probe on a second stream (25 MHz ticks)
//   arm : hb | name,co,bM,bN,threads,dyn
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
#include <algorithm>
#include <sstream>
static hipStream_t st,st2;
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("HIPERR %s %s\n",#x,hipGetErrorString(e_));fflush(stdout);exit(1);} }while(0)
#define CB(x) do{hipblasStatus_t e_=(x); if(e_!=HIPBLAS_STATUS_SUCCESS){printf("HBERR %s %d\n",#x,(int)e_);fflush(stdout);exit(1);} }while(0)
static uint64_t r_=88172645463325252ull; static inline uint64_t rn(){r_^=r_<<13;r_^=r_>>7;r_^=r_<<17;return r_;}
static uint16_t f2h(float f){ uint32_t x; memcpy(&x,&f,4); uint32_t s=(x>>16)&0x8000; int e=(int)((x>>23)&255)-127+15; uint32_t m=x&0x7fffff;
  if(e>=31) return s|0x7c00; if(e<=0){ if(e<-10) return s; m|=0x800000; int sh=14-e; uint32_t h=m>>sh; if((m>>(sh-1))&1) h++; return s|h; }
  uint32_t h=(e<<10)|(m>>13); if(m&0x1000) h++; return s|h; }
static uint16_t* mk16(size_t n,bool randn,bool upload=true){ uint16_t* p; CK(hipMalloc(&p,n*2)); if(!upload) { CK(hipMemset(p,0,n*2)); return p; } std::vector<uint16_t> h(n);
  if(randn){ for(size_t i=0;i<n;i+=2){ double u=(rn()>>11)*(1.0/9007199254740992.0),v=(rn()>>11)*(1.0/9007199254740992.0); if(u<1e-300)u=1e-300; double R=sqrt(-2*log(u)),t=6.283185307179586*v; h[i]=f2h((float)(R*cos(t))); if(i+1<n) h[i+1]=f2h((float)(R*sin(t))); } }
  else { for(size_t i=0;i<n;i+=4){uint64_t v=rn(); for(int j=0;j<4&&i+j<n;j++){uint16_t m=(v>>(16*j))&0x83ff; h[i+j]=m|(14<<10);} } }
  CK(hipMemcpy(p,h.data(),n*2,hipMemcpyHostToDevice)); return p;}
static hipModule_t loadco(const std::string&fn){ std::ifstream in(fn,std::ios::binary); if(!in){printf("{\"err\":\"noco %s\"}\n",fn.c_str());fflush(stdout);exit(1);} std::vector<char> co((std::istreambuf_iterator<char>(in)),{}); hipModule_t m; CK(hipModuleLoadData(&m,co.data())); return m; }
static double half2d(uint16_t v){ int s=v>>15,e=(v>>10)&31,m=v&1023; double x=(e==0)?m/1024.0*pow(2,-14):(1+m/1024.0)*pow(2,e-15); return s?-x:x; }
static double nowe(){ return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
struct Arm{std::string name,co; int bM,bN,th; size_t dyn; hipFunction_t f;};
static std::vector<std::string> split(const std::string&s,char c){ std::vector<std::string> o; std::stringstream ss(s); std::string t; while(std::getline(ss,t,c)) o.push_back(t); return o; }
int main(int argc,char**argv){ if(argc<10) return 2; int M=atoi(argv[1]),N=atoi(argv[2]),K=atoi(argv[3]); bool randn=!strcmp(argv[4],"randn"); int rounds=atoi(argv[5]),it=atoi(argv[6]),warm=atoi(argv[7]),spin=atoi(argv[8]);
  CK(hipStreamCreate(&st)); CK(hipStreamCreate(&st2));
  hipblasLtHandle_t hl; CB(hipblasLtCreate(&hl));
  std::vector<Arm> arms; for(int i=9;i<argc;i++){ if(!strcmp(argv[i],"hb")){ Arm a; a.name="hb"; arms.push_back(a); continue;} auto p=split(argv[i],','); Arm a; a.name=p[0];a.co=p[1];a.bM=atoi(p[2].c_str());a.bN=atoi(p[3].c_str());a.th=atoi(p[4].c_str());a.dyn=atol(p[5].c_str()); CK(hipModuleGetFunction(&a.f,loadco(a.co),a.name.c_str())); arms.push_back(a);}    
  hipFunction_t pf=nullptr; uint64_t* PB=nullptr; const int PMAX=4096; if(spin){ CK(hipModuleGetFunction(&pf,loadco("probe.co"),"clkprobe")); CK(hipMalloc(&PB,PMAX*4*8)); }
  uint16_t*A=mk16(size_t(M)*K,randn),*B=mk16(size_t(N)*K,randn),*C=mk16(size_t(M)*N,false,false),*C2=mk16(size_t(M)*N,false,false);
  hipblasLtMatrixLayout_t la,lb,lc; hipblasLtMatmulDesc_t d;
  CB(hipblasLtMatrixLayoutCreate(&la,HIP_R_16F,K,N,K)); CB(hipblasLtMatrixLayoutCreate(&lb,HIP_R_16F,K,M,K)); CB(hipblasLtMatrixLayoutCreate(&lc,HIP_R_16F,N,M,N));
  CB(hipblasLtMatmulDescCreate(&d,HIPBLAS_COMPUTE_32F,HIP_R_32F)); hipblasOperation_t ta=HIPBLAS_OP_T,tb=HIPBLAS_OP_N;
  CB(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta))); CB(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
  hipblasLtMatmulPreference_t pr; CB(hipblasLtMatmulPreferenceCreate(&pr)); size_t maxws=1<<20; CB(hipblasLtMatmulPreferenceSetAttribute(pr,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&maxws,sizeof(maxws)));
  hipblasLtMatmulHeuristicResult_t hr; int got=0; CB(hipblasLtMatmulAlgoGetHeuristic(hl,d,la,lb,lc,lc,pr,1,&hr,&got)); void* ws=nullptr; if(hr.workspaceSize) CK(hipMalloc(&ws,hr.workspaceSize));
  printf("{\"M\":%d,\"N\":%d,\"K\":%d,\"data\":\"%s\",\"heuristic_got\":%d,\"ws\":%zu,\"it\":%d,\"warm\":%d,\"spin\":%d}\n",M,N,K,argv[4],got,(size_t)hr.workspaceSize,it,warm,spin);fflush(stdout);
  float a1=1.f,b0=0.f; double fl=2.0*M*N*K; int n=arms.size();
  auto launch=[&](Arm&a){ if(a.name=="hb") CB(hipblasLtMatmul(hl,d,&a1,B,la,A,lb,&b0,C2,lc,C2,lc,&hr.algo,ws,hr.workspaceSize,st));
    else { void* args[3]={&A,&B,&C}; int gx=(N+a.bN-1)/a.bN,gy=(M+a.bM-1)/a.bM; CK(hipModuleLaunchKernel(a.f,gx,gy,1,a.th,1,1,a.dyn,st,args,nullptr)); } };
  for(int r=0;r<rounds;r++) for(int ii=0;ii<n;ii++){ Arm&a=arms[(ii+r)%n];
    for(int i=0;i<warm;i++) launch(a); CK(hipStreamSynchronize(st));
    double e0=nowe(); auto t0=std::chrono::steady_clock::now(); for(int i=0;i<it;i++) launch(a);
    int np=0; if(spin){ hipError_t q; while((q=hipStreamQuery(st))==hipErrorNotReady && np<PMAX){ int idx=np; void* pa[3]={&PB,&idx,&spin}; CK(hipModuleLaunchKernel(pf,1,1,1,1,1,1,0,st2,pa,nullptr)); CK(hipStreamSynchronize(st2)); np++; } }
    CK(hipStreamSynchronize(st)); double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/it; double e1=nowe();
    printf("{\"impl\":\"%s\",\"round\":%d,\"pos\":%d,\"M\":%d,\"N\":%d,\"K\":%d,\"data\":\"%s\",\"ms\":%.4f,\"TF\":%.2f,\"t0\":%.3f,\"t1\":%.3f",a.name.c_str(),r,ii,M,N,K,argv[4],ms,fl/ms*1e-9,e0,e1);
    if(spin&&np){ CK(hipStreamSynchronize(st2)); std::vector<uint64_t> h(np*4); CK(hipMemcpy(h.data(),PB,np*4*8,hipMemcpyDeviceToHost)); std::vector<double> mhz; for(int i=0;i<np;i++){ double dr=h[i*4+2]-h[i*4+0],dm=h[i*4+3]-h[i*4+1]; if(dr>0) mhz.push_back(dm/dr*25.0);} std::sort(mhz.begin(),mhz.end()); if(!mhz.empty()) printf(",\"np\":%d,\"mhz_med\":%.1f,\"mhz_p10\":%.1f,\"mhz_p90\":%.1f",np,mhz[mhz.size()/2],mhz[mhz.size()/10],mhz[mhz.size()*9/10]); }
    printf("}\n");fflush(stdout); }
  // numeric check once: every arm against hipBLASLt on a fresh C (row-major M x N = column-major N x M)
  bool hasHb=false; for(auto&a:arms) if(a.name=="hb") hasHb=true;
  if(hasHb && !spin){ size_t nn=size_t(M)*N; std::vector<uint16_t> h1(nn),h2(nn); for(auto&a:arms) if(a.name=="hb"){ launch(a); CK(hipStreamSynchronize(st)); }
    CK(hipMemcpy(h2.data(),C2,nn*2,hipMemcpyDeviceToHost));
    for(auto&a:arms) if(a.name!="hb"){ CK(hipMemset(C,0,nn*2)); launch(a); CK(hipStreamSynchronize(st)); CK(hipMemcpy(h1.data(),C,nn*2,hipMemcpyDeviceToHost)); double mx=0,mg=0; size_t bits=0; uint64_t cs=1469598103934665603ull; for(size_t i=0;i<nn;i++){ double x=half2d(h1[i]),y=half2d(h2[i]); mx=std::max(mx,fabs(x-y)); mg=std::max(mg,fabs(y)); bits+=(h1[i]!=h2[i]); cs=(cs^h1[i])*1099511628211ull; }
      printf("{\"check\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"max_abs_diff\":%.6g,\"max_abs\":%.6g,\"rel\":%.3g,\"elements_differ\":%zu,\"n\":%zu,\"fnv\":\"%016llx\"}\n",a.name.c_str(),M,N,K,mx,mg,mx/mg,bits,nn,(unsigned long long)cs);fflush(stdout);} }
  printf("{\"done\":1}\n"); return 0; }
