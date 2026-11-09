// N3: time paged decode k1 and k2 apart, arms rotated inside one process. usage: n3 <co> <arms.tsv> B L rounds iters seq(0/1)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstring>
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("HIPERR %s %s\n",#x,hipGetErrorString(e_));fflush(stdout);exit(1);} }while(0)
static hipStream_t st; static hipModule_t mod;
struct Arm{std::string n,cl; int S,th; size_t dyn; int ck,sw; hipFunction_t f,k2;};
static uint16_t f2h(float x){ __fp16 h=(__fp16)x; uint16_t u; memcpy(&u,&h,2); return u; }
template<class T> static T* up(const std::vector<T>&h){ T*p; CK(hipMalloc(&p,h.size()*sizeof(T))); CK(hipMemcpy(p,h.data(),h.size()*sizeof(T),hipMemcpyHostToDevice)); return p; }
static double now(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static uint64_t fnv(const void*p,size_t n){ uint64_t c=1469598103934665603ull; const uint8_t*b=(const uint8_t*)p; for(size_t i=0;i<n;i++) c=(c^b[i])*1099511628211ull; return c; }
int main(int ac,char**av){
  const char* co=av[1]; std::string armf=av[2]; int B=atoi(av[3]),L=atoi(av[4]),rounds=atoi(av[5]),iters=atoi(av[6]),seq=atoi(av[7]);
  int Hkv=4,G=6,D=256,PAGE=16,H=Hkv*G; int MAXP=L/PAGE; size_t NBLK=size_t(B)*MAXP;
  std::ifstream in(co,std::ios::binary); std::vector<char> cod((std::istreambuf_iterator<char>(in)),{}); CK(hipModuleLoadData(&mod,cod.data())); CK(hipStreamCreate(&st));
  std::vector<Arm> arms; { std::ifstream a(armf); std::string l; int Smax=0; while(std::getline(a,l)){ std::istringstream s(l); Arm x; s>>x.n>>x.S>>x.th>>x.dyn>>x.ck>>x.cl>>x.sw; CK(hipModuleGetFunction(&x.f,mod,x.n.c_str())); CK(hipModuleGetFunction(&x.k2,mod,("k2_"+std::to_string(x.S)).c_str())); arms.push_back(x);} }
  int Smax=0; for(auto&a:arms) Smax=std::max(Smax,a.S);
  std::mt19937_64 g(12345); std::normal_distribution<float> nd; std::vector<uint16_t> lut(1<<16); for(auto&v:lut) v=f2h(nd(g));
  auto rnd16=[&](size_t n){ std::vector<uint16_t> h(n); uint64_t r=g(); for(size_t i=0;i<n;i+=4){ r^=r<<13;r^=r>>7;r^=r<<17; for(int j=0;j<4&&i+j<n;j++) h[i+j]=lut[(r>>(16*j))&0xffff]; } return h; };
  auto Qh=rnd16(size_t(B)*H*D); auto Kh=rnd16(NBLK*PAGE*Hkv*D); auto Vh=rnd16(NBLK*PAGE*Hkv*D);
  std::vector<int> bt(NBLK), lens(B,L); for(size_t i=0;i<NBLK;i++) bt[i]=int(i); if(!seq) std::shuffle(bt.begin(),bt.end(),g);
  size_t nb=0; for(int b=0;b<B;b++){ if(b&1) lens[b]=L*3/4; nb+=size_t(2)*2*Hkv*D*lens[b]; }
  uint16_t*Q=up(Qh),*K=up(Kh),*V=up(Vh); int*BT=up(bt),*Lens=up(lens); float*Op,*Lse; uint16_t*O; 
  CK(hipMalloc(&Op,size_t(B)*H*Smax*D*4)); CK(hipMalloc(&Lse,size_t(B)*H*Smax*4)); CK(hipMalloc(&O,size_t(B)*H*D*2));
  size_t ntr=size_t(B)*Hkv*Smax*5; unsigned long long* Tr; CK(hipMalloc(&Tr,ntr*8)); std::vector<unsigned long long> trh(ntr);
  printf("{\"m\":\"setup\",\"B\":%d,\"L\":%d,\"bytes\":%zu,\"seq\":%d,\"arms\":%zu}\n",B,L,nb,seq,arms.size()); fflush(stdout);
  void* a1[8]={&BT,&K,&Lens,&Lse,&Op,&Q,&V,&Tr}; void* a2[3]={&Lse,&O,&Op};
  auto k1=[&](Arm&a){ CK(hipModuleLaunchKernel(a.f,a.sw?B*Hkv:a.S,a.sw?a.S:B*Hkv,1,a.th,1,1,a.dyn,st,a1,nullptr)); };
  auto k2=[&](Arm&a){ CK(hipModuleLaunchKernel(a.k2,B*H,1,1,256,1,1,0,st,a2,nullptr)); };
  for(int r=0;r<rounds;r++){ int na=arms.size();
    for(int ai=0;ai<na;ai++){ Arm&a=arms[(ai+r)%na];
      for(int i=0;i<3;i++) k1(a); CK(hipStreamSynchronize(st));
      double t0=now(); for(int i=0;i<iters;i++) k1(a); CK(hipStreamSynchronize(st)); double m1=(now()-t0)/iters;
      double m2=0,m12=0,cs=0;
      char buf[64]="-";
      if(a.cl=="full"){ for(int i=0;i<3;i++) k2(a); CK(hipStreamSynchronize(st));
        t0=now(); for(int i=0;i<iters;i++) k2(a); CK(hipStreamSynchronize(st)); m2=(now()-t0)/iters;
        t0=now(); for(int i=0;i<iters;i++){k1(a);k2(a);} CK(hipStreamSynchronize(st)); m12=(now()-t0)/iters;
        if(r==0){ std::vector<uint16_t> oh(size_t(B)*H*D); CK(hipMemcpy(oh.data(),O,oh.size()*2,hipMemcpyDeviceToHost)); snprintf(buf,64,"%016llx",(unsigned long long)fnv(oh.data(),oh.size()*2)); }
      }
      double tot=m12>0?m12:m1;
      printf("{\"m\":\"time\",\"round\":%d,\"pos\":%d,\"k\":\"%s\",\"S\":%d,\"th\":%d,\"dyn\":%zu,\"k1\":%.4f,\"k2\":%.4f,\"both\":%.4f,\"GBps\":%.1f,\"pct\":%.1f,\"cs\":\"%s\"}\n",r,ai,a.n.c_str(),a.S,a.th,a.dyn,m1,m2,m12,nb/tot*1e-6,nb/tot*1e-6/16,buf); fflush(stdout);
      if(a.ck){ for(int i=0;i<4;i++) k1(a); CK(hipStreamSynchronize(st)); CK(hipMemcpy(trh.data(),Tr,size_t(a.S)*B*Hkv*5*8,hipMemcpyDeviceToHost));
        int nbk=a.S*B*Hkv; std::vector<double> mz,cy; unsigned long long rmin=~0ull,rmax=0; for(int i=0;i<nbk;i++){ auto*t=&trh[i*5]; double dm=t[3]-t[2],dr=t[1]-t[0]; if(dr>0){mz.push_back(dm/dr*25.0);cy.push_back(dm);} rmin=std::min(rmin,t[0]); rmax=std::max(rmax,t[1]); }
        std::sort(mz.begin(),mz.end()); std::sort(cy.begin(),cy.end());
        // residency: events over per-CU spans, hw id bits as E1 (se,sh,cu)
        std::vector<std::pair<unsigned long long,int>> ev; std::vector<int> cur(4096,0),pk(4096,0); std::vector<std::vector<std::pair<unsigned long long,int>>> pc(4096);
        for(int i=0;i<nbk;i++){ auto*t=&trh[i*5]; if(t[1]<=t[0]) continue; unsigned hw=t[4]; int cu=((hw>>8)&15)|(((hw>>12)&1)<<4)|(((hw>>13)&3)<<5); pc[cu].push_back({t[0],1}); pc[cu].push_back({t[1],-1}); }
        int used=0; double sumpk=0; std::vector<double> res; for(auto&v:pc){ if(v.empty())continue; used++; std::sort(v.begin(),v.end()); int c=0,p=0; for(auto&e:v){c+=e.second;p=std::max(p,c);} sumpk+=p; }
        printf("{\"m\":\"clk\",\"round\":%d,\"k\":\"%s\",\"S\":%d,\"mhz_p5\":%.1f,\"mhz_med\":%.1f,\"mhz_p95\":%.1f,\"ticks_med\":%.0f,\"span_us\":%.1f,\"cus\":%d,\"peak_blk_cu_mean\":%.2f}\n",r,a.n.c_str(),a.S,mz[mz.size()/20],mz[mz.size()/2],mz[mz.size()*19/20],cy[cy.size()/2],(rmax-rmin)/25.0,used,sumpk/std::max(1,used)); fflush(stdout);
      }
    }
  }
  printf("DONE\n"); return 0; }
