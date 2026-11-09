// K1 harness, mb style. Usage: k1 micro | k1 lim | k1 gemm <round> <dir> | k1 badva
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <map>
#include <fstream>
#include <cstring>
#include <algorithm>
static hipStream_t st; static std::map<std::string,hipModule_t> mods; static hipModule_t micro;
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("HIPERR %s %s\n",#x,hipGetErrorString(e_));fflush(stdout);exit(1);} }while(0)
static uint64_t r_=88172645463325252ull; static inline uint64_t rn(){r_^=r_<<13;r_^=r_>>7;r_^=r_<<17;return r_;}
static uint16_t* mk16(size_t n,std::vector<uint16_t>* keep=nullptr){ uint16_t* p; CK(hipMalloc(&p,n*2)); std::vector<uint16_t> h(n);
  for(size_t i=0;i<n;i+=4){uint64_t v=rn(); for(int j=0;j<4&&i+j<n;j++){uint16_t m=(v>>(16*j))&0x83ff; h[i+j]=m|(14<<10);} }
  CK(hipMemcpy(p,h.data(),n*2,hipMemcpyHostToDevice)); if(keep)*keep=std::move(h); return p;}
static hipModule_t loadco(const std::string& path){ std::ifstream in(path,std::ios::binary); if(!in){printf("{\"err\":\"noco %s\"}\n",path.c_str());exit(1);} std::vector<char> co((std::istreambuf_iterator<char>(in)),{}); hipModule_t m; CK(hipModuleLoadData(&m,co.data())); return m; }
static hipFunction_t gemmf(const std::string& n){ auto it=mods.find(n); if(it==mods.end()) { std::ifstream t("co/"+n+".co"); mods[n]=loadco(t? "co/"+n+".co" : "co/ref_"+n+".co"); } hipFunction_t f; CK(hipModuleGetFunction(&f,mods[n],n.c_str())); return f; }
static hipFunction_t mf(const char* n){ hipFunction_t f; CK(hipModuleGetFunction(&f,micro,n)); return f; }
static double launch(hipFunction_t f,int gx,int gy,size_t dyn,std::vector<void*> a,int warm,int iters){
  for(int i=0;i<warm;i++) CK(hipModuleLaunchKernel(f,gx,gy,1,256,1,1,dyn,st,a.data(),nullptr));
  CK(hipStreamSynchronize(st));
  auto t0=std::chrono::steady_clock::now();
  for(int i=0;i<iters;i++) CK(hipModuleLaunchKernel(f,gx,gy,1,256,1,1,dyn,st,a.data(),nullptr));
  CK(hipStreamSynchronize(st));
  return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/std::max(iters,1);
}
static inline uint32_t patt(uint32_t i){return i*2654435761u+0x12345u;}
static uint32_t inv32(uint32_t a){uint32_t x=a; for(int i=0;i<5;i++) x*=2-a*x; return x;}
static const uint32_t PINV=inv32(2654435761u);
static inline uint32_t unpatt(uint32_t v){return (v-0x12345u)*PINV;}
#define SENT 0xDEADBEEFu
struct BadCase{const char* name; size_t dyn; uint32_t m0,range,vstart,allw,nl,mask,inst;};
static void badrun(uint32_t* G,const BadCase& c,int blocks){
  uint32_t ldsdw=c.dyn/4; size_t on=size_t(blocks)*ldsdw, mn=size_t(blocks)*256*8;
  uint32_t* O; uint32_t* M; CK(hipMalloc(&O,on*4)); CK(hipMalloc(&M,mn*4));
  std::vector<uint32_t> ho(on,0xABABABABu),hm(mn,0xABABABABu);
  CK(hipMemcpy(O,ho.data(),on*4,hipMemcpyHostToDevice)); CK(hipMemcpy(M,hm.data(),mn*4,hipMemcpyHostToDevice));
  uint32_t a0=ldsdw; std::vector<void*> a={&G,&O,&M,&a0,(void*)&c.m0,(void*)&c.range,(void*)&c.vstart,(void*)&c.allw,(void*)&c.nl,(void*)&c.mask,(void*)&c.inst};
  launch(mf("m_bad"),blocks,1,c.dyn,a,1,0);
  CK(hipMemcpy(ho.data(),O,on*4,hipMemcpyDeviceToHost)); CK(hipMemcpy(hm.data(),M,mn*4,hipMemcpyDeviceToHost));
  // block 0 detail
  printf("{\"t\":\"bad\",\"case\":\"%s\",\"blocks\":%d,\"dyn\":%zu,\"m0\":%u,\"range\":%u,\"vstart\":%u,\"allw\":%u,\"nl\":%u,\"evenlanes\":%u,\"inst\":%u,",c.name,blocks,c.dyn,c.m0,c.range,c.vstart,c.allw,c.nl,c.mask,c.inst);
  // lanes
  bool lane_ok=true; for(int t=0;t<256;t++) if(hm[t*8]!=(uint32_t)(t&63)) lane_ok=false;
  printf("\"lane_eq_t63\":%s,",lane_ok?"true":"false");
  uint32_t poison=0; for(int t=0;t<256;t++) if(hm[t*8+1]==0xABABABABu) poison++;
  printf("\"meta_unwritten_threads\":%u,",poison);
  for(int w=0;w<4;w++){ auto*m=&hm[(w*64)*8]; printf("\"w%d\":{\"e0\":\"%08x%08x\",\"e1\":\"%08x%08x\",\"e2\":\"%08x%08x\",\"trapsts\":\"%08x\"},",w,m[2],m[1],m[4],m[3],m[6],m[5],m[7]); }
  // non sentinel run list, block 0: (lds dword index, source word) as runs with stride 1
  printf("\"runs\":["); long i=0; int nr=0; uint64_t nonsent=0,unwritten=0;
  while(i<(long)ldsdw){ uint32_t v=ho[i]; if(v==0xABABABABu){unwritten++;i++;continue;} if(v==SENT){i++;continue;}
    long j=i+1; uint32_t s=unpatt(v); bool isp=(s<(1u<<28)); while(j<(long)ldsdw&&ho[j]!=SENT&&ho[j]!=0xABABABABu){ uint32_t sj=unpatt(ho[j]); if(isp? (sj!=s+(j-i)) : (ho[j]!=v)) break; j++; }
    nonsent+=j-i; if(nr<80){ if(isp) printf("%s[%ld,%ld,\"src\",%u]",nr?",":"",i,j-i,s); else printf("%s[%ld,%ld,\"val\",\"%08x\"]",nr?",":"",i,j-i,v);} nr++; i=j; }
  printf("],\"nruns\":%d,\"nonsent\":%llu,\"unwritten\":%llu",nr,(unsigned long long)nonsent,(unsigned long long)unwritten);
  if(blocks>1){ uint64_t mn_=~0ull,mx=0,tot=0; for(int b=0;b<blocks;b++){ uint64_t ns=0; for(uint32_t d=0;d<ldsdw;d++) if(ho[size_t(b)*ldsdw+d]!=SENT) ns++; mn_=std::min(mn_,ns);mx=std::max(mx,ns);tot+=ns;} printf(",\"perblock_nonsent_min\":%llu,\"max\":%llu",(unsigned long long)mn_,(unsigned long long)mx);}
  printf("}\n"); fflush(stdout); CK(hipFree(O)); CK(hipFree(M));
}
static int do_micro(){
  micro=loadco("micro.co"); CK(hipStreamCreate(&st));
  size_t gw=(512ull<<20)/4; uint32_t* G; CK(hipMalloc(&G,gw*4));
  { std::vector<uint32_t> h(gw); for(size_t i=0;i<gw;i++) h[i]=patt((uint32_t)i); CK(hipMemcpy(G,h.data(),gw*4,hipMemcpyHostToDevice)); }
  const uint32_t U=0xffffffffu;
  std::vector<BadCase> cs={
    {"map_all_waves",4096,0,U,0,1,2,0,0},{"map_m0_1024",8192,1024,U,0,1,2,0,0},{"map_vstart_4096",4096,0,U,4096,1,2,0,0},
    {"map_inst_offset_64",4096,0,U,0,1,1,0,1},{"map_even_lanes",4096,0,U,0,1,1,1,0},
    {"m0_unaligned_2",4096,2,U,0,0,1,0,0},{"m0_unaligned_1",4096,1,U,0,0,1,0,0},
    {"oob_top_half",4096,3968,U,0,0,1,0,0},{"oob_last_lane_only",4096,3844,U,0,0,1,0,0},{"oob_m0_L_minus_4",4096,4092,U,0,0,1,0,0},{"oob_m0_eq_L",4096,4096,U,0,0,1,0,0},
    {"oob_m0_far",4096,32768,U,0,0,1,0,0},{"oob_m0_ffc0",4096,0xFFC0,U,0,0,1,0,0},{"oob_m0_bit16",4096,0x10000,U,0,0,1,0,0},{"oob_m0_bit16_plus256",4096,0x10100,U,0,0,1,0,0},
    {"range_top_half",4096,0,128,0,0,1,0,0},{"range_zero",4096,0,0,0,0,1,0,0},{"range_straddle",4096,0,130,0,0,1,0,0},
    {"allwave_oob_cross",4096,4096-512,U,0,1,1,0,0}};
  for(auto&c:cs) badrun(G,c,1);
  for(auto&c:cs) if(!strncmp(c.name,"oob",3)||!strncmp(c.name,"allwave",7)) badrun(G,c,1024);
  // hazard
  uint32_t* E; size_t en=1024*256*5; CK(hipMalloc(&E,en*4)); uint32_t* S=E;
  for(int rep=0;rep<3;rep++) for(int m=0;m<5;m++){ char nm[16]; sprintf(nm,"m_haz%d",m);
    int ntr=500; uint32_t gwu=gw; std::vector<uint32_t> z(en,0); CK(hipMemcpy(E,z.data(),en*4,hipMemcpyHostToDevice));
    std::vector<void*> a={&G,&E,&S,&ntr,&gwu}; double ms=launch(mf(nm),1024,1,16384,a,0,1);
    CK(hipMemcpy(z.data(),E,en*4,hipMemcpyDeviceToHost)); uint64_t er=0,se=0,sl=0; for(size_t i=0;i<1024*256;i++){er+=z[i*3];se+=z[i*3+1];sl+=z[i*3+2];}
    printf("{\"t\":\"haz\",\"mode\":%d,\"rep\":%d,\"blocks\":1024,\"trials\":%d,\"dwords_checked\":%llu,\"errors\":%llu,\"sentinel\":%llu,\"stale\":%llu,\"ms\":%.3f}\n",m,rep,ntr,1024ull*256*8*ntr,(unsigned long long)er,(unsigned long long)se,(unsigned long long)sl,ms); fflush(stdout); }
  // hazard with one block too, to vary memory pressure
  for(int m=0;m<5;m++){ char nm[16]; sprintf(nm,"m_haz%d",m); int ntr=2000; uint32_t gwu=gw; std::vector<uint32_t> z(en,0); CK(hipMemcpy(E,z.data(),en*4,hipMemcpyHostToDevice));
    std::vector<void*> a={&G,&E,&S,&ntr,&gwu}; launch(mf(nm),1,1,16384,a,0,1); CK(hipMemcpy(z.data(),E,en*4,hipMemcpyDeviceToHost)); uint64_t er=0,se=0,sl=0; for(int i=0;i<256;i++){er+=z[i*3];se+=z[i*3+1];sl+=z[i*3+2];}
    printf("{\"t\":\"haz\",\"mode\":%d,\"rep\":0,\"blocks\":1,\"trials\":%d,\"dwords_checked\":%llu,\"errors\":%llu,\"sentinel\":%llu,\"stale\":%llu}\n",m,ntr,256ull*8*ntr,(unsigned long long)er,(unsigned long long)se,(unsigned long long)sl); fflush(stdout); }
  // vmcnt order
  for(int rep=0;rep<3;rep++) for(const char* n:{"m_ord8","m_ord15","m_ord0","m_ordh"}) for(int blocks:{1024,1}){
    int ntr=blocks==1?2000:200; uint32_t gwu=gw, hot=0; std::vector<uint32_t> z(en,0); CK(hipMemcpy(E,z.data(),en*4,hipMemcpyHostToDevice));
    std::vector<void*> a={&G,&E,&ntr,&gwu,&hot}; launch(mf(n),blocks,1,16384,a,0,1);
    CK(hipMemcpy(z.data(),E,en*4,hipMemcpyDeviceToHost)); uint64_t er=0,se=0,ci=0,cw=0,cz=0; for(int b=0;b<blocks;b++) for(int t=0;t<256;t++){ auto*p=&z[(size_t(b)*256+t)*5]; er+=p[0];se+=p[1]; if(t%64==0){ci+=p[2];cw+=p[3];cz+=p[4];} }
    double nw=blocks*4.0*ntr; printf("{\"t\":\"ord\",\"k\":\"%s\",\"rep\":%d,\"blocks\":%d,\"trials\":%d,\"lane_checks\":%llu,\"errors\":%llu,\"sentinel\":%llu,\"memtime_issue16\":%.1f,\"memtime_wait\":%.1f,\"memtime_rest\":%.1f}\n",n,rep,blocks,ntr,(unsigned long long)(blocks*256ull*8*ntr),(unsigned long long)er,(unsigned long long)se,ci/nw,cw/nw,cz/nw); fflush(stdout); }
  // latency
  for(int rep=0;rep<3;rep++) for(int cold:{0,1}) for(int blocks:{1,1024}){ int ntr=blocks==1?2000:200; uint32_t gwu=gw,c=cold; std::vector<uint32_t> z(en,0); CK(hipMemcpy(E,z.data(),en*4,hipMemcpyHostToDevice));
    std::vector<void*> a={&G,&E,&ntr,&gwu,&c}; launch(mf("m_lat"),blocks,1,16384,a,0,1); launch(mf("m_lat"),blocks,1,16384,a,0,1);
    CK(hipMemcpy(z.data(),E,en*4,hipMemcpyDeviceToHost)); double s[4]={0,0,0,0}; for(int b=0;b<blocks;b++) for(int t=0;t<256;t+=64){ auto*p=&z[(size_t(b)*256+t)*4]; for(int i=0;i<4;i++) s[i]+=p[i]; }
    double nw=blocks*4.0*ntr; printf("{\"t\":\"lat\",\"cold\":%d,\"rep\":%d,\"blocks\":%d,\"trials\":%d,\"dir8_issue\":%.1f,\"dir8_wait\":%.1f,\"x4x2_load_and_wait\":%.1f,\"x4x2_ds_write\":%.1f}\n",cold,rep,blocks,ntr,s[0]/nw,s[1]/nw,s[2]/nw,s[3]/nw); fflush(stdout);}
  // clock
  { uint16_t* A=mk16(4096*4096),*B=mk16(4096*4096);
    for(int rep=0;rep<3;rep++){ int nk=4096; std::vector<uint32_t> z(en,0); std::vector<void*> a={&A,&B,&E,&nk}; double ms=launch(mf("m_clk"),1024,1,0,a,0,1); CK(hipMemcpy(z.data(),E,en*4,hipMemcpyDeviceToHost)); double mt=0,rt=0; for(int b=0;b<1024;b++){ auto*p=&z[(size_t(b)*256)*3]; mt+=p[0]; rt+=p[1]; }
      printf("{\"t\":\"clk\",\"rep\":%d,\"wall_ms\":%.3f,\"mean_memtime_per_block\":%.0f,\"mean_memrealtime_per_block\":%.0f,\"ratio\":%.4f}\n",rep,ms,mt/1024,rt/1024,mt/rt); fflush(stdout);} }
  return 0;
}
static int do_micro2(){
  micro=loadco("micro.co"); CK(hipStreamCreate(&st));
  size_t gw=(512ull<<20)/4; uint32_t* G; CK(hipMalloc(&G,gw*4));
  { std::vector<uint32_t> h(gw); for(size_t i=0;i<gw;i++) h[i]=patt((uint32_t)i); CK(hipMemcpy(G,h.data(),gw*4,hipMemcpyHostToDevice)); }
  uint32_t* E; size_t en=1024*256*16; CK(hipMalloc(&E,en*4)); uint32_t* S=E;
  for(int rep=0;rep<3;rep++) for(int m:{0,1,2,3,4,5,7,8,9,10,6}) for(int blocks:{1024,1}){ char nm[16]; sprintf(nm,"m_haz%d",m);
    int ntr=blocks==1?2000:500; uint32_t gwu=gw; std::vector<uint32_t> z(en,0); CK(hipMemcpy(E,z.data(),en*4,hipMemcpyHostToDevice));
    std::vector<void*> a={&G,&E,&S,&ntr,&gwu}; launch(mf(nm),blocks,1,16384,a,0,1);
    CK(hipMemcpy(z.data(),E,en*4,hipMemcpyDeviceToHost)); uint64_t er=0,se=0; for(size_t i=0;i<size_t(blocks)*256;i++){er+=z[i*3];se+=z[i*3+1];}
    printf("{\"t\":\"haz\",\"mode\":%d,\"rep\":%d,\"blocks\":%d,\"trials\":%d,\"dwords_checked\":%llu,\"errors\":%llu,\"sentinel\":%llu}\n",m,rep,blocks,ntr,blocks*256ull*8*ntr,(unsigned long long)er,(unsigned long long)se); fflush(stdout); }
  uint32_t seed=1;
  for(int rep=0;rep<5;rep++) for(const char* n:{"m_ordf16_0","m_ordf15_0","m_ordf12_0","m_ordf8_0","m_ordf4_0","m_ordf0_0","m_ordf16_1","m_ordf8_1","m_ordf0_1"}) for(int blocks:{1,1024}){
    int ntr=blocks==1?2000:100; uint32_t gwu=gw, hot=0; seed++; std::vector<uint32_t> z(en,0); CK(hipMemcpy(E,z.data(),en*4,hipMemcpyHostToDevice));
    std::vector<void*> a={&G,&E,&ntr,&gwu,&hot,&seed}; launch(mf(n),blocks,1,16384,a,0,1);
    CK(hipMemcpy(z.data(),E,en*4,hipMemcpyDeviceToHost)); uint64_t c1[8]={0},c2[8]={0}; for(int b=0;b<blocks;b++) for(int t=0;t<256;t++){ auto*p=&z[(size_t(b)*256+t)*16]; for(int i=0;i<8;i++){c1[i]+=p[i];c2[i]+=p[8+i];} }
    printf("{\"t\":\"ordf\",\"k\":\"%s\",\"rep\":%d,\"blocks\":%d,\"trials\":%d,\"lanes_per_line\":%llu,\"chunk1\":[",n,rep,blocks,ntr,blocks*256ull*ntr); for(int i=0;i<8;i++) printf("%s%llu",i?",":"",(unsigned long long)c1[i]); printf("],\"chunk2\":["); for(int i=0;i<8;i++) printf("%s%llu",i?",":"",(unsigned long long)c2[i]); printf("]}\n"); fflush(stdout); }
  // tick rate: one block, so wall time is one workgroup plus dispatch overhead
  { uint16_t* A=mk16(4096*4096),*B=mk16(4096*4096);
    for(int rep=0;rep<5;rep++) for(int nk:{4096,65536}){ std::vector<uint32_t> z(en,0); std::vector<void*> a={&A,&B,&E,&nk}; double ms=launch(mf("m_clk"),1,1,0,a,0,1); CK(hipMemcpy(z.data(),E,en*4,hipMemcpyDeviceToHost)); auto*p=&z[0];
      if(nk==4096){ std::vector<void*> a2={&A,&B,&E,&nk}; for(auto gx_dyn:{std::pair<int,size_t>{1024,0},{1320,0},{440,65024},{1320,65024}}){ double ms2=launch(mf("m_clk"),gx_dyn.first,1,gx_dyn.second,a2,0,1); std::vector<uint32_t> y(size_t(gx_dyn.first)*256*3); CK(hipMemcpy(y.data(),E,y.size()*4,hipMemcpyDeviceToHost)); double mt=0,rt=0; for(int b=0;b<gx_dyn.first;b++){mt+=y[size_t(b)*256*3];rt+=y[size_t(b)*256*3+1];} printf("{\"t\":\"clkn\",\"rep\":%d,\"blocks\":%d,\"dyn\":%zu,\"wall_ms\":%.3f,\"memtime_per_block\":%.0f,\"real_per_block\":%.0f,\"GHz_if_real_25MHz\":%.3f}\n",rep,gx_dyn.first,gx_dyn.second,ms2,mt/gx_dyn.first,rt/gx_dyn.first,mt/rt*0.025); fflush(stdout);} }
      printf("{\"t\":\"clk1\",\"rep\":%d,\"nk\":%d,\"wall_ms\":%.3f,\"memtime\":%u,\"memrealtime\":%u,\"memtime_GHz_from_wall\":%.3f,\"real_MHz_from_wall\":%.2f,\"memtime_per_mfma\":%.2f}\n",rep,nk,ms,p[0],p[1],p[0]/ms*1e-6,p[1]/ms*1e-3,double(p[0])/(nk*64.0)); fflush(stdout);} }
  return 0;
}
static uint64_t fnv(const std::vector<uint16_t>& v){uint64_t cs=1469598103934665603ull; for(auto x:v){cs=(cs^x)*1099511628211ull;} return cs;}
struct Sh{int M,N,K; uint16_t*A,*B,*C;};
static void dumpbin(const std::string& p,const void* d,size_t bytes){ std::ofstream o(p,std::ios::binary); o.write((const char*)d,bytes); }
static void one_ref(const std::string& n,const Sh& s,int bK,size_t dyn,int round){ uint16_t*A=s.A,*B=s.B,*C=s.C; CK(hipMemset(C,0,size_t(s.M)*s.N*2));
  std::vector<void*> a={&A,&B,&C}; double ms=launch(gemmf(n),s.N/128,s.M/128,dyn,a,5,40);
  std::vector<uint16_t> hc(size_t(s.M)*s.N); CK(hipMemcpy(hc.data(),C,hc.size()*2,hipMemcpyDeviceToHost));
  printf("{\"t\":\"g\",\"cs\":\"%016llx\",\"k\":\"%s\",\"M\":%d,\"K\":%d,\"bK\":%d,\"dyn\":%zu,\"round\":%d,\"ms\":%.4f,\"TF\":%.2f}\n",(unsigned long long)fnv(hc),n.c_str(),s.M,s.K,bK,dyn,round,ms,2.0*s.M*s.N*s.K/ms*1e-9); fflush(stdout);}
static void one_g(const std::string& n,const Sh& s,int ch,size_t dyn,int round,bool chk=true,const std::string& dump=""){ uint16_t*A=s.A,*B=s.B,*C=s.C; int K=s.K,N=s.N,nch=s.K/ch,bm=1,km=1; CK(hipMemset(C,0,size_t(s.M)*s.N*2));
  std::vector<void*> a={&A,&B,&C,&K,&N,&nch,&bm,&km}; double ms=launch(gemmf(n),s.N/128,s.M/128,dyn,a,5,40);
  std::vector<uint16_t> hc(size_t(s.M)*s.N); CK(hipMemcpy(hc.data(),C,hc.size()*2,hipMemcpyDeviceToHost)); if(!dump.empty()) dumpbin(dump,hc.data(),hc.size()*2);
  printf("{\"t\":\"g\",\"cs\":\"%016llx\",\"k\":\"%s\",\"M\":%d,\"K\":%d,\"bK\":%d,\"dyn\":%zu,\"round\":%d,\"ms\":%.4f,\"TF\":%.2f}\n",(unsigned long long)fnv(hc),n.c_str(),s.M,s.K,ch,dyn,round,ms,2.0*s.M*s.N*s.K/ms*1e-9); fflush(stdout);}
// part: load only, math only. hot: same tile and K every step (mostly L2/L1 hits), cold: the real addresses.
static void one_p(const std::string& base,const char* part,const Sh& s,int ch,size_t dyn,int hot,int round){ std::string n=base+"_"+part; uint16_t*A=s.A,*B=s.B,*C=s.C; int K=s.K,N=s.N,nch=s.K/ch,bm=hot?0:1,km=hot?0:1;
  std::vector<void*> a={&A,&B,&C,&K,&N,&nch,&bm,&km}; double ms=launch(gemmf(n),s.N/128,s.M/128,dyn,a,5,40);
  double blocks=double(s.N/128)*(s.M/128); double bytes=blocks*nch*2*128*ch*2; double flops=2.0*s.M*s.N*s.K;
  printf("{\"t\":\"p\",\"k\":\"%s\",\"part\":\"%s\",\"hot\":%d,\"M\":%d,\"bK\":%d,\"dyn\":%zu,\"round\":%d,\"ms\":%.4f,\"GBps_if_load\":%.1f,\"TF_if_math\":%.2f}\n",n.c_str(),part,hot,s.M,ch,dyn,round,ms,bytes/ms*1e-6,flops/ms*1e-9); fflush(stdout);}
static int do_gemm(int round,const std::string& dir){
  CK(hipStreamCreate(&st));
  std::vector<uint16_t> ha,hb; size_t n4=4096ull*4096,n8=8192ull*8192; std::vector<uint16_t> h;
  r_=88172645463325252ull;
  uint16_t*A4=mk16(n4,&ha),*B4=mk16(n4,&hb),*C4=mk16(n4); if(round==0){dumpbin(dir+"/A4.bin",ha.data(),n4*2);dumpbin(dir+"/B4.bin",hb.data(),n4*2);}
  uint16_t*A8=mk16(n8,&ha),*B8=mk16(n8,&hb),*C8=mk16(n8); if(round==0){dumpbin(dir+"/A8.bin",ha.data(),n8*2);dumpbin(dir+"/B8.bin",hb.data(),n8*2);}
  std::vector<uint16_t>().swap(ha); std::vector<uint16_t>().swap(hb);
  Sh s4{4096,4096,4096,A4,B4,C4}, s8{8192,8192,8192,A8,B8,C8};
  struct V{std::string n;int ch;size_t dyn;}; 
  std::vector<V> gs={{"g0_32",32,16384},{"g0_32",32,32768},{"g0_32",32,65024},{"g1_32",32,16384},{"g1_32",32,32768},{"g1_32",32,65024},{"g1_32w3",32,32768},
                     {"g2_32",32,32768},{"g2_32",32,65024},{"g4_32",32,32768},{"g4_32",32,65024},{"g4_32w3",32,32768},{"g6_32",32,49152},{"g6_32",32,65024},
                     {"g3_32",32,16384},{"g3_32",32,32768},{"g5_32",32,16384},{"g5_32",32,32768},{"g3_32",32,65024},
                     {"g0_64",64,32768},{"g0_64w3",64,32768},{"g0_64",64,65024},{"g1_64",64,32768},{"g1_64w3",64,32768},{"g1_64",64,65024},{"g3_64",64,32768},{"g3_64w3",64,32768},{"g5_64",64,32768},{"g5_64w3",64,32768},{"g3_64",64,65024},{"g2_64",64,65536},{"g4_64",64,65536},{"g4_64w3",64,65536}};
  std::rotate(gs.begin(),gs.begin()+(round*5)%gs.size(),gs.end());
  for(auto* sp:{&s4,&s8}) {
    if(sp==&s4){ one_ref("b12832",*sp,32,16384,round); one_ref("b12832",*sp,32,32768,round); one_ref("b12864",*sp,64,32768,round); one_ref("rr12832",*sp,32,32768,round); one_ref("rr12864w3",*sp,64,32768,round);} 
    else { one_ref("b12864_8k",*sp,64,32768,round); one_ref("rr12864_8kw3",*sp,64,32768,round);} 
    for(auto&v:gs){ if(v.n=="g2_64"&&getenv("NO65536")) continue; std::string d=(round==0&&v.n=="g0_32"&&v.dyn==32768)?dir+(sp==&s4?"/C4.bin":"/C8.bin"):""; one_g(v.n,*sp,v.ch,v.dyn,round,true,d); }
  }
  // parts at the same dyn as the best full arrangement
  std::vector<V> ps={{"g0_32",32,32768},{"g1_32",32,32768},{"g2_32",32,32768},{"g4_32",32,32768},{"g3_32",32,32768},{"g5_32",32,32768},{"g0_64",64,32768},{"g1_64",64,32768},{"g3_64",64,32768},{"g5_64",64,32768},{"g0_32",32,65024},{"g1_32",32,65024},{"g2_32",32,65024},{"g4_32",32,65024},{"g6_32",32,65024},{"g3_32",32,65024},{"g5_32",32,65024}};
  std::rotate(ps.begin(),ps.begin()+(round*3)%ps.size(),ps.end());
  for(int hot:{0,1}) for(auto&v:ps) for(const char* part:{"l","m"}) one_p(v.n,part,s4,v.ch,v.dyn,hot,round);
  for(auto&v:ps) one_p(v.n,"l",s8,v.ch,v.dyn,0,round);
  printf("DONE round %d\n",round); return 0;
}
static int do_lim(){ // does LDS 65536 dispatch?
  CK(hipStreamCreate(&st)); uint16_t*A=mk16(4096*4096),*B=mk16(4096*4096),*C=mk16(4096*4096); Sh s{4096,4096,4096,A,B,C};
  for(size_t d:{(size_t)65024,(size_t)65280,(size_t)65536}){ printf("{\"t\":\"lim\",\"dyn\":%zu}\n",d); fflush(stdout); uint16_t*Cc=C; int K=4096,N=4096,nch=2,bm=1,km=1; std::vector<void*> a={&A,&B,&Cc,&K,&N,&nch,&bm,&km}; hipFunction_t f=gemmf("g3_32"); hipError_t e=hipModuleLaunchKernel(f,32,32,1,256,1,1,d,st,a.data(),nullptr); hipError_t e2=hipStreamSynchronize(st); printf("{\"t\":\"lim\",\"dyn\":%zu,\"launch\":\"%s\",\"sync\":\"%s\"}\n",d,hipGetErrorString(e),hipGetErrorString(e2)); fflush(stdout);} 
  return 0; }
static int do_badva(){ micro=loadco("micro.co"); CK(hipStreamCreate(&st)); uint32_t* E; CK(hipMalloc(&E,1<<20)); uint32_t* G=(uint32_t*)0x7000000000ull; uint32_t a0=1024,m0=0,r=0xffffffffu,vs=0,al=0,nl=1,ma=0,in=0; uint32_t*O=E,*M=E+(1<<16); std::vector<void*> a={&G,&O,&M,&a0,&m0,&r,&vs,&al,&nl,&ma,&in};
  printf("badva start\n"); fflush(stdout); hipError_t e=hipModuleLaunchKernel(mf("m_bad"),1,1,1,256,1,1,4096,st,a.data(),nullptr); hipError_t e2=hipStreamSynchronize(st); printf("badva launch %s sync %s\n",hipGetErrorString(e),hipGetErrorString(e2)); return 0; }
int main(int c,char** v){ std::string m=c>1?v[1]:""; if(m=="micro")return do_micro(); if(m=="micro2")return do_micro2(); if(m=="lim")return do_lim(); if(m=="gemm")return do_gemm(atoi(v[2]),v[3]); if(m=="badva")return do_badva(); return 2; }
