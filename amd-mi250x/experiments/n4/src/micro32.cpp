#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <fstream>
#define CK(x) do{hipError_t e_=(x); if(e_!=hipSuccess){printf("HIPERR %s %s\n",#x,hipGetErrorString(e_));return 1;} }while(0)
int main(){ hipStream_t st; CK(hipStreamCreate(&st)); std::ifstream in("micro32.co",std::ios::binary); std::vector<char> co((std::istreambuf_iterator<char>(in)),{}); hipModule_t m; CK(hipModuleLoadData(&m,co.data())); hipFunction_t f; CK(hipModuleGetFunction(&f,m,"m32"));
 unsigned* O; CK(hipMalloc(&O,64*16*4)); std::vector<unsigned> h(64*16);
 auto run=[&](int al,int ah,int bl,int bh){ CK(hipMemset(O,0,64*16*4)); void* a[5]={&O,&al,&ah,&bl,&bh}; CK(hipModuleLaunchKernel(f,1,1,1,64,1,1,0,st,a,nullptr)); CK(hipStreamSynchronize(st)); CK(hipMemcpy(h.data(),O,64*16*4,hipMemcpyDeviceToHost)); return 0; };
 // rows: which dst cells are hit for each A lane, B all ones
 for(int al=0;al<64;al+=1){ if(run(al,0,-1,0)) return 1; printf("A lane %d half0 hits:",al); int n=0; for(int l=0;l<64;l++) for(int v=0;v<16;v++) if(h[l*16+v]){ if(n<6) printf(" (%d,%d)",l,v); n++; } printf(" total %d\n",n); }
 // col: A all K not possible with one lane; use A=lane 0 half h, B one hot, map K match
 for(int ah=0;ah<4;ah++) for(int bl=0;bl<64;bl++) for(int bh=0;bh<4;bh++){ if(run(0,ah,bl,bh)) return 1; int n=0,fl=-1,fv=-1; for(int l=0;l<64;l++) for(int v=0;v<16;v++) if(h[l*16+v]){ if(n==0){fl=l;fv=v;} n++; } if(n) printf("A(0,%d) B(%d,%d) -> dst (%d,%d) n=%d\n",ah,bl,bh,fl,fv,n); }
 return 0; }
