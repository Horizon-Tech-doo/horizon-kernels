/* Track 5: hand-written adapter, not a compiler or vendor result. RCCL in the vLLM image (ROCm 7.2.3) looks only for
   ncclNetPlugin_v6..v10; LUMI's aws-ofi-rccl 1.4.0 exports only ncclNetPlugin_v5. v6 is v5 plus regMrDmaBuf, with the same
   properties struct (ROCm/rccl ext-net/example/nccl/net_v5.h, net_v6.h). The plugin never advertises NCCL_PTR_DMABUF, so
   regMrDmaBuf stays NULL. Build: gcc -shared -fPIC ofi_v6_adapter.c -o libnccl-net.so -L. -l:librccl-net.so.0 */
#include <stddef.h>
typedef int (*fn)();
typedef struct { const char *name; fn init, devices, getProperties, listen, connect, accept, regMr, deregMr, isend, irecv, iflush, test, closeSend, closeRecv, closeListen; } v5_t;
typedef struct { const char *name; fn init, devices, getProperties, listen, connect, accept, regMr, regMrDmaBuf, deregMr, isend, irecv, iflush, test, closeSend, closeRecv, closeListen; } v6_t;
extern v5_t ncclNetPlugin_v5;
v6_t ncclNetPlugin_v6;
__attribute__((constructor)) static void fill(void) {
  v5_t *a = &ncclNetPlugin_v5; v6_t *b = &ncclNetPlugin_v6;
  b->name = a->name; b->init = a->init; b->devices = a->devices; b->getProperties = a->getProperties; b->listen = a->listen;
  b->connect = a->connect; b->accept = a->accept; b->regMr = a->regMr; b->regMrDmaBuf = NULL; b->deregMr = a->deregMr;
  b->isend = a->isend; b->irecv = a->irecv; b->iflush = a->iflush; b->test = a->test; b->closeSend = a->closeSend;
  b->closeRecv = a->closeRecv; b->closeListen = a->closeListen;
}
