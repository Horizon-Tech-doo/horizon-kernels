# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU-readable wide task/counter protocol; kernel derives tasks arithmetically.

Legacy descriptor/signal buffers are placeholders and are never read by this
kernel. Lists below describe actual tasks, including both publication counters
when a wide activation tile crosses a DOWN split boundary.
"""
class WidePlan:
    def __init__(self, hidden, intermediate, splits, ncta, block_n, block_n_down=None, down_split_major=False):
        if type(down_split_major) is not bool:
            raise ValueError("DOWN task-order flag must be boolean")
        block_n_down=block_n if block_n_down is None else block_n_down
        sg,sd=splits
        if sg!=1 or block_n not in (128,256) or hidden%block_n or intermediate%block_n or intermediate%(128*sd) or ncta<1 or block_n_down not in (128,256) or hidden%block_n_down:
            raise ValueError('unsupported wide protocol geometry')
        self.ncta=ncta;self.tiles=intermediate//block_n;self.per=intermediate//(128*sd)
        self.n_counters=self.tiles+sd;self.lists=[[] for _ in range(ncta)]
        tasks=[]
        for bx in range(2*self.tiles):
            if bx<self.tiles:waits=[];signals=[bx]
            else:
                tile=bx-self.tiles;waits=[(tile,1)]
                signals=[self.tiles+(tile*(block_n//128)+half)//self.per for half in range(block_n//128)]
            tasks.append(('GU',bx,0,waits,signals))
        pairs=(((bx,bk) for bk in range(sd) for bx in range(hidden//block_n_down))
               if down_split_major else
               ((bx,bk) for bx in range(hidden//block_n_down) for bk in range(sd)))
        for bx,bk in pairs:tasks.append(('DOWN',bx,bk,[(self.tiles+bk,self.per)],[]))
        for i,t in enumerate(tasks):self.lists[i%ncta].append(t)
        self.check()

    def check(self):
        counts=[0]*self.n_counters
        for lst in self.lists:
            for _,_,_,_,signals in lst:
                for s in signals:counts[s]+=1
        assert counts[:self.tiles]==[1]*self.tiles
        assert counts[self.tiles:]==[self.per]*(self.n_counters-self.tiles)
        for lst in self.lists:
            for _,_,_,waits,_ in lst:
                assert all(counts[c]>=target for c,target in waits)

    def stats(self):
        n=[len(x) for x in self.lists]
        return dict(tasks=sum(n),max_per_cta=max(n),min_per_cta=min(n),counters=self.n_counters,
                    descriptor_reads=False,multi_counter_publications=True)

    def tensors(self,device):
        import torch
        zeros=lambda *shape:torch.zeros(shape,device=device,dtype=torch.int32)
        return dict(desc=zeros(self.ncta,1,8),sigtab=zeros(self.ncta,1),
                    ntask=torch.tensor([len(x) for x in self.lists],device=device,dtype=torch.int32),
                    counters=zeros(self.n_counters),done=zeros(1),err=zeros(1),maxt=1)
