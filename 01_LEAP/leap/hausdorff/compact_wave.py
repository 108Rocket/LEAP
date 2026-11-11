"""Keep the bounded traversal staging buffer; allocate tail payload only once sized."""
import torch
import triton
from leap.hausdorff.pointset_device import DeviceWave,prefix_counts
from leap.hausdorff.pointset_wave import make_frontier,traverse
from leap.common.range_buffers import compact_candidates


class CompactWave(DeviceWave):
    def __init__(self,n,device,tail_capacity=2048):
        self.n=n;self.cap=tail_capacity;self.frontier=make_frontier(n,device)
        def empty(shape,dtype):return torch.empty(shape,device=device,dtype=dtype)
        self.first_counts=empty(n,torch.int32);self.first_visits=empty((n,3),torch.int32)
        self.first_ids=empty((n,256),torch.int64);self.first_values=empty((n,256),torch.float32)
        self.prefix=empty((n,20),torch.int64);self.radius=empty(n,torch.float32)
        self.tail_counts=empty(n,torch.int32);self.tail_visits=empty((n,3),torch.int32)
        self.tail_stage=empty((n,tail_capacity),torch.int64)
        self.offsets=empty(n,torch.int64);self.stats=empty(2,torch.int32)
        self.output=empty((n,20),torch.int64);self.final_radius=empty(n,torch.float32)
        self.compiled={}

    def tail_filter(self,q,db,tree,radius,eps,seeds,selfids):
        stack,lower,sp,leaves=self.frontier
        args=(q.coordinates,db.coordinates,tree.members,tree.tree,radius,eps,seeds,selfids,stack,lower,sp)
        self.record('tail_filter',traverse[(self.n,)](*args,self.offsets,self.tail_stage,
            self.tail_counts,self.tail_counts,self.tail_visits,leaves,tree.groups,
            FIRST=False,CAP=self.cap,num_warps=4,enable_fp_fusion=False))
        prefix_counts[(1,)](self.tail_counts,self.offsets,self.stats,self.n,triton.next_power_of_2(self.n),num_warps=4)
        maximum,total=map(int,self.stats.tolist())
        if self.n*max(maximum,1)>8_000_000:raise MemoryError('Tail exceeds original bounded policy')
        ids=torch.empty(total,device=q.points.device,dtype=torch.int64);rows=torch.empty_like(ids)
        values=torch.empty(total,device=q.points.device)
        overflow=maximum>self.cap
        if overflow:
            written=torch.empty_like(self.tail_counts)
            traverse[(self.n,)](*args,self.offsets,ids,rows,written,self.tail_visits,leaves,
                tree.groups,FIRST=False,WRITE=True,num_warps=4,enable_fp_fusion=False)
            if not torch.equal(written,self.tail_counts):raise RuntimeError('Overflow count changed')
        elif total:
            compact_candidates[(self.n,triton.cdiv(maximum,128))](self.tail_stage,self.tail_counts,
                self.offsets,ids,rows,self.cap,128)
        return dict(ids=ids,rows=rows,values=values,counts=self.tail_counts,visits=self.tail_visits,
                    offsets=self.offsets,maximum=maximum,overflow=int(overflow))
