import torch
import triton
import triton.language as tl
from leap.hausdorff.pointset_wave import traverse, make_frontier
from leap.common.raw_geometry_refinement import point_direction, hausdorff_pair_kernel
from leap.common.range_buffers import compact_candidates
from leap.hausdorff.cached_refinement import seed_position


@triton.jit
def fixed_raw(P,OFF,T,TOFF,IDS,COUNTS,OUT,CAP:tl.constexpr,
              SEEDS=None,RAW=None,SEED_K:tl.constexpr=0):
    row=tl.program_id(0);slot=tl.program_id(1)
    count=tl.load(COUNTS+row)
    value=tl.full((),float('inf'),tl.float32)
    if slot<count:
        d=tl.load(IDS+row*CAP+slot)
        cached=False
        if SEED_K:
            at=seed_position(SEEDS,row,d,SEED_K)
            cached=at<SEED_K
            if cached:
                value=tl.load(RAW+row*SEED_K+at)
        if not cached:
            a=point_direction(P,OFF,row,T,TOFF,d)
            b=point_direction(T,TOFF,d,P,OFF,row)
            value=tl.sqrt(tl.maximum(a,b))
    tl.store(OUT+row*CAP+slot,value)


@triton.jit
def merge_top20(PREV,IDS,VALUES,COUNTS,OFF,OUT,RADIUS,
                CAP:tl.constexpr,BLOCK:tl.constexpr,FIXED:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK)
    count=tl.load(COUNTS+row)
    if FIXED:
        base=row*CAP
    else:
        base=tl.load(OFF+row)
    j=i-20;valid=(i>=20)&(j<count)
    d=tl.load(IDS+base+j,valid,other=0)
    v=tl.load(VALUES+base+j,valid,other=0.)
    key=(v.to(tl.int32,bitcast=True).to(tl.int64)<<32)|d.to(tl.int64)
    key=tl.where(valid,key,9187343244130779135)
    prev=tl.load(PREV+row*20+i,i<20,other=0)
    key=tl.where(i<20,prev,key)
    ordered=tl.sort(key,descending=False)
    tl.store(OUT+row*20+i,ordered,i<20)
    twentieth=tl.sum(tl.where(i==19,ordered,0),0)
    value=(twentieth>>32).to(tl.int32).to(tl.float32,bitcast=True)
    tl.store(RADIUS+row,value)


@triton.jit
def prefix_counts(COUNTS,OFF,STATS,N:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B);c=tl.load(COUNTS+i,i<N,other=0)
    tl.store(OFF+i,tl.cumsum(c,0)-c,i<N)
    tl.store(STATS,tl.max(c,0))
    tl.store(STATS+1,tl.sum(c,0))


class DeviceWave:
    def __init__(self,n,device,tail_capacity=2048):
        self.n=n;self.cap=tail_capacity;self.frontier=make_frontier(n,device)
        def empty(shape,dtype):return torch.empty(shape,device=device,dtype=dtype)
        self.first_counts=empty(n,torch.int32);self.first_visits=empty((n,3),torch.int32)
        self.first_ids=empty((n,256),torch.int64);self.first_values=empty((n,256),torch.float32)
        self.prefix=empty((n,20),torch.int64);self.radius=empty(n,torch.float32)
        self.tail_counts=empty(n,torch.int32);self.tail_visits=empty((n,3),torch.int32)
        self.tail_stage=empty((n,tail_capacity),torch.int64)
        self.offsets=empty(n,torch.int64);self.stats=empty(2,torch.int32)
        self.ids=empty(n*tail_capacity,torch.int64);self.rows=empty(n*tail_capacity,torch.int64)
        self.values=empty(n*tail_capacity,torch.float32)
        self.output=empty((n,20),torch.int64);self.final_radius=empty(n,torch.float32)
        self.compiled={}

    def record(self,name,compiled):
        self.compiled[name]=compiled

    def first(self,q,db,tree,r0,eps,seeds,selfids,seedkeys):
        stack,lower,sp,leaves=self.frontier
        self.record('first_filter',traverse[(self.n,)](
            q.coordinates,db.coordinates,tree.members,tree.tree,r0,eps,seeds,selfids,
            stack,lower,sp,self.first_counts,self.first_ids,self.first_counts,
            self.first_counts,self.first_visits,leaves,tree.groups,FIRST=True,CAP=256,
            num_warps=4,enable_fp_fusion=False))
        self.record('first_raw',fixed_raw[(self.n,256)](
            q.points,q.offsets,db.points,db.offsets,self.first_ids,self.first_counts,self.first_values,
            256,num_warps=4,enable_fp_fusion=False))
        self.record('first_merge',merge_top20[(self.n,)](
            seedkeys,self.first_ids,self.first_values,self.first_counts,self.offsets,
            self.prefix,self.radius,256,512,True,num_warps=4,enable_fp_fusion=False))
        return self.prefix,self.radius

    def tail_filter(self,q,db,tree,radius,eps,seeds,selfids):
        stack,lower,sp,leaves=self.frontier
        args=(q.coordinates,db.coordinates,tree.members,tree.tree,radius,eps,seeds,selfids,stack,lower,sp)
        self.record('tail_filter',traverse[(self.n,)](
            *args,self.offsets,self.tail_stage,self.tail_counts,self.tail_counts,self.tail_visits,
            leaves,tree.groups,FIRST=False,CAP=self.cap,num_warps=4,enable_fp_fusion=False))
        prefix_counts[(1,)](self.tail_counts,self.offsets,self.stats,self.n,
                             triton.next_power_of_2(self.n),num_warps=4)
        maximum,total=map(int,self.stats.tolist())
        if self.n*max(maximum,1)>8_000_000:raise MemoryError('Tail exceeds original bounded policy')
        overflow=maximum>self.cap
        if overflow:
            ids=torch.empty(total,device=self.ids.device,dtype=torch.int64)
            rows=torch.empty_like(ids);written=torch.empty_like(self.tail_counts)
            traverse[(self.n,)](*args,self.offsets,ids,rows,written,self.tail_visits,leaves,
                tree.groups,FIRST=False,WRITE=True,num_warps=4,enable_fp_fusion=False)
            if not torch.equal(written,self.tail_counts):raise RuntimeError('Overflow count changed')
            values=torch.empty(total,device=self.ids.device)
        else:
            ids=self.ids[:total];rows=self.rows[:total];values=self.values[:total]
            if total:
                compact_candidates[(self.n,triton.cdiv(maximum,128))](
                    self.tail_stage,self.tail_counts,self.offsets,ids,rows,self.cap,128)
        return dict(ids=ids,rows=rows,values=values,counts=self.tail_counts,
                    visits=self.tail_visits,offsets=self.offsets,maximum=maximum,overflow=int(overflow))

    def tail_raw(self,q,db,tail):
        if len(tail['ids']):
            self.record('tail_raw',hausdorff_pair_kernel[(len(tail['ids']),)](
                q.points,q.offsets,db.points,db.offsets,tail['rows'],tail['ids'],tail['values'],
                True,num_warps=4,enable_fp_fusion=False))

    def finish(self,tail):
        self.record('tail_merge',merge_top20[(self.n,)](
            self.prefix,tail['ids'],tail['values'],tail['counts'],tail['offsets'],
            self.output,self.final_radius,0,triton.next_power_of_2(tail['maximum']+20),False,
            num_warps=4,enable_fp_fusion=False))
        return self.output

    def buffer_bytes(self):
        tensors=[v for v in vars(self).values() if isinstance(v,torch.Tensor)]+list(self.frontier)
        return sum(v.numel()*v.element_size() for v in tensors)
