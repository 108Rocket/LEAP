"""Single-request fixed buffers with a checked, exact overflow fallback."""
import numpy as np
import torch
import triton
import triton.language as tl

from leap.hausdorff import pointset_index as nf
from leap.hausdorff import pointset_index as fast
from leap.hausdorff.pointset_device import DeviceWave, fixed_raw
from leap.hausdorff.pointset_merge import merge_top20
from leap.hausdorff.pointset_wave import traverse
from leap.common.raw_geometry_refinement import hausdorff_pair_kernel
from leap.common.raw import normalize_points_into


@triton.jit
def prepare_query(Q,SCALE,STATUS,EPS):
    row=tl.program_id(0);i=tl.arange(0,128)
    q=tl.load(Q+row*128+i)
    slack=2e-5*(1+tl.max(tl.abs(q),0)+tl.load(SCALE))
    tl.store(EPS+row,slack)
    tl.store(STATUS+1,tl.max(tl.abs(q),0)<float('inf'))


@triton.jit
def decode_seeds(KEYS,IDS,N:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,64)
    key=tl.load(KEYS+row*50+i,i<50,other=0)
    tl.store(IDS+row*50+i,key&0xffffffff,i<50)


@triton.jit
def seed_top(IDS,VALUES,KEYS,RADIUS):
    row=tl.program_id(0);i=tl.arange(0,64)
    d=tl.load(IDS+row*50+i,i<50,other=0)
    v=tl.load(VALUES+row*50+i,i<50,other=0.)
    k=(v.to(tl.int32,bitcast=True).to(tl.int64)<<32)|d
    k=tl.where(i<50,k,9187343244130779135)
    k=tl.sort(k,descending=False)
    tl.store(KEYS+row*20+i,k,i<20)
    kth=tl.sum(tl.where(i==19,k,0),0)
    tl.store(RADIUS+row,(kth>>32).to(tl.int32).to(tl.float32,bitcast=True))


@triton.jit
def safe_counts(COUNTS,SAFE,MAXIMUM,N:tl.constexpr,CAP:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B);c=tl.load(COUNTS+i,i<N,other=0)
    tl.store(SAFE+i,tl.minimum(c,CAP),i<N)
    tl.store(MAXIMUM,tl.max(c,0))


class BufferedSingle:
    def __init__(self,q,db,tree,*,anchors=None,capacity=2048):
        assert len(q.coordinates)==1
        self.q=q;self.db=db;self.tree=tree;self.anchors=anchors;self.cap=capacity
        device=q.points.device
        self.eps=torch.empty(1,device=device)
        self.qids=torch.full((1,),-1,device=device,dtype=torch.int64)
        self.seed_keys=torch.empty((1,50),device=device,dtype=torch.int64)
        self.seeds=torch.empty_like(self.seed_keys)
        self.visits=torch.empty((1,3),device=device,dtype=torch.int32)
        self.seed_values=torch.empty((1,50),device=device)
        self.rows=torch.zeros(50,device=device,dtype=torch.int64)
        self.initial=torch.empty((1,20),device=device,dtype=torch.int64)
        self.radius=torch.empty(1,device=device)
        self.wave=DeviceWave(1,device,tail_capacity=capacity)
        self.safe=torch.empty(1,device=device,dtype=torch.int32)
        self.maximum=torch.empty(2,device=device,dtype=torch.int32)
        self.graph=None;self.fallbacks=0;self.compiled={}

    def execute(self):
        q,d,t=self.q,self.db,self.tree
        if self.anchors is not None:
            nf.field_kernel[(1,8)](q.points,q.offsets,self.anchors,q.coordinates,True,64,
                                   num_warps=4,enable_fp_fusion=False)
        prepare_query[(1,)](q.coordinates,t.database_scale,self.maximum,self.eps,
                            num_warps=4,enable_fp_fusion=False)
        self.compiled['candidate']=fast.candidate_kernel[(1,)](
            q.coordinates,d.coordinates,t.members,t.tree,self.qids,self.eps,self.seed_keys,self.visits,
            t.groups,64,50,128,num_warps=4,enable_fp_fusion=False)
        decode_seeds[(1,)](self.seed_keys,self.seeds,1,num_warps=4)
        hausdorff_pair_kernel[(50,)](q.points,q.offsets,d.points,d.offsets,self.rows,self.seeds,
            self.seed_values,True,num_warps=4,enable_fp_fusion=False)
        seed_top[(1,)](self.seeds,self.seed_values,self.initial,self.radius,num_warps=4)
        self.wave.first(q,d,t,self.radius,self.eps,self.seeds,self.qids,self.initial)
        w=self.wave;stack,lower,sp,leaves=w.frontier
        self.compiled['tail_filter']=traverse[(1,)](
            q.coordinates,d.coordinates,t.members,t.tree,w.radius,self.eps,self.seeds,self.qids,
            stack,lower,sp,w.offsets,w.tail_stage,w.tail_counts,w.tail_counts,w.tail_visits,
            leaves,t.groups,FIRST=False,CAP=self.cap,num_warps=4,enable_fp_fusion=False)
        safe_counts[(1,)](w.tail_counts,self.safe,self.maximum,1,self.cap,1)
        self.compiled['tail_raw']=fixed_raw[(1,self.cap)](
            q.points,q.offsets,d.points,d.offsets,w.tail_stage,self.safe,w.values,self.cap,
            num_warps=4,enable_fp_fusion=False)
        self.compiled['tail_merge']=merge_top20[(1,)](
            w.prefix,w.tail_stage,w.values,self.safe,w.offsets,w.output,w.final_radius,
            self.cap,triton.next_power_of_2(self.cap+20),True,num_warps=4,enable_fp_fusion=False)

    def warmup(self):
        for _ in range(2):self.execute()
        torch.cuda.synchronize(self.q.points.device)
        stream=torch.cuda.Stream(device=self.q.points.device)
        stream.wait_stream(torch.cuda.current_stream(self.q.points.device))
        with torch.cuda.stream(stream):self.execute()
        torch.cuda.current_stream(self.q.points.device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.execute()
        torch.cuda.synchronize(self.q.points.device)

    def run(self,fallback,graph=True):
        if graph and self.graph is None:
            raise RuntimeError('Call warmup before graph replay')
        if graph:self.graph.replay()
        else:self.execute()
        # No result is exposed before the capacity check. Never truncate.
        maximum,finite=self.maximum.tolist()
        if not finite:
            raise ValueError('Near/far query fields overflowed float32')
        if maximum>self.cap:
            self.fallbacks+=1
            return fallback()
        return self.wave.output

    def bytes(self):
        return self.wave.buffer_bytes()+sum(v.numel()*v.element_size()
            for v in vars(self).values() if isinstance(v,torch.Tensor))


class BufferedSingleFull(BufferedSingle):
    """One-pass Full with the same checked buffers and graph lifecycle."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Full enumerates surviving seeds too, so never insert them twice.
        self.empty_previous=torch.full((1,20),9187343244130779135,
                                        device=self.q.points.device,dtype=torch.int64)

    def execute(self):
        q,d,t=self.q,self.db,self.tree
        if self.anchors is not None:
            nf.field_kernel[(1,8)](q.points,q.offsets,self.anchors,q.coordinates,True,64,
                                   num_warps=4,enable_fp_fusion=False)
        prepare_query[(1,)](q.coordinates,t.database_scale,self.maximum,self.eps,
                            num_warps=4,enable_fp_fusion=False)
        self.compiled['candidate']=nf.candidate_kernel[(1,)](
            q.coordinates,d.coordinates,t.members,t.tree,self.qids,self.eps,self.seed_keys,self.visits,
            t.groups,64,50,128,FAST=getattr(t,'search_engine','original')=='fast',
            num_warps=4,enable_fp_fusion=False)
        decode_seeds[(1,)](self.seed_keys,self.seeds,1,num_warps=4)
        hausdorff_pair_kernel[(50,)](q.points,q.offsets,d.points,d.offsets,self.rows,self.seeds,
            self.seed_values,True,num_warps=4,enable_fp_fusion=False)
        seed_top[(1,)](self.seeds,self.seed_values,self.initial,self.radius,num_warps=4)
        w=self.wave
        self.compiled['filter']=nf.enumerate_kernel[(1,)](
            q.coordinates,d.coordinates,t.members,t.tree,self.radius,self.qids,self.eps,
            w.offsets,w.tail_stage,w.rows,w.tail_counts,w.tail_visits,
            t.groups,64,WRITE=False,STAGE_CAP=self.cap,DIM=128,
            num_warps=4,enable_fp_fusion=False)
        safe_counts[(1,)](w.tail_counts,self.safe,self.maximum,1,self.cap,1)
        self.compiled['tail_raw']=fixed_raw[(1,self.cap)](
            q.points,q.offsets,d.points,d.offsets,w.tail_stage,self.safe,w.values,self.cap,
            SEEDS=self.seeds,RAW=self.seed_values,SEED_K=50,
            num_warps=4,enable_fp_fusion=False)
        self.compiled['tail_merge']=merge_top20[(1,)](
            self.empty_previous,w.tail_stage,w.values,self.safe,w.offsets,w.output,w.final_radius,
            self.cap,triton.next_power_of_2(self.cap+20),True,num_warps=4,enable_fp_fusion=False)


class BufferedSingleRanking:
    """The native ranking kernels on stable buffers, with explicit startup capture."""
    def __init__(self,q,db,tree,*,anchors,top_k=20):
        if len(q.coordinates)!=1 or not 1<=top_k<=min(64,len(db.coordinates)):
            raise ValueError('Buffered ranking requires one query and Top-k<=64')
        self.q=q;self.db=db;self.tree=tree;self.anchors=anchors;self.k=top_k
        device=q.points.device
        self.eps=torch.empty(1,device=device)
        self.status=torch.empty(2,device=device,dtype=torch.int32)
        self.qids=torch.full((1,),-1,device=device,dtype=torch.int64)
        self.keys=torch.empty((1,top_k),device=device,dtype=torch.int64)
        self.visits=torch.empty((1,3),device=device,dtype=torch.int32)
        self.graph=None

    def execute(self):
        q,d,t=self.q,self.db,self.tree
        nf.field_kernel[(1,8)](q.points,q.offsets,self.anchors,q.coordinates,True,64,
                              num_warps=4,enable_fp_fusion=False)
        prepare_query[(1,)](q.coordinates,t.database_scale,self.status,self.eps,
                            num_warps=4,enable_fp_fusion=False)
        nf.candidate_kernel[(1,)](q.coordinates,d.coordinates,t.members,t.tree,
            self.qids,self.eps,self.keys,self.visits,t.groups,t.block,self.k,t.dim,
            FAST=getattr(t,'search_engine','original')=='fast',
            num_warps=4,enable_fp_fusion=False)

    def warmup(self):
        for _ in range(2):self.execute()
        torch.cuda.synchronize(self.q.points.device)
        stream=torch.cuda.Stream(device=self.q.points.device)
        stream.wait_stream(torch.cuda.current_stream(self.q.points.device))
        with torch.cuda.stream(stream):self.execute()
        torch.cuda.current_stream(self.q.points.device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.execute()
        torch.cuda.synchronize(self.q.points.device)

    def run(self,*,graph=True):
        if graph:
            if self.graph is None:raise RuntimeError('Call warmup before graph replay')
            self.graph.replay()
        else:self.execute()
        if not int(self.status[1]):raise ValueError('Near/far query fields overflowed float32')
        return self.keys


class SingleInput:
    def __init__(self,db,asset,capacity=512):
        from leap.common.model import TrajectoryIndex
        self.capacity=capacity
        self.mean=np.asarray(asset['geometry_mean'],dtype=np.float64)
        self.std=np.asarray(asset['geometry_std'],dtype=np.float64)
        self.host=torch.empty((capacity,2),dtype=torch.float32,pin_memory=True)
        self.host_offsets=torch.empty(2,dtype=torch.int64,pin_memory=True)
        device=db.points.device
        self.anchors=torch.tensor(asset['anchors'],device=device,dtype=torch.float32)
        self.q=TrajectoryIndex(torch.empty((1,128),device=device),
            torch.empty((capacity,2),device=device),torch.empty(2,device=device,dtype=torch.int64),db.asset_id)

    def load(self,path):
        path=np.asarray(path,dtype=np.float64)
        if path.ndim!=2 or path.shape[1]!=2 or len(path)<1 or not np.isfinite(path).all():
            raise ValueError('Expected a finite two-dimensional trajectory')
        if len(path)>self.capacity:raise MemoryError('Input exceeds single-request buffer')
        normalize_points_into(path,self.mean,self.std,self.host.numpy())
        if not np.isfinite(self.host.numpy()[:len(path)]).all():
            raise ValueError('Normalized query geometry overflowed float32')
        self.host_offsets[0]=0;self.host_offsets[1]=len(path)
        self.q.points[:len(path)].copy_(self.host[:len(path)],non_blocking=True)
        self.q.offsets.copy_(self.host_offsets,non_blocking=True)

    def encode(self):
        nf.field_kernel[(1,8)](self.q.points,self.q.offsets,self.anchors,self.q.coordinates,True,64,
                               num_warps=4,enable_fp_fusion=False)
