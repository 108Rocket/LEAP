"""Fixed-capacity single DFD execution, with complete overflow fallback."""
import torch
import triton
import triton.language as tl

from leap.dfd.dfd_refinement import cached_kernel
from leap.dfd.dfd_tree import native_enum
from leap.hausdorff.pointset_merge import merge_top20


@triton.jit
def safe_count(COUNT, SAFE, CAP:tl.constexpr):
    tl.store(SAFE,tl.minimum(tl.load(COUNT),CAP))


@triton.jit
def refine_bucket(P,OFF,T,TOFF,Q,D,COUNT,INITIAL,RAW,RADIUS,O,FLAGS,WORK,
                  CAP:tl.constexpr,BLOCK:tl.constexpr,LOWER:tl.constexpr,S:tl.constexpr):
    pair=tl.program_id(0)
    if pair<tl.minimum(tl.load(COUNT),CAP):
        d=tl.load(D+pair)
        na=tl.load(OFF+1)-tl.load(OFF)
        nb=tl.load(TOFF+d+1)-tl.load(TOFF+d)
        width=tl.maximum(na,nb)
        if (width>LOWER)&(width<=BLOCK):
            cached_kernel(P,OFF,T,TOFF,Q,D,INITIAL,RAW,RADIUS,O,FLAGS,WORK,BLOCK,S,64)


class Storage:
    def __init__(self,plan,capacity,seeds,block):
        self.cap=capacity
        d=plan.device
        self.initial=torch.empty((1,seeds),device=d,dtype=torch.int64)
        self.seed_rows=torch.zeros(seeds,device=d,dtype=torch.int64)
        self.qids=torch.empty(1,device=d,dtype=torch.int64)
        self.rows=torch.zeros(capacity,device=d,dtype=torch.int64)
        self.ids=torch.empty(capacity,device=d,dtype=torch.int64)
        self.counts=torch.empty(1,device=d,dtype=torch.int32)
        self.safe=torch.empty_like(self.counts)
        self.visits=torch.empty((1,2),device=d,dtype=torch.int32)
        self.values=torch.empty(capacity,device=d)
        self.flags=torch.empty(capacity,device=d,dtype=torch.int32)
        self.work=torch.empty((capacity,3,block),device=d)
        self.empty=torch.full((1,20),9187343244130779135,device=d,dtype=torch.int64)
        self.top=torch.empty_like(self.empty)
        self.final_radius=torch.empty(1,device=d)
        self.zero=torch.zeros(1,device=d,dtype=torch.int64)

    def operations(self,plan,q,d,tree,block):
        s=self.initial.shape[1]
        raw=plan.pair_values(q,d,self.seed_rows,self.initial.reshape(-1),0).reshape(1,s)
        radius=raw.topk(20,largest=False).values[:,-1].contiguous()
        slack=(2e-5*(1+q.coordinates.abs().amax(1)+tree.database_scale)).contiguous()
        ep=plan.endpoint(q)
        native_enum[(1,)](q.coordinates,d.coordinates,ep,tree.dfd_geometry['endpoints'],
            tree.members,*tree.dfd_geometry['native'],radius,self.qids,slack,
            self.counts,self.ids,self.rows,self.counts,self.visits,
            tree.groups,tree.block,False,STAGE_CAP=self.cap,num_warps=4,enable_fp_fusion=False)
        threshold=(radius+slack).contiguous()
        lower=0
        for width in plan.widths(block):
            refine_bucket[(self.cap,)](q.points,q.offsets,d.points,d.offsets,self.rows,self.ids,
                self.counts,self.initial,raw,threshold,self.values,self.flags,self.work,
                self.cap,width,lower,s,num_warps=4,enable_fp_fusion=False)
            lower=width
        safe_count[(1,)](self.counts,self.safe,self.cap)
        merge_top20[(1,)](self.empty,self.ids,self.values,self.safe,self.zero,self.top,
            self.final_radius,self.cap,self.cap,True,num_warps=4,enable_fp_fusion=False)
        return radius


@torch.no_grad()
def run(plan,q,d,tree,qids,initial,capacity):
    if capacity<20 or capacity>8192:
        raise ValueError('Static DFD capacity must be in [20,8192]')
    block=max(64,plan.database_block,triton.next_power_of_2(plan.query_length))
    # Keep the original bounded path for exceptionally long inputs.
    if capacity*3*block*4>256*1024**2:
        return None
    key=('dfd_full',capacity,initial.shape[1],block,q.points.data_ptr(),
         q.offsets.data_ptr(),q.coordinates.data_ptr())
    if not hasattr(plan,'dfd_storage'):
        plan.dfd_storage={}
        plan.dfd_fallbacks=0
    storage=plan.dfd_storage.get(key)
    if storage is None:
        # A growing input invalidates captures through the existing session.
        plan.graphs.clear()
        plan.dfd_storage.clear()
        storage=Storage(plan,capacity,initial.shape[1],block)
        plan.dfd_storage[key]=storage
    storage.initial.copy_(initial)
    storage.qids.copy_(qids)
    if key in plan.graphs:
        graph,radius=plan.graphs[key]
        graph.replay()
        plan.stats['graph_replays']+=1
    elif plan.capture_enabled:
        storage.operations(plan,q,d,tree,block)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=torch.cuda.current_stream(plan.device)):
            radius=storage.operations(plan,q,d,tree,block)
        plan.graphs[key]=graph,radius
        plan.stats['graph_builds']+=1
        graph.replay()
    else:
        radius=storage.operations(plan,q,d,tree,block)
        plan.stats['eager_requests']+=1
    count=int(storage.counts[0])
    if count>capacity:
        plan.dfd_fallbacks+=1
        return None
    if count<20:
        raise RuntimeError('Range bound excluded initial neighbors; check geometry and precision')
    top=storage.top
    return dict(indices=top&0xffffffff,distances=(top>>32).int().contiguous().view(torch.float32),
                counts=storage.counts,radius=radius,flags=storage.flags[:count],visits=storage.visits)
