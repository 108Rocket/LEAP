"""DFD retained geometry and monotone-subset score, eq:dfd-subset.

Greedy RDP-style selection preserves endpoints and the original point sequence.
The retained sets are nested; short trajectories repeat their last vertex.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def combine_max(a,b): return tl.maximum(a,b)


@triton.jit
def combine_min(a,b): return tl.minimum(a,b)


@triton.jit
def simplify_kernel(P,OFF,I8,I16,I32,A8,A16,A32,BLOCK:tl.constexpr):
    q=tl.program_id(0); begin=tl.load(OFF+q); n=tl.load(OFF+q+1)-begin
    i=tl.arange(0,BLOCK)
    x=tl.load(P+(begin+i)*2,i<n,other=0.); y=tl.load(P+(begin+i)*2+1,i<n,other=0.)
    lastx=tl.load(P+(begin+n-1)*2); lasty=tl.load(P+(begin+n-1)*2+1)
    tl.store(I8+q*8+i,n-1,i<8); tl.store(A8+q*16+i*2,lastx,i<8); tl.store(A8+q*16+i*2+1,lasty,i<8)
    tl.store(I16+q*16+i,n-1,i<16); tl.store(A16+q*32+i*2,lastx,i<16); tl.store(A16+q*32+i*2+1,lasty,i<16)
    tl.store(I32+q*32+i,n-1,i<32); tl.store(A32+q*64+i*2,lastx,i<32); tl.store(A32+q*64+i*2+1,lasty,i<32)
    selected=(i==0)|(i==n-1)
    for step in range(2,32):
        if step<n:
            lo=tl.associative_scan(tl.where(selected,i,0),0,combine_max)
            hi=tl.associative_scan(tl.where(selected,i,n-1),0,combine_min,reverse=True)
            ax=tl.load(P+(begin+lo)*2); ay=tl.load(P+(begin+lo)*2+1)
            bx=tl.load(P+(begin+hi)*2); by=tl.load(P+(begin+hi)*2+1)
            vx=bx-ax; vy=by-ay; dx=x-ax; dy=y-ay
            alpha=tl.minimum(1.,tl.maximum(0.,(dx*vx+dy*vy)/tl.maximum(vx*vx+vy*vy,1e-12)))
            ex=dx-alpha*vx; ey=dy-alpha*vy
            error=tl.where((i<n)&~selected,ex*ex+ey*ey,-1.)
            best=tl.argmax(error,0,tie_break_left=True)
            selected=selected|(i==best)
        rank=tl.cumsum(selected.to(tl.int32),0)-1
        if step==7:
            tl.store(I8+q*8+rank,i,selected)
            tl.store(A8+q*16+rank*2,x,selected); tl.store(A8+q*16+rank*2+1,y,selected)
        if step==15:
            tl.store(I16+q*16+rank,i,selected)
            tl.store(A16+q*32+rank*2,x,selected); tl.store(A16+q*32+rank*2+1,y,selected)
        if step==31:
            tl.store(I32+q*32+rank,i,selected)
            tl.store(A32+q*64+rank*2,x,selected); tl.store(A32+q*64+rank*2+1,y,selected)


@triton.jit
def prefix_min(a,b):
    return tl.minimum(a,b)


@triton.jit
def direction(P,OFF,a,T,TOFF,b,W,S:tl.constexpr,BLOCK:tl.constexpr,CUSTOM:tl.constexpr):
    a0=tl.load(OFF+a); na=tl.load(OFF+a+1)-a0
    b0=tl.load(TOFF+b); nb=tl.load(TOFF+b+1)-b0
    j=tl.arange(0,BLOCK)
    bx=tl.load(T+(b0+j)*2,j<nb,other=0.); by=tl.load(T+(b0+j)*2+1,j<nb,other=0.)
    previous=tl.full((BLOCK,),0.,tl.float32)
    for k in range(S):
        if CUSTOM: i=tl.load(W+a*S+k)
        else: i=k*(na-1)//(S-1)
        x=tl.load(P+(a0+i)*2); y=tl.load(P+(a0+i)*2+1)
        cost=(x-bx)*(x-bx)+(y-by)*(y-by)
        # Target vertices may be skipped, but the retained source points cannot go backward.
        best_prefix=tl.associative_scan(previous,0,prefix_min)
        previous=tl.where(j<nb,tl.maximum(cost,best_prefix),float('inf'))
    return tl.min(previous,0)


@triton.jit
def kernel(P,OFF,T,TOFF,Q,D,WQ,WD,O,S:tl.constexpr,BLOCK:tl.constexpr,CUSTOM:tl.constexpr):
    pair=tl.program_id(0); q=tl.load(Q+pair); d=tl.load(D+pair)
    a=direction(P,OFF,q,T,TOFF,d,WQ,S,BLOCK,CUSTOM)
    b=direction(T,TOFF,d,P,OFF,q,WD,S,BLOCK,CUSTOM)
    tl.store(O+pair,tl.sqrt(tl.maximum(a,b)))



@triton.jit
def witness_direction(P, OFF, a, T, TOFF, b, S: tl.constexpr):
    a0 = tl.load(OFF+a); na = tl.load(OFF+a+1)-a0
    b0 = tl.load(TOFF+b); nb = tl.load(TOFF+b+1)-b0
    s = tl.arange(0, S); j = tl.arange(0, 32)
    i = (s*(na-1))//(S-1)
    x = tl.load(P+(a0+i)*2); y = tl.load(P+(a0+i)*2+1)
    nearest = tl.full((S,), float('inf'), tl.float32)
    for start in range(0, nb, 32):
        u = tl.load(T+(b0+start+j)*2, start+j<nb, other=0.)
        v = tl.load(T+(b0+start+j)*2+1, start+j<nb, other=0.)
        dx = x[:, None]-u[None, :]; dy = y[:, None]-v[None, :]
        squared = tl.where(start+j[None, :]<nb, dx*dx+dy*dy, float('inf'))
        nearest = tl.minimum(nearest, tl.min(squared, 1))
    return tl.max(nearest, 0)


@triton.jit
def witness_kernel(P, OFF, T, TOFF, Q, D, O, S: tl.constexpr):
    pair = tl.program_id(0); q = tl.load(Q+pair); d = tl.load(D+pair)
    a = witness_direction(P, OFF, q, T, TOFF, d, S)
    b = witness_direction(T, TOFF, d, P, OFF, q, S)
    tl.store(O+pair, tl.sqrt(tl.maximum(a, b)))


def witness_bound(queries, index, qraw, dids, samples=8):
    out = torch.empty(dids.shape, device=dids.device, dtype=torch.float32)
    if dids.numel():
        witness_kernel[(dids.numel(),)](queries.points, queries.offsets, index.points, index.offsets,
                 qraw.contiguous(), dids.contiguous(), out, samples, num_warps=4, enable_fp_fusion=False)
    return out




def signature(index, summary):
    values = [index.points, index.offsets, *summary['ids'].values()]
    return tuple((t.data_ptr(), t._version, tuple(t.shape)) for t in values)


def subset(index,lo,hi):
    from leap.common.model import TrajectoryIndex
    summary={'ids':{s:v[lo:hi] for s,v in index.dfd_summary['ids'].items()},
             'block':index.dfd_summary['block']}
    return TrajectoryIndex(index.coordinates[lo:hi],index.points,index.offsets[lo:hi+1],
                           index.asset_id,dfd_summary=summary)


@torch.no_grad()
def prepare(index):
    """Cache retained indices once; never replace packed raw vertices."""
    from leap.common.runtime.context import current
    plan = current()
    if plan is not None:
        return plan.dfd_summary(index)
    if index.points.dtype == torch.float32 and len(index.offsets) - 1 >= 4096:
        from leap.dtw._summary_encoding import dfd_summary
        return dfd_summary(index)
    return _prepare_reference(index)


@torch.no_grad()
def _prepare_reference(index):
    n=len(index.coordinates); device=index.points.device
    maximum=int((index.offsets[1:]-index.offsets[:-1]).max())
    block=max(32,triton.next_power_of_2(maximum))
    ids={s:torch.empty((n,s),device=device,dtype=torch.int32) for s in (8,16,32)}
    coords={s:torch.empty((n,s,2),device=device,dtype=torch.float32) for s in (8,16,32)}
    simplify_kernel[(n,)](index.points,index.offsets,*ids.values(),*coords.values(),block,
                         num_warps=4,enable_fp_fusion=False)
    return {'ids':ids,'block':triton.next_power_of_2(maximum)}


@torch.no_grad()
def ordered_bound(queries,index,qids,dids,samples=8):
    """eq:dfd-subset: max of both directed monotone subset bottlenecks."""
    out=torch.empty(dids.shape,device=dids.device,dtype=torch.float32)
    block=max(queries.dfd_summary['block'],index.dfd_summary['block'])
    if dids.numel():
        kernel[(dids.numel(),)](queries.points,queries.offsets,index.points,index.offsets,
            qids.contiguous(),dids.contiguous(),queries.dfd_summary['ids'][samples],
            index.dfd_summary['ids'][samples],out,samples,block,True,
            num_warps=4,enable_fp_fusion=False)
    return out
