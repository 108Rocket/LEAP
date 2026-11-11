"""Certified raw-geometry rejection inside a fixed candidate pool.

Probe scores are never used as geometric bounds. Each partial direction visits
original source vertices and ALL target segments. Its maximum is a lower bound
on the completed raw vertex-to-segment score. Strict rejection preserves ties.
"""
import torch
import triton
import triton.language as tl
from leap.common.numeric import geometry_divide
from leap.common.raw import direction, fused_raw_scores


@triton.jit
def bounded_direction(P,OFF,a,T,TOFF,b,radius,S:tl.constexpr,STRIPED:tl.constexpr):
    a0=tl.load(OFF+a);a1=tl.load(OFF+a+1)
    b0=tl.load(TOFF+b);b1=tl.load(TOFF+b+1)
    i=tl.arange(0,S);j=tl.arange(0,32)
    blocks=tl.cdiv(a1-a0,S)
    far=tl.full((),0,P.dtype.element_ty)
    work=tl.full((),0,tl.int32)
    block=0
    rejected=False
    while (block<blocks) & ~rejected:
        if STRIPED:
            # A permutation of the original vertices, spread across the path.
            source=a0+i*blocks+block
        else:
            source=a0+block*S+i
        valid=source<a1
        x=tl.load(P+source*2,valid,other=0)
        y=tl.load(P+source*2+1,valid,other=0)
        near=tl.full((S,),float('inf'),P.dtype.element_ty)
        for end in range(b0,b1-1,32):
            ax=tl.load(T+(end+j)*2,end+j<b1-1,other=0)
            ay=tl.load(T+(end+j)*2+1,end+j<b1-1,other=0)
            bx=tl.load(T+(end+j+1)*2,end+j<b1-1,other=0)
            by=tl.load(T+(end+j+1)*2+1,end+j<b1-1,other=0)
            ex=bx-ax;ey=by-ay
            rx=x[:,None]-ax[None,:];ry=y[:,None]-ay[None,:]
            denom=tl.maximum(ex*ex+ey*ey,1e-30)
            u=geometry_divide(rx*ex[None,:]+ry*ey[None,:],denom[None,:])
            u=tl.minimum(1.0,tl.maximum(0.0,u))
            dx=rx-u*ex[None,:];dy=ry-u*ey[None,:]
            d=tl.where(end+j[None,:]<b1-1,dx*dx+dy*dy,float('inf'))
            near=tl.minimum(near,tl.min(d,1))
        far=tl.maximum(far,tl.max(tl.where(valid,near,0),0))
        work+=tl.sum(valid.to(tl.int32),0)*(b1-b0-1).to(tl.int32)
        # Compare in the same rounded sqrt domain as the original scorer.
        rejected=tl.sqrt(far)>radius
        block+=1
    return far,rejected,work


@triton.jit
def bounded_pairs(P,OFF,T,TOFF,IDS,RADIUS,OUT,WORK,REJECTED,
                  K:tl.constexpr,SEED:tl.constexpr,S:tl.constexpr,STRIPED:tl.constexpr,DIAG:tl.constexpr):
    pair=tl.program_id(0)
    a=pair//(K-SEED)
    c=pair%(K-SEED)+SEED
    b=tl.load(IDS+a*K+c)
    radius=tl.load(RADIUS+a)
    ab,rejected,work=bounded_direction(P,OFF,a,T,TOFF,b,radius,S,STRIPED)
    ba=tl.full((),0,P.dtype.element_ty)
    if ~rejected:
        ba,rejected,w=bounded_direction(T,TOFF,b,P,OFF,a,radius,S,STRIPED)
        work+=w
    score=tl.where(rejected,float('inf'),tl.sqrt(tl.maximum(ab,ba)))
    tl.store(OUT+a*K+c,score)
    if DIAG:
        tl.store(WORK+a*K+c,work)
        tl.store(REJECTED+a*K+c,rejected.to(tl.int32))


@triton.jit
def bounded_flat_pairs(P,OFF,T,TOFF,Q,D,RADIUS,OUT,WORK,REJECTED,
                       S:tl.constexpr,DIAG:tl.constexpr):
    pair=tl.program_id(0)
    q=tl.load(Q+pair); d=tl.load(D+pair)
    radius=tl.load(RADIUS+q)
    ab,rejected,work=bounded_direction(P,OFF,q,T,TOFF,d,radius,S,False)
    ba=tl.full((),0,P.dtype.element_ty)
    if ~rejected:
        ba,rejected,w=bounded_direction(T,TOFF,d,P,OFF,q,radius,S,False)
        work+=w
    tl.store(OUT+pair,tl.where(rejected,float('inf'),tl.sqrt(tl.maximum(ab,ba))))
    if DIAG:
        tl.store(WORK+pair,work)
        tl.store(REJECTED+pair,rejected)


@torch.no_grad()
def bounded_hausdorff_distances(queries,database,qids,dids,radii,*,source_block=32,diagnostic=False):
    """Exact surviving vertex-to-polyline scores; reject only strictly above radii.

    Each pair runs independently. radii contains one nonnegative search budget
    per query, including any caller-provided numerical slack. No Top-k threshold
    is updated here. Source blocks finish scanning ALL opposite segments before
    their partial maximum can certify rejection.
    """
    device=queries.points.device
    if source_block not in (8,16,32):
        raise ValueError('Source block must be 8, 16 or 32')
    if qids.ndim!=1 or qids.shape!=dids.shape or qids.dtype!=torch.int64 or dids.dtype!=torch.int64:
        raise ValueError('Expected matching one-dimensional int64 pair IDs')
    if radii.shape!=(len(queries.coordinates),) or radii.dtype!=queries.points.dtype:
        raise ValueError('Expected one FP32 radius per query')
    if device.type!='cuda' or any(t.device!=device for t in (database.points,qids,dids,radii)):
        raise ValueError('Geometry, IDs and radii must share a CUDA device')
    out=torch.empty(dids.shape,device=device,dtype=queries.points.dtype)
    work=torch.empty_like(dids,dtype=torch.int32) if diagnostic else out
    rejected=torch.empty_like(dids,dtype=torch.bool) if diagnostic else out
    if dids.numel():
        bounded_flat_pairs[(dids.numel(),)](queries.points,queries.offsets,database.points,database.offsets,
            qids.contiguous(),dids.contiguous(),radii.contiguous(),out,work,rejected,source_block,diagnostic,
            num_warps=4,enable_fp_fusion=False)
    return (out,{'work':work,'rejected':rejected}) if diagnostic else out


@triton.jit
def recover_tied_rows(P,OFF,T,TOFF,IDS,ROWS,OUT,K:tl.constexpr):
    pair=tl.program_id(0)
    a=tl.load(ROWS+pair//K)
    c=pair%K
    b=tl.load(IDS+a*K+c)
    ab=direction(P,OFF,a,T,TOFF,b)
    ba=direction(T,TOFF,b,P,OFF,a)
    tl.store(OUT+a*K+c,tl.sqrt(tl.maximum(ab,ba)))


@torch.no_grad()
def bounded_scores(queries,database,candidates,seed=32,source_block=8,striped=True,diagnostic=False):
    n,k=candidates.shape
    if not (20<=seed<k) or source_block not in (8,16,32):
        raise ValueError('Require20<=seed<K and source block8/16/32')
    candidates=candidates.contiguous()
    seed_scores=fused_raw_scores(queries,database,candidates[:,:seed].contiguous())
    radius=seed_scores.topk(20,largest=False).values[:,-1].contiguous()
    out=torch.empty((n,k),device=candidates.device,dtype=queries.points.dtype)
    out[:,:seed].copy_(seed_scores)
    work=torch.zeros_like(candidates,dtype=torch.int32) if diagnostic else out
    rejected=torch.zeros_like(candidates,dtype=torch.int32) if diagnostic else out
    bounded_pairs[(n*(k-seed),)](queries.points,queries.offsets,database.points,database.offsets,
        candidates,radius,out,work,rejected,k,seed,source_block,striped,diagnostic,
        num_warps=4,enable_fp_fusion=False)
    return out,{'radius':radius,'work':work,'rejected':rejected}


@torch.no_grad()
def select_with_original_ties(queries,database,candidates,scores):
    # torch.topk tie order can change when discarded scores become infinity.
    # Recompute only rows with an internal/boundary Top20 tie, then use exactly
    # the original full-pool input for those rows. This is included in timings.
    values=scores.topk(21,largest=False).values
    tied=(values[:,:-1]==values[:,1:]).any(1)
    rows=torch.nonzero(tied,as_tuple=False).flatten()
    if len(rows):
        recover_tied_rows[(len(rows)*candidates.shape[1],)](queries.points,queries.offsets,
            database.points,database.offsets,candidates,rows,scores,candidates.shape[1],
            num_warps=4,enable_fp_fusion=False)
    distances,order=scores.topk(20,largest=False)
    return {'indices':candidates.gather(1,order),'distances':distances,'tie_rows':len(rows)}


@torch.no_grad()
def early_refine(queries,database,candidates,seed=32,source_block=8,striped=True,diagnostic=False):
    scores,details=bounded_scores(queries,database,candidates,seed,source_block,striped,diagnostic)
    if diagnostic:
        details['pre_tie_scores']=scores.clone()
    result=select_with_original_ties(queries,database,candidates,scores)
    if diagnostic:
        result.update(details)
        result['scores']=scores
    return result
