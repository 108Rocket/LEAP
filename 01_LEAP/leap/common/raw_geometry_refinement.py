"""Level 3: Raw Geometry Refinement (RGR).

eq:target uses original source vertices and target segments, not point-set
Hausdorff. eq:dtw-target uses the ordered Euclidean-sum DP in dtw.py.
Kernels retain squared Hausdorff distances until the final square root.
"""
import torch
import triton
import triton.language as tl
from leap.common.numeric import geometry_divide
from leap.dtw.dtw import dtw_pairs as dtw_distances


@triton.jit
def directed_hausdorff_squared(P, OFF, a, T, TOFF, b):
    # eq:target: h(q,T) = max_{x in V_q} min_{y in E_T} ||x-y||.
    a0=tl.load(OFF+a);a1=tl.load(OFF+a+1)
    b0=tl.load(TOFF+b);b1=tl.load(TOFF+b+1)
    i=tl.arange(0,32);j=tl.arange(0,32)
    h_qT_squared=tl.full((),0,P.dtype.element_ty)
    for start in range(a0,a1,32):
        x=tl.load(P+(start+i)*2,start+i<a1,other=0)
        y=tl.load(P+(start+i)*2+1,start+i<a1,other=0)
        nearest_squared=tl.full((32,),float('inf'),P.dtype.element_ty)
        for end in range(b0,b1-1,32):
            ax=tl.load(T+(end+j)*2,end+j<b1-1,other=0)
            ay=tl.load(T+(end+j)*2+1,end+j<b1-1,other=0)
            bx=tl.load(T+(end+j+1)*2,end+j<b1-1,other=0)
            by=tl.load(T+(end+j+1)*2+1,end+j<b1-1,other=0)
            ex=bx-ax;ey=by-ay
            rx=x[:,None]-ax[None,:];ry=y[:,None]-ay[None,:]
            denom=tl.maximum(ex*ex+ey*ey,1e-30)
            # eq:projection: t* = clip((x-u).e / ||e||^2, 0, 1).
            t_star=geometry_divide(rx*ex[None,:]+ry*ey[None,:],denom[None,:])
            t_star=tl.minimum(1.0,tl.maximum(0.0,t_star))
            dx=rx-t_star*ex[None,:];dy=ry-t_star*ey[None,:]
            d=tl.where(end+j[None,:]<b1-1,dx*dx+dy*dy,float('inf'))
            nearest_squared=tl.minimum(nearest_squared,tl.min(d,1))
        h_qT_squared=tl.maximum(h_qT_squared,tl.max(tl.where(start+i<a1,nearest_squared,0),0))
    return h_qT_squared


@triton.jit
def candidate_distance_kernel(P, OFF, T, TOFF, IDS, O, K: tl.constexpr):
    pair=tl.program_id(0);a=pair//K;b=tl.load(IDS+pair)
    h_qT_squared=directed_hausdorff_squared(P,OFF,a,T,TOFF,b)
    h_Tq_squared=directed_hausdorff_squared(T,TOFF,b,P,OFF,a)
    tl.store(O+pair,tl.sqrt(tl.maximum(h_qT_squared,h_Tq_squared)))


@torch.no_grad()
def hausdorff_candidate_distances(queries, database, candidates):
    """Evaluate H(q,T) for each row of candidate IDs."""
    out=torch.empty(candidates.shape,device=candidates.device,dtype=queries.points.dtype)
    candidate_distance_kernel[(candidates.numel(),)](
        queries.points,queries.offsets,database.points,database.offsets,candidates,
        out,candidates.shape[1],num_warps=4,enable_fp_fusion=False)
    return out


@triton.jit
def point_direction(P,OFF,a,T,TOFF,b):
    a0=tl.load(OFF+a);a1=tl.load(OFF+a+1)
    b0=tl.load(TOFF+b);b1=tl.load(TOFF+b+1)
    i=tl.arange(0,32);j=tl.arange(0,32)
    far=tl.full((),0.,P.dtype.element_ty)
    for start in range(a0,a1,32):
        x=tl.load(P+(start+i)*2,start+i<a1,other=0.)
        y=tl.load(P+(start+i)*2+1,start+i<a1,other=0.)
        near=tl.full((32,),float('inf'),P.dtype.element_ty)
        for end in range(b0,b1,32):
            u=tl.load(T+(end+j)*2,end+j<b1,other=0.)
            v=tl.load(T+(end+j)*2+1,end+j<b1,other=0.)
            dx=x[:,None]-u[None,:];dy=y[:,None]-v[None,:]
            d=dx*dx+dy*dy
            near=tl.minimum(near,tl.min(tl.where(end+j[None,:]<b1,d,float('inf')),1))
        far=tl.maximum(far,tl.max(tl.where(start+i<a1,near,0.),0))
    return far


@triton.jit
def hausdorff_pair_kernel(P,OFF,T,TOFF,Q,D,OUT,POINT:tl.constexpr):
    pair=tl.program_id(0);q=tl.load(Q+pair);d=tl.load(D+pair)
    if POINT:
        h_qT_squared=point_direction(P,OFF,q,T,TOFF,d)
        h_Tq_squared=point_direction(T,TOFF,d,P,OFF,q)
    else:
        h_qT_squared=directed_hausdorff_squared(P,OFF,q,T,TOFF,d)
        h_Tq_squared=directed_hausdorff_squared(T,TOFF,d,P,OFF,q)
    tl.store(OUT+pair,tl.sqrt(tl.maximum(h_qT_squared,h_Tq_squared)))


@torch.no_grad()
def hausdorff_distances(queries,database,qids,dids,point=False):
    """eq:target: H(q,T) = max(h(q,T), h(T,q)) for flat pairs.

    point=True is the legacy H_V diagnostic, not the paper's Hausdorff target.
    """
    out=torch.empty(dids.shape,device=dids.device,dtype=queries.points.dtype)
    hausdorff_pair_kernel[(dids.numel(),)](
        queries.points,queries.offsets,database.points,database.offsets,
        qids.contiguous(),dids.contiguous(),out,point,num_warps=4,enable_fp_fusion=False)
    return out
