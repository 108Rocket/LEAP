"""GEF -> TBF -> RGR under the shared search threshold r_hat_q.

See eq:search-radius and eq:hierarchy. GEF and TBF stay fused in traversal;
the formula helpers inline on GPU. FP32 boundary slack is empirical.
"""
import torch
import triton
import triton.language as tl
from leap.common.range_buffers import enumerate_candidates, top_keys
from leap.common.group_envelope_filter import node_bound_kernel
from leap.common.trajectory_bound_filter import cross_field_bound_kernel
from leap.common.early_refinement import bounded_hausdorff_distances
from leap.hausdorff.cached_refinement import cached_pairs
from leap.common.raw_geometry_refinement import hausdorff_distances, hausdorff_distances as pair_values, hausdorff_pair_kernel as raw_pairs, point_direction


@triton.jit
def enumerate_kernel(Q,D,MEM,TREE,RADIUS,QIDS,SLACK,BASE,OUTD,OUTQ,COUNTS,VISITS,
                     G:tl.constexpr,B:tl.constexpr,WRITE:tl.constexpr,STAGE_CAP:tl.constexpr=0):
    row=tl.program_id(0)
    f=tl.arange(0,64);lane=tl.arange(0,B);slot=tl.arange(0,32);side=tl.arange(0,2)
    a_q=tl.load(Q+row*128+f);b_q=tl.load(Q+row*128+64+f)
    self_id=tl.load(QIDS+row)
    threshold=tl.load(RADIUS+row)+tl.load(SLACK+row)
    stack=tl.where(slot==0,1,0);sp=0
    count=tl.full((),0,tl.int32);nodes=tl.full((),0,tl.int32)
    examined=tl.full((),0,tl.int32)
    while sp>=0:
        node=tl.sum(tl.where(slot==sp,stack,0),0);sp-=1;nodes+=1
        if node<G:
            children=node*2+side
            a_bar_U=tl.load(TREE+children[:,None]*128+f[None,:])
            b_under_U=tl.load(TREE+children[:,None]*128+64+f[None,:])
            # GEF, eq:node-bound: ell(q,U).
            ell_qU=node_bound_kernel(a_q[None,:],b_q[None,:],a_bar_U,b_under_U,1)
            l0=tl.sum(tl.where(side==0,ell_qU,0.),0)
            l1=tl.sum(tl.where(side==1,ell_qU,0.),0)
            if l1<=threshold:
                sp+=1;stack=tl.where(slot==sp,node*2+1,stack)
            if l0<=threshold:
                sp+=1;stack=tl.where(slot==sp,node*2,stack)
        else:
            ids=tl.load(MEM+(node-G)*B+lane)
            valid=(ids>=0)&(ids!=self_id)
            a_T=tl.load(D+ids[:,None]*128+f[None,:],valid[:,None],other=0.)
            b_T=tl.load(D+ids[:,None]*128+64+f[None,:],valid[:,None],other=0.)
            # TBF, eq:lower-bound: L_Theta(q,T).
            L_qT=cross_field_bound_kernel(a_q[None,:],b_q[None,:],a_T,b_T,1)
            keep=valid&(L_qT<=threshold)
            if WRITE or STAGE_CAP > 0:
                rank=tl.cumsum(keep.to(tl.int32),0)-1
                if STAGE_CAP > 0:
                    tl.store(OUTD+row*STAGE_CAP+count+rank,ids,keep&(count+rank<STAGE_CAP))
                else:
                    base=tl.load(BASE+row)
                    tl.store(OUTD+base+count+rank,ids,keep)
                    tl.store(OUTQ+base+count+rank,row,keep)
            count+=tl.sum(keep.to(tl.int32),0)
            examined+=tl.sum(valid.to(tl.int32),0)
    tl.store(COUNTS+row,count)
    tl.store(VISITS+row*2,nodes);tl.store(VISITS+row*2+1,examined)


@torch.no_grad()
def search_chunk(queries,index,tree,qids,seed_k=50,top_k=20,*,initial=None,bounded=True,point=False):
    """eq:search-radius: S_q -> D_S_q -> r_hat_q -> C(q) -> raw Top-k."""
    s=seed_k
    tree.validate(index)
    assert qids.dtype==torch.int64 and qids.is_cuda
    z_q=queries.coordinates.contiguous();n=len(z_q)
    if initial is None:
        S_q,initial_visits=tree.candidates(z_q,qids,s)
    else:
        S_q=initial.contiguous()
        if S_q.ndim!=2 or S_q.shape[0]!=n or S_q.shape[1]<top_k:
            raise ValueError('Initial candidates must cover top_k for every query')
        s=S_q.shape[1]
        initial_visits=torch.zeros((n,2),device=z_q.device,dtype=torch.int32)
    rows=torch.arange(n,device=z_q.device)
    D_S_q=hausdorff_distances(queries,index,rows[:,None].expand_as(S_q).reshape(-1),S_q.reshape(-1),point).reshape(n,s)
    # eq:search-radius: r_hat_q = kth_{T in S(q)} H(q,T).
    r_hat_q=D_S_q.topk(top_k,largest=False).values[:,-1].contiguous()
    slack=(2e-5*(1+z_q.abs().amax(1)+tree.database_scale)).contiguous()
    # eq:hierarchy: ell(q,U) <= L_Theta(q,T) <= H(q,T).
    args=(z_q,tree.d,tree.members,tree.tree,r_hat_q,qids,slack)
    counts,visits,dids,qraw,offsets,maximum=enumerate_candidates(
        enumerate_kernel,args,n,z_q.device,tree.groups,tree.block,top_k,400_000_000)
    # A replaced distance function may implement a different target (e.g. SDD H_V).
    # Keep that scorer unless it explicitly adopts the bounded polyline kernel.
    if hausdorff_distances is pair_values:
        values=cached_pairs(queries,index,qraw,dids,S_q,D_S_q,(r_hat_q+slack).contiguous(),
                            point=point,bounded=bounded)
    else:
        values=hausdorff_distances(queries,index,qraw,dids,point)
    top=top_keys(values,dids,qraw,offsets,n,maximum,top_k)
    result={'indices':top&0xffffffff,'distances':(top>>32).int().contiguous().view(torch.float32),
            'counts':counts,'visits':visits,'seed_visits':initial_visits,'seed_ids':S_q,'radius':r_hat_q}
    return result
