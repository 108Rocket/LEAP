"""Point-Count-Weighted Alignment for initial candidates (eq:dtw-weighted-seed).

F_ij is implemented on rolling anti-diagonals, scaled by max(n,m)/8.
This approximate score initializes search; it is never a pruning bound.
"""
import torch
import triton
import triton.language as tl
from leap.common.interval_index import lex_keys
from leap.dtw.ordered_dtw import candidates as ordered_candidates


@triton.jit
def alignment_kernel(Q,D,QO,DO,POOL,OUT,K:tl.constexpr,QS):
    row=tl.program_id(0);q=row+QS
    lane=tl.program_id(1)*16+tl.arange(0,16)
    d=tl.load(POOL+row*K+lane,lane<K,other=0)
    n=tl.load(QO+q+1)-tl.load(QO+q)
    m=tl.load(DO+d+1,lane<K,other=0)-tl.load(DO+d,lane<K,other=0)
    maximum=tl.maximum(n,m).to(tl.float32)
    j=tl.arange(0,8)
    bx=tl.load(D+d[:,None]*16+j[None,:]*2,lane[:,None]<K,other=0.)
    by=tl.load(D+d[:,None]*16+j[None,:]*2+1,lane[:,None]<K,other=0.)
    F_prev=tl.full((16,8),float('inf'),tl.float32)
    F_prev2=tl.full((16,8),float('inf'),tl.float32)
    for diagonal in range(15):
        i=diagonal-j
        valid=(i>=0)&(i<8)
        ax=tl.load(Q+q*16+i*2,valid,other=0.)
        ay=tl.load(Q+q*16+i*2+1,valid,other=0.)
        # Two preceding anti-diagonals supply up, left and diagonal predecessors.
        shifted=tl.sum(tl.where(j[None,None,:]==j[None,:,None]-1,F_prev[:,None,:],0.),2)
        shifted2=tl.sum(tl.where(j[None,None,:]==j[None,:,None]-1,F_prev2[:,None,:],0.),2)
        left=tl.where(j[None,:]>0,shifted,float('inf'))
        cross=tl.where(j[None,:]>0,shifted2,float('inf'))
        dx=ax[None,:]-bx;dy=ay[None,:]-by
        c_ij=tl.sqrt(dx*dx+dy*dy)
        # eq:dtw-weighted-seed, F_ij: costs c_ij*n/8, c_ij*m/8, c_ij*max(n,m)/8.
        up=F_prev+(n/maximum)[:,None]*c_ij
        horizontal=left+(m/maximum)[:,None]*c_ij
        F_ij=tl.minimum(up,tl.minimum(horizontal,cross+c_ij))
        F_ij=tl.where((i[None,:]==0)&(j[None,:]==0),c_ij,F_ij)
        cur=tl.where(valid[None,:],F_ij,float('inf'))
        F_prev2=F_prev;F_prev=cur
    warped=tl.sum(tl.where(j[None,:]==7,F_prev,0.),1)
    score=maximum*(warped/8.)
    tl.store(OUT+row*K+lane,score,lane<K)


@torch.no_grad()
def candidates(queries,database,k,qids=None,*,pool=128):
    """Blocked max-count scan -> pool -> weighted summary alignment Top-K.

    The pool grows to k when callers request more candidates than its configured
    size. This is an approximate cascade, not global Top-K under the DP score.
    """
    nd=len(database.coordinates)
    available=nd-int(qids is not None)
    if not 1<=k<=available or isinstance(pool,bool) or not isinstance(pool,int) or pool<1:
        raise ValueError('Invalid weighted candidate count or pool')
    width=min(max(pool,k),available)
    initial=ordered_candidates(queries,database,width,qids,max_work=True)
    result=[]
    for start in range(0,len(initial),2000):
        ids=initial[start:start+2000].contiguous()
        score=torch.empty(ids.shape,device=ids.device,dtype=torch.float32)
        alignment_kernel[(len(ids),triton.cdiv(width,16))](
            queries.ordered_summary,database.ordered_summary,queries.offsets,database.offsets,
            ids,score,width,start,num_warps=4,enable_fp_fusion=False)
        keys=lex_keys(score,ids).topk(k,largest=False).values
        result.append(keys&0xffffffff)
    if not result:
        return torch.empty((0,k),device=queries.points.device,dtype=torch.int64)
    return torch.cat(result)
