"""Cached bin means mu_i(T) and initial score s_0 (eq:dtw-seed-scan).

The approximate 16D score selects initial candidates, not pruning decisions.
"""
import torch
import triton
import triton.language as tl
from leap.common.interval_index import lex_keys


def signature(points,offsets,summary):
    return tuple((t.data_ptr(),t._version) for t in (points,offsets,summary))


@triton.jit
def summary_kernel(P,O,OUT,N:tl.constexpr):
    row=tl.program_id(0)
    start=tl.load(O+row);n=tl.load(O+row+1)-start
    j=tl.arange(0,N)
    x=tl.load(P+(start+j)*2,j<n,other=0.)
    y=tl.load(P+(start+j)*2+1,j<n,other=0.)
    for b in range(8):
        mask=(j<n)&((j*8)//n==b)
        count=tl.sum(mask.to(tl.float32),0)
        # mu_b is the mean of consecutive point-index bin b, not arc resampling.
        center=tl.minimum(((2*b+1)*n)//16,n-1)
        cx=tl.load(P+(start+center)*2);cy=tl.load(P+(start+center)*2+1)
        mx=tl.where(count>0,tl.sum(tl.where(mask,x,0.),0)/tl.maximum(count,1.),cx)
        my=tl.where(count>0,tl.sum(tl.where(mask,y,0.),0)/tl.maximum(count,1.),cy)
        tl.store(OUT+row*16+b*2,mx);tl.store(OUT+row*16+b*2+1,my)


@torch.no_grad()
def summarize(points,offsets):
    n=len(offsets)-1
    block=triton.next_power_of_2(int((offsets[1:]-offsets[:-1]).max()))
    out=torch.empty((n,16),device=points.device,dtype=torch.float32)
    summary_kernel[(n,)](points,offsets,out,block,num_warps=4,enable_fp_fusion=False)
    return out


@triton.jit
def score_kernel(Q,D,QO,DO,SELF,OUT,ND:tl.constexpr,QS,DS,SAME:tl.constexpr,MAX_WORK:tl.constexpr):
    row=tl.program_id(0);q=row+QS
    lane=tl.program_id(1)*16+tl.arange(0,16);d=lane+DS
    n=tl.load(QO+q+1)-tl.load(QO+q)
    m=tl.load(DO+d+1,lane<ND,other=0)-tl.load(DO+d,lane<ND,other=0)
    total=tl.full((16,),0.,tl.float32)
    for b in range(8):
        dx=tl.load(Q+q*16+b*2)-tl.load(D+d*16+b*2,lane<ND,other=0.)
        dy=tl.load(Q+q*16+b*2+1)-tl.load(D+d*16+b*2+1,lane<ND,other=0.)
        total+=tl.sqrt(dx*dx+dy*dy)
    # Preserve the legacy operation order; weighted mode uses maximum point count.
    scale=tl.maximum(n,m) if MAX_WORK else (n+m)*0.5
    score=scale*(total/8.)
    if SAME:
        score=tl.where(d==tl.load(SELF+q),float('inf'),score)
    tl.store(OUT+row*ND+lane,score,lane<ND)


@torch.no_grad()
def candidates(queries,database,k,qids=None,*,max_work=False):
    """Bounded exact Top-K for the approximate score, with database-ID ties."""
    nq,nd=len(queries.coordinates),len(database.coordinates)
    if not 1<=k<=nd:
        raise ValueError('Invalid ordered candidate count')
    result=[]
    for start in range(0,nq,2000):
        stop=min(start+2000,nq);best=None
        for dstart in range(0,nd,4096):
            width=min(4096,nd-dstart)
            score=torch.empty((stop-start,width),device=queries.points.device,dtype=torch.float32)
            score_kernel[(stop-start,triton.cdiv(width,16))](queries.ordered_summary,database.ordered_summary,
                queries.offsets,database.offsets,qids if qids is not None else queries.offsets,
                score,width,start,dstart,qids is not None,max_work,num_warps=4,enable_fp_fusion=False)
            ids=torch.arange(dstart,dstart+width,device=score.device)[None].expand_as(score)
            keys=lex_keys(score,ids).topk(min(k,width),largest=False).values
            if best is not None:
                keys=torch.cat((best,keys),1)
                keys=keys.topk(min(k,keys.shape[1]),largest=False).values
            best=keys
        result.append(best&0xffffffff)
    if not result:
        return torch.empty((0,k),device=queries.points.device,dtype=torch.int64)
    return torch.cat(result)
