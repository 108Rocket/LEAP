"""One GPU length-bucket layout and one count readback per raw candidate set."""
import torch
import triton
import triton.language as tl
from leap.dtw import dtw_bounded as native
from leap.dtw.dtw import dtw_kernel
from leap.dtw.dtw_frontier import batch_frontier_kernel
from leap.common.runtime.context import current

@triton.jit
def histogram(OFF,IDS,H,TOTAL:tl.constexpr,G:tl.constexpr,K:tl.constexpr,B:tl.constexpr,KB:tl.constexpr):
    group=tl.program_id(0);i=group*B+tl.arange(0,B);b=tl.arange(0,KB)
    d=tl.load(IDS+i,i<TOTAL,other=0)
    length=tl.load(OFF+d+1)-tl.load(OFF+d)
    high=1<<b;low=high>>1
    mask=(i[None,:]<TOTAL)&(b[:,None]<K)&(length[None,:]>low[:,None])&(length[None,:]<=high[:,None])
    tl.store(H+b*G+group,tl.sum(mask.to(tl.int32),1),b<K)


@triton.jit
def pack(OFF,Q,D,LIMIT,H,PREFIX,PQ,PD,PL,ORDER,TOTAL:tl.constexpr,G:tl.constexpr,
         B:tl.constexpr,HAS_LIMIT:tl.constexpr):
    group=tl.program_id(0);bucket=tl.program_id(1)
    size=tl.load(H+bucket*G+group)
    if size>0:
        i=group*B+tl.arange(0,B);valid=i<TOTAL
        d=tl.load(D+i,valid,other=0);q=tl.load(Q+i,valid,other=0)
        length=tl.load(OFF+d+1)-tl.load(OFF+d)
        high=1<<bucket;low=high>>1
        selected=valid&(length>low)&(length<=high)
        rank=tl.cumsum(selected.to(tl.int32),0)-1
        pos=tl.load(PREFIX+bucket*G+group)+rank
        tl.store(PQ+pos,q,selected);tl.store(PD+pos,d,selected);tl.store(ORDER+pos,i,selected)
        if HAS_LIMIT:
            value=tl.load(LIMIT+i,selected,other=0.)
            tl.store(PL+pos,value,selected)


@torch.no_grad()
def pairs(queries,database,qids,dids,limits=None,*,bucket=False,check=8,record=False,chunk=16384,maximum=None):
    plan=current()
    if not bucket or record or plan is not None or queries.points.dtype!=torch.float32:
        return native.pairs(queries,database,qids,dids,limits,bucket=bucket,check=check,record=record,chunk=chunk,maximum=maximum)
    if qids.shape!=dids.shape or qids.ndim!=1:raise ValueError('Expected matching pair vectors')
    if qids.dtype!=torch.int64 or dids.dtype!=torch.int64:raise ValueError('Pair IDs must be int64')
    if any(t.device!=queries.points.device for t in (database.points,qids,dids)):raise ValueError('Device mismatch')
    if limits is not None and (limits.shape!=dids.shape or limits.dtype!=queries.points.dtype or limits.device!=dids.device):
        raise ValueError('Invalid pair limits')
    if check<1 or chunk<1:raise ValueError('Invalid check/chunk')
    n=len(dids);out=torch.empty(n,device=dids.device,dtype=torch.float32)
    if not n:return out
    if maximum is None:maximum=triton.next_power_of_2(int((database.offsets[1:]-database.offsets[:-1]).max()))
    k=maximum.bit_length();groups=triton.cdiv(n,256)
    counts=torch.empty((k,groups),device=dids.device,dtype=torch.int32)
    histogram[(groups,)](database.offsets,dids,counts,n,groups,k,256,triton.next_power_of_2(k),num_warps=4)
    totals=counts.sum(1)
    offsets=totals.cumsum(0)-totals
    prefixes=counts.long().cumsum(1)-counts+offsets[:,None]
    sizes=totals.tolist()
    if sum(sizes)!=n:raise ValueError('Cached maximum does not cover actual pair lengths')
    qs=torch.empty_like(qids);ds=torch.empty_like(dids);order=torch.empty_like(dids)
    budgets=torch.empty_like(out) if limits is not None else out
    pack[(groups,k)](database.offsets,qids,dids,dids if limits is None else limits,counts,prefixes,
        qs,ds,budgets,order,n,groups,256,limits is not None,num_warps=4)
    values=torch.empty_like(out)
    frontier=native._use_batch_frontier(queries,limits,bucket=True,check=check,plan=None)
    kernel=batch_frontier_kernel if frontier else native.bounded_kernel
    begin=0
    for bin,size in enumerate(sizes):
        block=1<<bin
        if size:
            batch=min(chunk,size,(32*1024*1024)//(3*block))
            work=torch.empty((batch,3,block),device=dids.device,dtype=torch.float32)
            for lo in range(begin,begin+size,batch):
                hi=min(lo+batch,begin+size)
                args=(queries.points,queries.offsets,database.points,database.offsets,qs[lo:hi],ds[lo:hi],values[lo:hi],work)
                if limits is None:dtw_kernel[(hi-lo,)](*args,block,num_warps=4,enable_fp_fusion=False)
                else:kernel[(hi-lo,)](*args,budgets[lo:hi],work,block,check,False,
                    num_warps=1 if frontier else 4,enable_fp_fusion=False)
        begin+=size
    out[order]=values
    return out
