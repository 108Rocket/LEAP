"""Bounded-memory per-query Top-K over ragged DTW candidate values."""
import torch
import triton
import triton.language as tl


@triton.jit
def top_kernel(INITIAL,INITIAL_D,DIDS,VALUES,OFF,COUNTS,OUT,NSEED:tl.constexpr,K:tl.constexpr):
    row=tl.program_id(0);j=tl.arange(0,128)
    inf=tl.full((),9187343244130779135,tl.int64)
    seed_id=tl.load(INITIAL+row*NSEED+j,j<NSEED,other=0)
    seed_d=tl.load(INITIAL_D+row*NSEED+j,j<NSEED,other=0.)
    key=(seed_d.to(tl.int32,bitcast=True).to(tl.int64)<<32)|seed_id
    best=tl.sort(tl.where(j<NSEED,key,inf),descending=False)
    base=tl.load(OFF+row);count=tl.load(COUNTS+row)
    for start in range(0,count,128):
        valid=start+j<count
        did=tl.load(DIDS+base+start+j,valid,other=0)
        value=tl.load(VALUES+base+start+j,valid,other=0.)
        key=(value.to(tl.int32,bitcast=True).to(tl.int64)<<32)|did
        key=tl.where(valid,key,inf)
        both=tl.sort(tl.reshape(tl.join(best,key),(256,)),descending=False)
        best,_=tl.split(tl.trans(tl.reshape(both,(2,128))))
    tl.store(OUT+row*K+j,best,j<K)


def topk(initial,initial_d,dids,values,offsets,counts,k):
    if k>128 or initial.shape[1]>128:
        from leap.common.interval_index import lex_keys
        n,s=initial.shape
        width=s+int(counts.max())
        if n*width>8_000_000:
            raise MemoryError('Large-k final reduction exceeds bounded workspace; split query batch')
        keys=torch.full((n,width),9187343244130779135,device=counts.device,dtype=torch.int64)
        keys[:,:s]=lex_keys(initial_d,initial)
        rows=torch.repeat_interleave(torch.arange(n,device=counts.device),counts.long())
        columns=torch.arange(len(dids),device=counts.device)-offsets[rows]+s
        keys[rows,columns]=lex_keys(values,dids)
        return keys.topk(k,largest=False).values
    out=torch.empty((len(counts),k),device=counts.device,dtype=torch.int64)
    top_kernel[(len(counts),)](initial,initial_d,dids,values,offsets,counts,out,initial.shape[1],k,
                              num_warps=4,enable_fp_fusion=False)
    return out
