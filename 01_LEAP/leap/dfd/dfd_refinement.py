"""RGR for DFD: cached original scores and radius-bounded DP (eq:dfd-target)."""
import torch
import triton
import triton.language as tl

@triton.jit
def ordered_fast(P,OFF,a,T,TOFF,b,r2,BLOCK:tl.constexpr):
    a0=tl.load(OFF+a); na=tl.load(OFF+a+1)-a0
    b0=tl.load(TOFF+b); nb=tl.load(TOFF+b+1)-b0
    j=tl.arange(0,BLOCK)
    bx=tl.load(T+(b0+j)*2,j<nb,other=0.); by=tl.load(T+(b0+j)*2+1,j<nb,other=0.)
    cursor=tl.full((),0,tl.int32); k=tl.full((),0,tl.int32)
    while (k<16)&(cursor<nb):
        i=k*(na-1)//15
        dx=tl.load(P+(a0+i)*2)-bx; dy=tl.load(P+(a0+i)*2+1)-by
        feasible=(j>=cursor)&(j<nb)&(dx*dx+dy*dy<=r2)
        cursor=tl.min(tl.where(feasible,j,BLOCK),0); k+=1
    return cursor<nb


@triton.jit
def original_value(P,OFF,q,T,TOFF,d,WORK,pair,r2,BLOCK:tl.constexpr,CUTOFF:tl.constexpr):
    a0=tl.load(OFF+q); na=tl.load(OFF+q+1)-a0
    b0=tl.load(TOFF+d); nb=tl.load(TOFF+d+1)-b0
    j=tl.arange(0,BLOCK)
    bx=tl.load(T+(b0+j)*2,j<nb,other=0.); by=tl.load(T+(b0+j)*2+1,j<nb,other=0.)
    work=WORK+pair*3*BLOCK
    for bank in range(3): tl.store(work+bank*BLOCK+j,float('inf'))
    tl.debug_barrier()
    diagonal=tl.full((),0,tl.int32); alive=tl.full((),True,tl.int1)
    while (diagonal<na+nb-1)&alive:
        i=diagonal-j; valid=(i>=0)&(i<na)&(j<nb)
        ax=tl.load(P+(a0+i)*2,valid,other=0.); ay=tl.load(P+(a0+i)*2+1,valid,other=0.)
        previous=tl.load(work+((diagonal+2)%3)*BLOCK+j,volatile=True)
        left=tl.load(work+((diagonal+2)%3)*BLOCK+j-1,j>0,other=float('inf'),volatile=True)
        cross=tl.load(work+((diagonal+1)%3)*BLOCK+j-1,j>0,other=float('inf'),volatile=True)
        predecessor=tl.minimum(previous,tl.minimum(left,cross))
        predecessor=tl.where((i==0)&(j==0),0.,predecessor)
        cost=(ax-bx)*(ax-bx)+(ay-by)*(ay-by)
        current=tl.where(valid,tl.maximum(cost,predecessor),float('inf'))
        tl.store(work+(diagonal%3)*BLOCK+j,current)
        tl.debug_barrier()
        if CUTOFF:
            # A coupling advances by one or two diagonals. Both frontiers must fail.
            if (diagonal%8==7)&(diagonal<na+nb-2):
                alive=tl.min(tl.minimum(current,previous),0)<=r2
        diagonal+=1
    value=tl.load(work+((na+nb-2)%3)*BLOCK+nb-1,volatile=True)
    return tl.where(alive,tl.sqrt(value),float('inf')),alive



@triton.jit
def cached_kernel(P,OFF,T,TOFF,Q,D,INITIAL,RAW,RADIUS,O,FLAGS,WORK,
                  BLOCK:tl.constexpr,INITIAL_K:tl.constexpr,CACHE:tl.constexpr=64):
    pair=tl.program_id(0); q=tl.load(Q+pair); d=tl.load(D+pair); s=tl.arange(0,CACHE)
    initial=tl.load(INITIAL+q*INITIAL_K+s,s<INITIAL_K,other=-1)
    at=tl.min(tl.where(initial==d,s,CACHE),0)
    if at<INITIAL_K:
        tl.store(O+pair,tl.load(RAW+q*INITIAL_K+at)); tl.store(FLAGS+pair,0)
    else:
        r=tl.load(RADIUS+q); flag=tl.full((),1,tl.int32)
        keep=ordered_fast(P,OFF,q,T,TOFF,d,r*r,BLOCK)
        if keep: keep=ordered_fast(T,TOFF,d,P,OFF,q,r*r,BLOCK)
        if not keep:
            value=tl.full((),float('inf'),P.dtype.element_ty); flag=2
        else:
            value,alive=original_value(P,OFF,q,T,TOFF,d,WORK,pair,r*r,BLOCK,True)
            if not alive: flag=6
        tl.store(O+pair,value); tl.store(FLAGS+pair,flag)


@torch.no_grad()
def cached_pairs(queries,index,qraw,dids,initial,raw_initial,threshold):
    from leap.common.runtime.context import current
    plan = current()
    if plan is not None:
        return plan.pair_values(queries, index, qraw, dids, 3, threshold, initial, raw_initial)
    block=max(queries.dfd_summary['block'],index.dfd_summary['block'])
    from leap.common.runtime import options
    if options.dfd_grouped_refinement and queries.points.dtype == torch.float32 and block > 64:
        from leap.dfd.dfd_grouped_refinement import cached_pairs as grouped
        return grouped(queries,index,qraw,dids,initial,raw_initial,threshold)
    out=torch.empty(dids.shape,device=dids.device,dtype=queries.points.dtype)
    flags=torch.empty(dids.shape,device=dids.device,dtype=torch.int32)
    if not len(dids):
        return out,flags

    def evaluate(qs,ds,values,status,width):
        chunk=max(1,min(len(ds),(32*1024*1024)//(3*width)))
        work=torch.empty((chunk,3,width),device=dids.device,dtype=queries.points.dtype)
        for lo in range(0,len(ds),chunk):
            hi=min(len(ds),lo+chunk)
            cached_kernel[(hi-lo,)](queries.points,queries.offsets,index.points,index.offsets,
                qs[lo:hi],ds[lo:hi],initial,raw_initial,threshold,values[lo:hi],status[lo:hi],
                work,width,initial.shape[1],max(64,triton.next_power_of_2(initial.shape[1])),
                num_warps=4,enable_fp_fusion=False)

    if block<=64:
        evaluate(qraw,dids,out,flags,block)
        return out,flags
    # The ordered check runs both directions; each bucket must fit BOTH paths.
    lengths=torch.maximum(queries.offsets[qraw+1]-queries.offsets[qraw],
                          index.offsets[dids+1]-index.offsets[dids])
    lower=0
    for width in (2**i for i in range(6,block.bit_length())):
        rows=((lengths>lower)&(lengths<=width)).nonzero().flatten()
        lower=width
        if not len(rows):
            continue
        values=torch.empty(len(rows),device=dids.device,dtype=queries.points.dtype)
        status=torch.empty(len(rows),device=dids.device,dtype=torch.int32)
        evaluate(qraw[rows].contiguous(),dids[rows].contiguous(),values,status,width)
        out[rows]=values
        flags[rows]=status
    return out,flags
