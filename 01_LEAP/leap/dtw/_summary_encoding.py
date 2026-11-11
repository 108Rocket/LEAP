"""Length-bucketed construction of unchanged geometric summaries."""

import torch
import triton
import triton.language as tl

MIN_ROWS = 4096


def buckets(index, minimum=32):
    lengths = index.offsets[1:] - index.offsets[:-1]
    maximum = int(lengths.max())
    previous = 0
    for power in range((minimum.bit_length()-1), max(minimum, triton.next_power_of_2(maximum)).bit_length()):
        width = 1 << power
        rows = torch.nonzero((lengths > previous) & (lengths <= width), as_tuple=False).flatten()
        if rows.numel():
            yield width, rows
        previous = width


@triton.jit
def maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def minimum(a, b):
    return tl.minimum(a, b)


@triton.jit
def simplify(P, OFF, ROWS, I8, I16, I32, BLOCK: tl.constexpr):
    q = tl.load(ROWS+tl.program_id(0))
    begin = tl.load(OFF+q); n = tl.load(OFF+q+1)-begin
    i = tl.arange(0, BLOCK)
    x = tl.load(P+(begin+i)*2, i<n, other=0.)
    y = tl.load(P+(begin+i)*2+1, i<n, other=0.)
    tl.store(I8+q*8+i, n-1, i<8)
    tl.store(I16+q*16+i, n-1, i<16)
    tl.store(I32+q*32+i, n-1, i<32)
    selected = (i==0) | (i==n-1)
    for step in range(2, 32):
        if step<n:
            lo = tl.associative_scan(tl.where(selected, i, 0), 0, maximum)
            hi = tl.associative_scan(tl.where(selected, i, n-1), 0, minimum, reverse=True)
            ax=tl.load(P+(begin+lo)*2); ay=tl.load(P+(begin+lo)*2+1)
            bx=tl.load(P+(begin+hi)*2); by=tl.load(P+(begin+hi)*2+1)
            vx=bx-ax; vy=by-ay; dx=x-ax; dy=y-ay
            alpha=tl.minimum(1.,tl.maximum(0.,(dx*vx+dy*vy)/tl.maximum(vx*vx+vy*vy,1e-12)))
            ex=dx-alpha*vx; ey=dy-alpha*vy
            error=tl.where((i<n)&~selected,ex*ex+ey*ey,-1.)
            best=tl.argmax(error,0,tie_break_left=True)
            selected=selected|(i==best)
        rank=tl.cumsum(selected.to(tl.int32),0)-1
        if step==7: tl.store(I8+q*8+rank,i,selected)
        if step==15: tl.store(I16+q*16+rank,i,selected)
        if step==31: tl.store(I32+q*32+rank,i,selected)


@torch.no_grad()
def dfd_summary(index):
    n = len(index.offsets)-1
    ids = {s:torch.empty((n,s),device=index.points.device,dtype=torch.int32) for s in (8,16,32)}
    for width, rows in buckets(index):
        simplify[(len(rows),)](index.points,index.offsets,rows,*ids.values(),width,
                              num_warps=4,enable_fp_fusion=False)
    block=triton.next_power_of_2(int((index.offsets[1:]-index.offsets[:-1]).max()))
    return {'ids':ids,'block':block}


@triton.jit
def radial_kernel(P,O,TH,ROWS,OUT,M:tl.constexpr,BINS:tl.constexpr,N:tl.constexpr):
    row=tl.load(ROWS+tl.program_id(0)); f=tl.program_id(1)
    lo=tl.load(O+row); n=tl.load(O+row+1)-lo
    j=tl.arange(0,N); valid=j<n
    x=tl.load(P+(lo+j)*2,valid,other=0.)-tl.load(TH+f*2)
    y=tl.load(P+(lo+j)*2+1,valid,other=0.)-tl.load(TH+f*2+1)
    r=tl.sqrt(tl.maximum(x*x+y*y,1e-24))
    base=(row*M+f)*(BINS+2)
    tl.store(OUT+base,tl.min(tl.where(valid,r,float('inf')),0))
    tl.store(OUT+base+1,tl.max(tl.where(valid,r,-float('inf')),0))
    for b in range(BINS):
        keep=valid&((j*BINS)//n==b)
        count=tl.sum(keep.to(tl.float32),0)
        tl.store(OUT+base+2+b,tl.sum(tl.where(keep,r,0.),0)/tl.maximum(count,1.))


@triton.jit
def ordered_kernel(P,O,TH,ROWS,OUT,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr):
    row=tl.load(ROWS+tl.program_id(0));f=tl.program_id(1)
    lo=tl.load(O+row);n=tl.load(O+row+1)-lo;k=tl.minimum(n,K)
    i=tl.arange(0,N)
    x=tl.load(P+(lo+i)*2,i<n,other=0.)-tl.load(TH+f*2)
    y=tl.load(P+(lo+i)*2+1,i<n,other=0.)-tl.load(TH+f*2+1)
    r=tl.sqrt(tl.maximum(x*x+y*y,1e-24))
    for s in range(K):
        mask=(i<n)&((i*k)//n==s)
        low=tl.min(tl.where(mask,r,float('inf')),0)
        high=tl.max(tl.where(mask,r,-float('inf')),0)
        base=((row*K+s)*M+f)*2
        tl.store(OUT+base,tl.where(s<k,low,0.))
        tl.store(OUT+base+1,tl.where(s<k,high,0.))


@torch.no_grad()
def radial(index,theta,bins=4):
    n=len(index.offsets)-1;m=len(theta)
    out=torch.empty((n,m,bins+2),device=theta.device,dtype=index.points.dtype)
    for width,rows in buckets(index):
        radial_kernel[(len(rows),m)](index.points,index.offsets,theta,rows,out,m,bins,width,
                                    num_warps=4,enable_fp_fusion=False)
    return out


@torch.no_grad()
def ordered(index,theta,bins=4):
    n=len(index.offsets)-1;m=len(theta)
    out=torch.empty((n,bins,m,2),device=theta.device,dtype=index.points.dtype)
    for width,rows in buckets(index):
        ordered_kernel[(len(rows),m)](index.points,index.offsets,theta,rows,out,m,bins,width,
                                     num_warps=4,enable_fp_fusion=False)
    return out


