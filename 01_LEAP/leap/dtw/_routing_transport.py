"""Native-order reductions and sparse transport for large FP32 routing.

These kernels retain the reference sum/scan tree and ordered FMA evaluation.
The dispatcher limits use to the validated CUDA configuration and shapes.
"""
import triton
import triton.language as tl


@triton.jit
def sum_native(X, OUT, LOCAL: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 32)
    a = tl.full((32,), 0., tl.float32)
    b = tl.full((32,), 0., tl.float32)
    c = tl.full((32,), 0., tl.float32)
    d = tl.full((32,), 0., tl.float32)
    for start in range(triton.cdiv(LOCAL, 128)):
        j = start * 128 + lane * 4
        a += tl.load(X + row * LOCAL + j, (j < LOCAL) & (j < WIDTH), other=0.)
        b += tl.load(X + row * LOCAL + j+1, (j+1 < LOCAL) & (j+1 < WIDTH), other=0.)
        c += tl.load(X + row * LOCAL + j+2, (j+2 < LOCAL) & (j+2 < WIDTH), other=0.)
        d += tl.load(X + row * LOCAL + j+3, (j+3 < LOCAL) & (j+3 < WIDTH), other=0.)
    value = ((a + b) + c) + d
    # Adjacent pairs match lane zero after native shuffle-down 1,2,4,8,16.
    v16 = tl.sum(tl.reshape(value, (16, 2)), 1)
    v8 = tl.sum(tl.reshape(v16, (8, 2)), 1)
    v4 = tl.sum(tl.reshape(v8, (4, 2)), 1)
    v2 = tl.sum(tl.reshape(v4, (2, 2)), 1)
    tl.store(OUT + row, tl.sum(v2, 0))


@triton.jit
def scan_step(value, CHUNK: tl.constexpr, STEP: tl.constexpr):
    grouped = tl.reshape(value, (CHUNK // (2 * STEP), 2 * STEP))
    columns = tl.arange(0, 2 * STEP)
    previous = tl.sum(tl.where(columns[None,:] == STEP-1, grouped, 0.), 1)
    grouped = tl.where(columns[None,:] >= STEP, grouped + previous[:,None], grouped)
    return tl.reshape(grouped, (CHUNK,))


@triton.jit
def scan_native(X, OUT, LOCAL: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, CHUNK)
    carry = tl.full((), 0., tl.float32)
    for start in range(triton.cdiv(LOCAL, CHUNK)):
        j = start * CHUNK + lane
        value = tl.load(X + row * LOCAL + j, j < LOCAL, other=0.)
        value = tl.where(lane == 0, value + carry, value)
        # Match PyTorch's Sklansky scan, including the carry at each chunk start.
        for level in tl.static_range(0, 10):
            if (1 << level) < CHUNK:
                value = scan_step(value, CHUNK, 1 << level)
        tl.store(OUT + row * LOCAL + j, value, j < LOCAL)
        carry = tl.sum(tl.where(lane == CHUNK-1, value, 0.), 0)


@triton.jit
def interval_kernel(P,O,SUM,TOTAL,OUT,WIDTH:tl.constexpr,ROWS:tl.constexpr=8):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    col=tl.arange(0,16)
    start=tl.load(O+row);end=tl.load(O+row+1);n=(end-start).to(tl.int32)
    den=tl.load(TOTAL+row)
    low_edge=(col//2)/8.;high_edge=(col//2+1)/8.
    lower=tl.full((ROWS,16),0,tl.int32)
    upper=tl.broadcast_to(n[:,None],(ROWS,16))
    # Mass is positive. Find the first prefix strictly beyond the window edge.
    while tl.sum(tl.sum((lower<upper).to(tl.int32),1),0)>0:
        middle=(lower+upper)//2
        valid=(lower<upper)&(middle<n[:,None])
        prefix=tl.div_rn(tl.load(SUM+row[:,None]*WIDTH+middle,valid,other=0.),den[:,None])
        advance=valid&(prefix<=low_edge[None,:])
        lower=tl.where(advance,middle+1,lower)
        upper=tl.where(valid&~advance,middle,upper)
    j=lower
    previous=tl.div_rn(tl.load(SUM+row[:,None]*WIDTH+j-1,(j>0)&(j<=n[:,None]),other=0.),den[:,None])
    active=(j<n[:,None])&(previous<high_edge[None,:])
    result=tl.full((ROWS,16),0.,tl.float32)
    while tl.sum(tl.sum(active.to(tl.int32),1),0)>0:
        right=tl.div_rn(tl.load(SUM+row[:,None]*WIDTH+j,active,other=0.),den[:,None])
        weight=tl.maximum(tl.minimum(right,high_edge[None,:])-tl.maximum(previous,low_edge[None,:]),0.)
        point=tl.load(P+(start[:,None]+j)*2+(col[None,:]%2),active,other=0.)
        result=tl.fma(tl.where(active,weight,0.),point,result)
        previous=right;j+=1
        active=active&(j<n[:,None])&(previous<high_edge[None,:])
    tl.store(OUT+row[:,None]*16+col[None,:],result*8.)
