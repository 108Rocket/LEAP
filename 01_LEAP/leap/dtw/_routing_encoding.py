"""Fused pointwise preprocessing; retain native reductions and GEMMs."""
from types import SimpleNamespace
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def geometry(P,O,POINTS,LI,COUNT:tl.constexpr,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//WIDTH;j=i%WIDTH
    start=tl.load(O+row,i<COUNT,other=0);end=tl.load(O+row+1,i<COUNT,other=0)
    valid=(i<COUNT)&(j<end-start)
    x=tl.load(P+(start+j)*2,valid,other=0.)
    y=tl.load(P+(start+j)*2+1,valid,other=0.)
    px=tl.load(P+(start+j-1)*2,valid&(j>0),other=0.)
    py=tl.load(P+(start+j-1)*2+1,valid&(j>0),other=0.)
    dx=tl.where(valid&(j>0),x-px,0.);dy=tl.where(valid&(j>0),y-py,0.)
    length=libdevice.sqrt_rn(dx*dx+dy*dy)
    tl.store(POINTS+2*i,x,i<COUNT);tl.store(POINTS+2*i+1,y,i<COUNT)
    tl.store(LI+i,length,i<COUNT)


@triton.jit
def features(P,O,LI,TOTAL,ARC,F,COUNT:tl.constexpr,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//WIDTH;j=i%WIDTH
    start=tl.load(O+row,i<COUNT,other=0);end=tl.load(O+row+1,i<COUNT,other=2)
    n=end-start;valid=(i<COUNT)&(j<n)
    x=tl.load(P+(start+j)*2,valid,other=0.);y=tl.load(P+(start+j)*2+1,valid,other=0.)
    px=tl.load(P+(start+j-1)*2,valid&(j>0),other=0.)
    py=tl.load(P+(start+j-1)*2+1,valid&(j>0),other=0.)
    nx=tl.load(P+(start+j+1)*2,valid&(j+1<n),other=0.)
    ny=tl.load(P+(start+j+1)*2+1,valid&(j+1<n),other=0.)
    dx=tl.where(valid&(j>0),x-px,0.);dy=tl.where(valid&(j>0),y-py,0.)
    ex=tl.where(valid&(j+1<n),nx-x,0.);ey=tl.where(valid&(j+1<n),ny-y,0.)
    li=tl.load(LI+i,i<COUNT,other=0.)
    lo=tl.load(LI+i+1,valid&(j+1<n),other=0.)
    total=tl.load(TOTAL+row,i<COUNT,other=1.)
    mean=tl.maximum(tl.div_rn(total,(n-1).to(tl.float32)),1e-6)
    denominator=tl.maximum(li*lo,1e-10)
    f0=libdevice.log1p(tl.div_rn(li,mean));f1=libdevice.log1p(tl.div_rn(lo,mean))
    f2=tl.div_rn(dx*ex+dy*ey,denominator);f3=tl.div_rn(dx*ey-dy*ex,denominator)
    f4=tl.div_rn(j.to(tl.float32),(n-1).to(tl.float32))
    arc=tl.load(ARC+i,i<COUNT,other=0.)
    f5=tl.div_rn(arc,tl.maximum(total,1e-6))
    tl.store(F+i*6,tl.where(valid,f0,0.),i<COUNT)
    tl.store(F+i*6+1,tl.where(valid,f1,0.),i<COUNT)
    tl.store(F+i*6+2,tl.where(valid,f2,0.),i<COUNT)
    tl.store(F+i*6+3,tl.where(valid,f3,0.),i<COUNT)
    tl.store(F+i*6+4,tl.where(valid,f4,0.),i<COUNT)
    tl.store(F+i*6+5,tl.where(valid,f5,0.),i<COUNT)


def prepare(index,width):
    n=len(index.offsets)-1;count=n*width
    lengths=index.offsets[1:]-index.offsets[:-1]
    points=torch.empty((n,width,2),device=index.points.device,dtype=index.points.dtype)
    li=torch.empty((n,width),device=index.points.device,dtype=index.points.dtype)
    geometry[(triton.cdiv(count,256),)](index.points,index.offsets,points,li,count,width,256,num_warps=4,enable_fp_fusion=False)
    total=li.sum(1,keepdim=True);arc=li.cumsum(1)
    f=torch.empty((n,width,6),device=index.points.device,dtype=index.points.dtype)
    features[(triton.cdiv(count,256),)](index.points,index.offsets,li,total,arc,f,count,width,256,num_warps=4,enable_fp_fusion=False)
    return points,lengths,f


@triton.jit
def first_layer(F,LENGTHS,W,B,PADDING,HIDDEN,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    n=tl.load(LENGTHS+row);h=tl.arange(0,16)
    acc=tl.full((BLOCK,16),0.,tl.float32)
    if tl.program_id(1)*BLOCK<n:
        for k in tl.static_range(6):
            x=tl.load(F+(row*WIDTH+j)*6+k,(j<WIDTH)&(j<n),other=0.)
            w=tl.load(W+h*6+k)
            acc=tl.fma(x[:,None],w[None,:],acc)
        acc=acc+tl.load(B+h)[None,:]
        value=libdevice.tanh(acc)
    else:
        value=tl.broadcast_to(tl.load(PADDING+h)[None,:],(BLOCK,16))
    tl.store(HIDDEN+(row*WIDTH+j[:,None])*16+h[None,:],value,j[:,None]<WIDTH)


@triton.jit
def mass_kernel(LOGIT,LENGTHS,MASS,COUNT:tl.constexpr,WIDTH:tl.constexpr,BLOCK:tl.constexpr,EXP:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=i//WIDTH;j=i%WIDTH
    n=tl.load(LENGTHS+row,i<COUNT,other=0)
    if tl.sum(((i<COUNT)&(j<n)).to(tl.int32),0)>0:
        x=tl.load(LOGIT+i,i<COUNT,other=0.)
        log_mass=1.5*libdevice.tanh(x*(1./1.5))
        if EXP==0: value=libdevice.exp(log_mass)
        else:value=tl.exp(log_mass)
    else:value=tl.full((BLOCK,),0.,tl.float32)
    tl.store(MASS+i,tl.where(j<n,value,0.),i<COUNT)


@triton.jit
def overlap_kernel(SUM,TOTAL,OUT,COUNT:tl.constexpr,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=i//WIDTH;j=i%WIDTH
    den=tl.load(TOTAL+row,i<COUNT,other=1.)
    right=tl.div_rn(tl.load(SUM+i,i<COUNT,other=0.),den)
    left=tl.div_rn(tl.load(SUM+i-1,(i<COUNT)&(j>0),other=0.),den)
    for k in tl.static_range(8):
        value=tl.maximum(tl.minimum(right,(k+1)/8.)-tl.maximum(left,k/8.),0.)
        tl.store(OUT+i*8+k,value,i<COUNT)


def forward(model,points,lengths,f,*,fast_first=True,fast_mass=True,padding=None):
    n,width=points.shape[:2];count=n*width
    if fast_first:
        hidden=torch.empty((n,width,16),device=points.device,dtype=points.dtype)
        if padding is None:padding=torch.tanh(model.net[0].bias)
        first_layer[(n,triton.cdiv(width,64))](f,lengths,model.net[0].weight,model.net[0].bias,padding,hidden,width,64,
                                            num_warps=4,enable_fp_fusion=False)
        logits=model.net[2](hidden).squeeze(-1)
    else:logits=model.net(f).squeeze(-1)
    if fast_mass:
        mass=torch.empty((n,width),device=points.device,dtype=points.dtype)
        mass_kernel[(triton.cdiv(count,256),)](logits,lengths,mass,count,width,256,0,num_warps=4,enable_fp_fusion=False)
    else:
        mask=torch.arange(width,device=points.device)[None]<lengths[:,None]
        mass=(1.5*torch.tanh(logits/1.5)).exp()*mask
    total=mass.sum(1,keepdim=True);cumulative=mass.cumsum(1)
    overlap=torch.empty((n,width,8),device=points.device,dtype=points.dtype)
    overlap_kernel[(triton.cdiv(count,256),)](cumulative,total,overlap,count,width,256,num_warps=4,enable_fp_fusion=False)
    return (8*torch.einsum('bnk,bnd->bkd',overlap,points)).flatten(1).contiguous()


@torch.no_grad()
def encode(model,index,batch=512):
    width=int((index.offsets[1:]-index.offsets[:-1]).max())
    padding=torch.tanh(model.net[0].bias)
    output=torch.empty((len(index.offsets)-1,16),device=index.points.device,dtype=index.points.dtype)
    for lo in range(0,len(output),batch):
        sub=SimpleNamespace(points=index.points,offsets=index.offsets[lo:lo+batch+1])
        # cuBLAS uses a different first-layer accumulation on small/tail shapes.
        fast_first = len(sub.offsets) - 1 == 512 and width >= 64
        output[lo:lo+batch]=forward(model,*prepare(sub,width),padding=padding,
                                  fast_first=fast_first)
    return output


