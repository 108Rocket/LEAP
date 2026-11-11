"""Compact DTW routing intermediates with call-local buffer reuse.

Global-width reductions retain the reference FP32 accumulation order. Pointwise
features and network work use a batch-local width; returned summaries own storage.
Only the public routing dispatcher selects this large CUDA FP32 input path.
"""
from types import SimpleNamespace
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from leap.dtw import _routing_encoding as fused
from leap.dtw import dtw_routing
@triton.jit
def compact_mass(LOGITS, LENGTHS, OUT, COUNT: tl.constexpr,
                 WIDTH: tl.constexpr, LOCAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // WIDTH
    j = i % WIDTH
    n = tl.load(LENGTHS + row, i < COUNT, other=0)
    valid = (i < COUNT) & (j < n)
    if tl.sum(valid.to(tl.int32), 0) > 0:
        x = tl.load(LOGITS + row * LOCAL + j, valid, other=0.)
        value = libdevice.exp(1.5 * libdevice.tanh(x * (1. / 1.5)))
    else:
        value = tl.full((BLOCK,), 0., tl.float32)
    tl.store(OUT + i, tl.where(valid, value, 0.), i < COUNT)


@triton.jit
def geometry_compact(P,O,POINTS,LI,COUNT:tl.constexpr,WIDTH:tl.constexpr,GLOBAL_WIDTH:tl.constexpr,BLOCK:tl.constexpr):
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
    tl.store(LI+row*GLOBAL_WIDTH+j,length,i<COUNT)


@triton.jit
def features_compact(P,O,LI,TOTAL,ARC,F,COUNT:tl.constexpr,WIDTH:tl.constexpr,GLOBAL_WIDTH:tl.constexpr,BLOCK:tl.constexpr):
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
    li=tl.load(LI+row*GLOBAL_WIDTH+j,i<COUNT,other=0.)
    lo=tl.load(LI+row*GLOBAL_WIDTH+j+1,valid&(j+1<n),other=0.)
    total=tl.load(TOTAL+row,i<COUNT,other=1.)
    mean=tl.maximum(tl.div_rn(total,(n-1).to(tl.float32)),1e-6)
    denominator=tl.maximum(li*lo,1e-10)
    f0=libdevice.log1p(tl.div_rn(li,mean));f1=libdevice.log1p(tl.div_rn(lo,mean))
    f2=tl.div_rn(dx*ex+dy*ey,denominator);f3=tl.div_rn(dx*ey-dy*ex,denominator)
    f4=tl.div_rn(j.to(tl.float32),(n-1).to(tl.float32))
    arc=tl.load(ARC+row*GLOBAL_WIDTH+j,i<COUNT,other=0.)
    f5=tl.div_rn(arc,tl.maximum(total,1e-6))
    tl.store(F+i*6,tl.where(valid,f0,0.),i<COUNT)
    tl.store(F+i*6+1,tl.where(valid,f1,0.),i<COUNT)
    tl.store(F+i*6+2,tl.where(valid,f2,0.),i<COUNT)
    tl.store(F+i*6+3,tl.where(valid,f3,0.),i<COUNT)
    tl.store(F+i*6+4,tl.where(valid,f4,0.),i<COUNT)
    tl.store(F+i*6+5,tl.where(valid,f5,0.),i<COUNT)


@triton.jit
def overlap_compact(SUM,TOTAL,OUT,COUNT:tl.constexpr,WIDTH:tl.constexpr,GLOBAL_WIDTH:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=i//WIDTH;j=i%WIDTH
    den=tl.load(TOTAL+row,i<COUNT,other=1.)
    right=tl.div_rn(tl.load(SUM+row*GLOBAL_WIDTH+j,i<COUNT,other=0.),den)
    left=tl.div_rn(tl.load(SUM+row*GLOBAL_WIDTH+j-1,(i<COUNT)&(j>0),other=0.),den)
    for k in tl.static_range(8):
        value=tl.maximum(tl.minimum(right,(k+1)/8.)-tl.maximum(left,k/8.),0.)
        tl.store(OUT+i*8+k,value,i<COUNT)


class Workspace:
    def __init__(self, model, width, batch):
        self.model, self.width, self.batch = model, width, batch
        device = next(model.parameters()).device
        def empty(*shape):
            return torch.empty(shape, device=device, dtype=torch.float32)
        self.empty = empty
        self.li, self.arc, self.mass, self.cumulative = [empty(batch, width) for _ in range(4)]
        self.total = empty(batch, 1)
        self.padding = torch.tanh(model.net[0].bias)
        self.locals = {}

    def local(self, local):
        if local not in self.locals:
            n = self.batch
            self.locals[local] = dict(points=self.empty(n,local,2), features=self.empty(n,local,6),
                hidden=self.empty(n,local,16), logits=self.empty(n,local),
                overlap=self.empty(n,local,8), summary=self.empty(n,8,2))
        return self.locals[local]

    def run(self, index, local, output):
        n, width = self.batch, self.width
        b = self.local(local)
        count = n*local
        lengths = index.offsets[1:]-index.offsets[:-1]
        # Only pointwise work is compacted; native reductions keep global width.
        self.li.zero_()
        geometry_compact[(triton.cdiv(count,256),)](index.points,index.offsets,b['points'],self.li,
            count,local,width,256,num_warps=4,enable_fp_fusion=False)
        torch.sum(self.li,dim=1,keepdim=True,out=self.total)
        torch.cumsum(self.li,dim=1,out=self.arc)
        features_compact[(triton.cdiv(count,256),)](index.points,index.offsets,self.li,self.total,self.arc,
            b['features'],count,local,width,256,num_warps=4,enable_fp_fusion=False)
        fused.first_layer[(n,triton.cdiv(local,64))](b['features'],lengths,
            self.model.net[0].weight,self.model.net[0].bias,self.padding,b['hidden'],local,64,
            num_warps=4,enable_fp_fusion=False)
        torch.addmm(self.model.net[2].bias,b['hidden'].reshape(-1,16),self.model.net[2].weight.t(),
                    out=b['logits'].reshape(-1,1))
        compact_mass[(triton.cdiv(n*width,256),)](b['logits'],lengths,self.mass,n*width,width,local,256,
            num_warps=4,enable_fp_fusion=False)
        torch.sum(self.mass,dim=1,keepdim=True,out=self.total)
        torch.cumsum(self.mass,dim=1,out=self.cumulative)
        overlap_compact[(triton.cdiv(count,256),)](self.cumulative,self.total,b['overlap'],count,
            local,width,256,num_warps=4,enable_fp_fusion=False)
        torch.bmm(b['overlap'].transpose(1,2),b['points'],out=b['summary'])
        torch.mul(b['summary'],8,out=output.reshape(n,8,2))


@torch.no_grad()
def encode(model,index,batch=512):
    if index.points.dtype != torch.float32 or len(index.offsets)-1 <= 512 or batch != 512:
        return dtw_routing._encode_reference(model,index,batch)
    # Transfer lengths once instead of synchronizing once per batch.
    lengths = np.diff(index.offsets.cpu().numpy())
    width = int(lengths.max())
    if width < 64:
        return fused.encode(model,index,batch)
    result = torch.empty((len(lengths),16),device=index.points.device,dtype=index.points.dtype)
    workspace = Workspace(model,width,batch)
    for lo in range(0,len(result),batch):
        sub = SimpleNamespace(points=index.points,offsets=index.offsets[lo:lo+batch+1])
        if len(sub.offsets)-1 != batch:
            result[lo:lo+batch] = fused.forward(model,*fused.prepare(sub,width),fast_first=False)
        else:
            local = min(width,max(64,triton.next_power_of_2(int(lengths[lo:lo+batch].max()))))
            workspace.run(sub,local,result[lo:lo+batch])
    return result
