"""Isolated contractive radial, elliptical and single-harmonic probe fields."""
import math
import numpy as np
import torch
import triton
import triton.language as tl
from leap.common.dual_field import fields_packed, fields_torch
from leap.common.model import TrajectoryIndex
from leap.common.data import checked_pack
from leap.common.raw import normalize_points


def sine_lower(phase0, phase1, amplitude):
    low,high=torch.minimum(phase0,phase1),torch.maximum(phase0,phase1)
    extremum=torch.where(amplitude>=0,1.5*math.pi,.5*math.pi)
    crosses=torch.ceil((low-extremum)/(2*math.pi))<=torch.floor((high-extremum)/(2*math.pi))
    endpoint=torch.minimum(amplitude*torch.sin(phase0),amplitude*torch.sin(phase1))
    return torch.where(crosses,-amplitude.abs(),endpoint)


class LearnedFields(torch.nn.Module):
    def __init__(self,anchors,mean,std,mode):
        super().__init__()
        if mode not in ('radial','ellipse','fourier'):
            raise ValueError(mode)
        self.mode=mode
        self.mean=np.asarray(mean,np.float64)
        self.std=np.asarray(std,np.float64)
        self.anchors=torch.nn.Parameter(torch.as_tensor(anchors,dtype=torch.float32,device='cuda').clone())
        angles=torch.arange(64,device='cuda',dtype=torch.float32)*(math.pi*(3-math.sqrt(5)))
        if mode=='ellipse':
            self.angle=torch.nn.Parameter(angles)
            self.axis=torch.nn.Parameter(torch.ones(64,device='cuda'))
        if mode=='fourier':
            frequencies=(2*math.pi)/torch.tensor([1.,2.,4.],device='cuda')[torch.arange(64,device='cuda')%3]
            self.omega=torch.nn.Parameter(torch.stack((angles.cos(),angles.sin()),-1)*frequencies[:,None])
            self.phase=torch.nn.Parameter(torch.remainder(angles,2*math.pi))
            self.amplitude=torch.nn.Parameter(torch.zeros(64,device='cuda'))

    def forward(self,points,lengths):
        if self.mode=='radial':
            return fields_torch(points,lengths,self.anchors)
        r=self.anchors[None,None]-points[:,:,None]
        edge=points[:,1:]-points[:,:-1]
        if self.mode=='ellipse':
            c,s=self.angle.cos(),self.angle.sin()
            def rotate(v):
                return torch.stack((v[...,0]*c+v[...,1]*s,(-v[...,0]*s+v[...,1]*c)*self.axis),-1)
            r=rotate(r)
            e=rotate(edge[:,:,None])
        else:
            e=edge[:,:,None]
        u=(r[:,:-1]*e).sum(-1)/e.square().sum(-1).clamp_min(1e-30)
        vertex=r.square().sum(-1).clamp_min(1e-24).sqrt()
        segment=(r[:,:-1]-u.clamp(0,1)[...,None]*e).square().sum(-1).clamp_min(1e-24).sqrt()
        if self.mode=='fourier':
            phase=-(r*self.omega).sum(-1)+self.phase
            # Vector norm uses a finite zero subgradient when a frequency vanishes.
            denominator=1+self.amplitude.abs()*self.omega.norm(dim=-1)
            vertex=(vertex+self.amplitude*phase.sin())/denominator
            segment=(segment+sine_lower(phase[:,:-1],phase[:,1:],self.amplitude))/denominator
        valid=torch.arange(points.shape[1],device=points.device)[None]<lengths[:,None]
        seg_valid=torch.arange(points.shape[1]-1,device=points.device)[None]<lengths[:,None]-1
        return torch.cat((vertex.masked_fill(~valid[:,:,None],torch.inf).amin(1),
                          segment.masked_fill(~seg_valid[:,:,None],torch.inf).amin(1)),-1)

    @torch.no_grad()
    def project(self,low,high):
        self.anchors.copy_(self.anchors.clamp(low,high))
        if self.mode=='ellipse':
            self.axis.clamp_(.25,1.)
        if self.mode=='fourier':
            self.amplitude.clamp_(-.05,.05)
            norms=self.omega.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            self.omega.mul_((8./norms).clamp_max(1.))

    @torch.no_grad()
    def packed(self,points,offsets):
        if self.mode=='radial':
            return fields_packed(points,offsets,self.anchors)
        if self.mode=='ellipse':
            parameters=torch.stack((self.angle.cos(),self.angle.sin(),self.axis),-1).contiguous()
        else:
            denominator=1+self.amplitude.abs()*self.omega.norm(dim=-1)
            parameters=torch.stack((self.amplitude,self.omega[:,0],self.omega[:,1],self.phase,denominator),-1).contiguous()
        out=torch.empty((len(offsets)-1,128),device=points.device)
        learned_kernel[((len(offsets)-1)*64,)](points,offsets,self.anchors,parameters,out,
                                              self.mode,num_warps=4,enable_fp_fusion=False)
        return out

    @torch.no_grad()
    def encode(self,paths):
        points,offsets=checked_pack(paths)
        p=torch.from_numpy(normalize_points(points,self.mean,self.std)).cuda()
        o=torch.from_numpy(offsets).cuda()
        z=self.packed(p,o)
        assert torch.isfinite(z).all()
        return TrajectoryIndex(z,p,o,'learned-field:'+self.mode)

    def payload(self):
        return {name:value.detach().cpu().tolist() for name,value in self.state_dict().items()}

    def restore(self,state):
        self.load_state_dict({k:torch.as_tensor(v,dtype=torch.float32,device='cuda') for k,v in state.items()})


@triton.jit
def learned_kernel(P,OFF,C,PARAM,OUT,MODE:tl.constexpr):
    pair=tl.program_id(0)
    row=pair//64;j=pair%64
    begin=tl.load(OFF+row);end=tl.load(OFF+row+1)
    cx=tl.load(C+j*2);cy=tl.load(C+j*2+1)
    if MODE=='ellipse':
        cosine=tl.load(PARAM+j*3);sine=tl.load(PARAM+j*3+1);axis=tl.load(PARAM+j*3+2)
    else:
        amplitude=tl.load(PARAM+j*5)
        wx=tl.load(PARAM+j*5+1);wy=tl.load(PARAM+j*5+2)
        phase=tl.load(PARAM+j*5+3);denominator=tl.load(PARAM+j*5+4)
    lane=tl.arange(0,32)
    va=tl.full((),float('inf'),tl.float32)
    vb=tl.full((),float('inf'),tl.float32)
    for start in range(begin,end,32):
        valid=start+lane<end
        seg=start+lane+1<end
        x=tl.load(P+(start+lane)*2,valid,other=0.)
        y=tl.load(P+(start+lane)*2+1,valid,other=0.)
        nx=tl.load(P+(start+lane+1)*2,seg,other=0.)
        ny=tl.load(P+(start+lane+1)*2+1,seg,other=0.)
        rx=cx-x;ry=cy-y;ex=nx-x;ey=ny-y
        if MODE=='ellipse':
            tx=cosine*rx+sine*ry;ty=(-sine*rx+cosine*ry)*axis
            te=cosine*ex+sine*ey;tf=(-sine*ex+cosine*ey)*axis
            rx=tx;ry=ty;ex=te;ey=tf
        u=tl.div_rn(rx*ex+ry*ey,tl.maximum(ex*ex+ey*ey,1e-30))
        u=tl.minimum(1.,tl.maximum(0.,u))
        dx=rx-u*ex;dy=ry-u*ey
        vertex=tl.sqrt(tl.maximum(rx*rx+ry*ry,1e-24))
        segment=tl.sqrt(tl.maximum(dx*dx+dy*dy,1e-24))
        if MODE=='fourier':
            p0=-(rx*wx+ry*wy)+phase
            p1=-((cx-nx)*wx+(cy-ny)*wy)+phase
            t0=amplitude*tl.sin(p0);t1=amplitude*tl.sin(p1)
            extremum=tl.where(amplitude>=0,4.71238898038469,1.5707963267948966)
            lo=tl.minimum(p0,p1);hi=tl.maximum(p0,p1)
            first=-tl.floor(-(lo-extremum)/6.283185307179586)
            last=tl.floor((hi-extremum)/6.283185307179586)
            min_sine=tl.where(first<=last,-tl.abs(amplitude),tl.minimum(t0,t1))
            vertex=(vertex+t0)/denominator
            segment=(segment+min_sine)/denominator
        va=tl.minimum(va,tl.min(tl.where(valid,vertex,float('inf')),0))
        vb=tl.minimum(vb,tl.min(tl.where(seg,segment,float('inf')),0))
    tl.store(OUT+row*128+j,va)
    tl.store(OUT+row*128+64+j,vb)
