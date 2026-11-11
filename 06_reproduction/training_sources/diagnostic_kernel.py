"""Exact raw and selected-source-vertex diagnostics; not a retrieval model."""
import torch
import triton
import triton.language as tl


@triton.jit
def directed(P, OFF, SELECTED, a, b):
    a0=tl.load(OFF+a); a1=tl.load(OFF+a+1)
    b0=tl.load(OFF+b); b1=tl.load(OFF+b+1)
    i=tl.arange(0,32); j=tl.arange(0,32)
    far=tl.full((),-1,tl.float32)
    snap=tl.full((),0,tl.float32)
    witness=tl.full((),2147483647,tl.int64)
    for start in range(a0,a1,32):
        valid=start+i<a1
        x=tl.load(P+(start+i)*2,valid,other=0)
        y=tl.load(P+(start+i)*2+1,valid,other=0)
        near=tl.full((32,),float('inf'),tl.float32)
        for end in range(b0,b1-1,32):
            seg=end+j<b1-1
            ax=tl.load(P+(end+j)*2,seg,other=0)
            ay=tl.load(P+(end+j)*2+1,seg,other=0)
            bx=tl.load(P+(end+j+1)*2,seg,other=0)
            by=tl.load(P+(end+j+1)*2+1,seg,other=0)
            ex=bx-ax; ey=by-ay
            rx=x[:,None]-ax[None,:]; ry=y[:,None]-ay[None,:]
            u=tl.div_rn(rx*ex[None,:]+ry*ey[None,:],tl.maximum(ex*ex+ey*ey,1e-30)[None,:])
            u=tl.minimum(1.,tl.maximum(0.,u))
            dx=rx-u*ex[None,:]; dy=ry-u*ey[None,:]
            near=tl.minimum(near,tl.min(tl.where(seg[None,:],dx*dx+dy*dy,float('inf')),1))
        block=tl.max(tl.where(valid,near,-1),0)
        wi=tl.min(tl.where(valid&(near==block),start+i,2147483647),0)
        witness=tl.where(block>far,wi,tl.where(block==far,tl.minimum(witness,wi),witness))
        far=tl.maximum(far,block)
        chosen=tl.load(SELECTED+start+i,valid,other=0)
        snap=tl.maximum(snap,tl.max(tl.where(valid&(chosen>0),near,0),0))
    return far,snap,witness


@triton.jit
def pair_kernel(P,OFF,SELECTED,Q,D,O,W):
    row=tl.program_id(0)
    q=tl.load(Q+row); d=tl.load(D+row)
    f,s,w=directed(P,OFF,SELECTED,q,d)
    g,t,v=directed(P,OFF,SELECTED,d,q)
    tl.store(O+row*2,tl.sqrt(tl.maximum(f,g)))
    tl.store(O+row*2+1,tl.sqrt(tl.maximum(s,t)))
    tl.store(W+row,tl.where(f>=g,w,v))


@torch.no_grad()
def pair_stats(index, selected, queries, databases):
    out=torch.empty((len(queries),2),device='cuda',dtype=torch.float32)
    witnesses=torch.empty(len(queries),device='cuda',dtype=torch.int32)
    pair_kernel[(len(queries),)](index.points,index.offsets,selected,queries.contiguous(),databases.contiguous(),out,witnesses,num_warps=4,enable_fp_fusion=False)
    return out,witnesses
