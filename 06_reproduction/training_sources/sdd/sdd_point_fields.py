"""SDD point-set Hausdorff instantiation of the archived probe learner."""
import torch
import triton
import triton.language as tl


@triton.jit
def directed(P,OFF,a,b):
    a0=tl.load(OFF+a);a1=tl.load(OFF+a+1)
    b0=tl.load(OFF+b);b1=tl.load(OFF+b+1)
    i=tl.arange(0,32);j=tl.arange(0,32)
    far=tl.full((),-1.,tl.float32)
    witness=tl.full((),2147483647,tl.int64)
    for start in range(a0,a1,32):
        valid=start+i<a1
        x=tl.load(P+(start+i)*2,valid,other=0.)
        y=tl.load(P+(start+i)*2+1,valid,other=0.)
        near=tl.full((32,),float('inf'),tl.float32)
        for end in range(b0,b1,32):
            mask=end+j<b1
            u=tl.load(P+(end+j)*2,mask,other=0.)
            v=tl.load(P+(end+j)*2+1,mask,other=0.)
            dx=x[:,None]-u[None,:];dy=y[:,None]-v[None,:]
            near=tl.minimum(near,tl.min(tl.where(mask[None,:],dx*dx+dy*dy,float('inf')),1))
        block=tl.max(tl.where(valid,near,-1.),0)
        wi=tl.min(tl.where(valid&(near==block),start+i,2147483647),0)
        witness=tl.where(block>far,wi,tl.where(block==far,tl.minimum(witness,wi),witness))
        far=tl.maximum(far,block)
    return far,witness


@triton.jit
def kernel(P,OFF,Q,D,OUT,W):
    row=tl.program_id(0);q=tl.load(Q+row);d=tl.load(D+row)
    f,w=directed(P,OFF,q,d);g,v=directed(P,OFF,d,q)
    h=tl.sqrt(tl.maximum(f,g))
    tl.store(OUT+row*2,h);tl.store(OUT+row*2+1,h)
    tl.store(W+row,tl.where(f>=g,w,v))


@torch.no_grad()
def pair_stats(index,selected,queries,databases):
    out=torch.empty((len(queries),2),device=index.points.device)
    witness=torch.empty(len(queries),device=index.points.device,dtype=torch.int32)
    kernel[(len(queries),)](index.points,index.offsets,queries.contiguous(),databases.contiguous(),
                            out,witness,num_warps=4,enable_fp_fusion=False)
    return out,witness


def install():
    from leap.common import dual_field
    from leap.common import raw
    from leap.hausdorff import range_search
    from leap.common.raw_geometry_refinement import hausdorff_distances
    original=dual_field.fields_packed
    def packed(points,offsets,anchors):
        a=original(points,offsets,anchors)[:,:len(anchors)]
        return torch.cat((a,a),-1).contiguous()
    def differentiable(points,lengths,anchors):
        # Local chunks retain the same scalar field and gradient on long inputs.
        fields=[]
        for lo in range(0,len(points),16):
            p=points[lo:lo+16]
            n=lengths[lo:lo+16]
            width=int(n.max())
            d=(p[:,:width,None]-anchors[None,None]).square().sum(-1)
            mask=torch.arange(width,device=p.device)[None]<n[:,None]
            a=d.masked_fill(~mask[:,:,None],torch.inf).amin(1).clamp_min(1e-24).sqrt()
            fields.append(torch.cat((a,a),-1))
        return torch.cat(fields)
    def pairs(q,d,qi,di,point=True):
        return hausdorff_distances(q,d,qi,di,point=True)
    def candidates(q,d,ids):
        qi=torch.arange(len(ids),device=ids.device)[:,None].expand_as(ids)
        return pairs(q,d,qi.reshape(-1),ids.reshape(-1)).reshape_as(ids)
    dual_field.fields_packed=packed
    dual_field.fields_torch=differentiable
    raw.fused_raw_scores=candidates
    range_search.hausdorff_distances=pairs
    # Placement imports this alias directly for its train-only search radius.
    range_search.pair_values=pairs
    return packed
