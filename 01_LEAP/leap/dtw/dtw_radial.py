"""DTW radial mass bounds; the adopted signal mode freezes these filter probes."""
import torch
import triton
import triton.language as tl


def stats_torch(points, lengths, theta, bins=4):
    r = (points[:, :, None] - theta[None, None]).square().sum(-1).clamp_min(1e-24).sqrt()
    j = torch.arange(points.shape[1], device=points.device)[None]
    valid = j < lengths[:, None]
    low = r.masked_fill(~valid[..., None], torch.inf).amin(1)
    high = r.masked_fill(~valid[..., None], -torch.inf).amax(1)
    means = []
    for b in range(bins):
        mask = valid & ((j * bins) // lengths[:, None] == b)
        means.append((r * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None])
    return torch.stack([low, high] + means, -1)


def bound_torch(a, b, na, nb,guarded=True):
    # Each source point is charged at most once; Jensen gives a bucket-mean bound.
    bins = a.shape[-1] - 2
    ab = a[..., 0] * 0
    ba = b[..., 0] * 0
    for s in range(bins):
        ca = ((s+1)*na+bins-1)//bins - (s*na+bins-1)//bins
        cb = ((s+1)*nb+bins-1)//bins - (s*nb+bins-1)//bins
        ab = ab + ca[..., None] * torch.maximum(b[..., 0]-a[..., 2+s], a[..., 2+s]-b[..., 1]).clamp_min(0)
        ba = ba + cb[..., None] * torch.maximum(a[..., 0]-b[..., 2+s], b[..., 2+s]-a[..., 1]).clamp_min(0)
    lower=torch.maximum(ab.amax(-1), ba.amax(-1))
    if not guarded:
        return lower
    # Cancellation grows with radial scale and length, even when DTW is tiny.
    guard=64*torch.finfo(a.dtype).eps*torch.maximum(na,nb)*(a[...,1].amax(-1)+b[...,1].amax(-1))
    return (lower-guard).clamp_min(0)


@triton.jit
def encode_kernel(P,O,TH,OUT,M:tl.constexpr,BINS:tl.constexpr,N:tl.constexpr):
    row=tl.program_id(0); f=tl.program_id(1)
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


@torch.no_grad()
def encode(index,theta,bins=4):
    if index.points.dtype == torch.float32 and len(index.offsets) - 1 >= 4096:
        from leap.dtw._summary_encoding import radial
        return radial(index, theta, bins)
    return _encode_reference(index, theta, bins)


@torch.no_grad()
def _encode_reference(index,theta,bins=4):
    n=len(index.offsets)-1; m=len(theta)
    block=triton.next_power_of_2(int((index.offsets[1:]-index.offsets[:-1]).max()))
    out=torch.empty((n,m,bins+2),device=theta.device,dtype=index.points.dtype)
    encode_kernel[(n,m)](index.points,index.offsets,theta,out,m,bins,block,num_warps=4,enable_fp_fusion=False)
    return out


@triton.jit
def pair_kernel(A,B,AO,BO,Q,D,OUT,K,M:tl.constexpr,BINS:tl.constexpr,F:tl.constexpr):
    p=tl.program_id(0)*16+tl.arange(0,16); f=tl.arange(0,F)
    q=tl.load(Q+p,p<K,other=0); d=tl.load(D+p,p<K,other=0)
    na=tl.load(AO+q+1)-tl.load(AO+q)
    nb=tl.load(BO+d+1)-tl.load(BO+d)
    mask=(p[:,None]<K)&(f[None,:]<M)
    aq=(q[:,None]*M+f[None,:])*(BINS+2)
    bd=(d[:,None]*M+f[None,:])*(BINS+2)
    amin=tl.load(A+aq,mask,other=0.); amax=tl.load(A+aq+1,mask,other=0.)
    bmin=tl.load(B+bd,mask,other=0.); bmax=tl.load(B+bd+1,mask,other=0.)
    ab=tl.full((16,F),0.,A.dtype.element_ty); ba=tl.full((16,F),0.,A.dtype.element_ty)
    for s in range(BINS):
        am=tl.load(A+aq+2+s,mask,other=0.); bm=tl.load(B+bd+2+s,mask,other=0.)
        ca=((s+1)*na+BINS-1)//BINS-(s*na+BINS-1)//BINS
        cb=((s+1)*nb+BINS-1)//BINS-(s*nb+BINS-1)//BINS
        ab+=ca[:,None]*tl.maximum(0.,tl.maximum(bmin-am,am-bmax))
        ba+=cb[:,None]*tl.maximum(0.,tl.maximum(amin-bm,bm-amax))
    result=tl.maximum(tl.max(ab,1),tl.max(ba,1))
    guard=64.*1.1920928955078125e-7*tl.maximum(na,nb)*(tl.max(amax,1)+tl.max(bmax,1))
    result=tl.maximum(result-guard,0.)
    tl.store(OUT+p,result,p<K)


@torch.no_grad()
def pairs(a,b,ao,bo,q,d):
    out=torch.empty(q.shape,device=q.device,dtype=a.dtype)
    if len(q):
        pair_kernel[(triton.cdiv(len(q),16),)](a,b,ao,bo,q.contiguous(),d.contiguous(),out,len(q),a.shape[1],a.shape[2]-2,
                                               triton.next_power_of_2(a.shape[1]),num_warps=4,enable_fp_fusion=False)
    return out


@torch.no_grad()
def node_envelopes(stats,tree):
    members=tree.members
    s=stats[members.clamp_min(0)]
    valid=(members>=0)[...,None]
    low=s[...,0].masked_fill(~valid,torch.inf).amin(1)
    high=s[...,1].masked_fill(~valid,-torch.inf).amax(1)
    out=torch.empty((tree.groups*2,stats.shape[1],2),device=stats.device,dtype=stats.dtype)
    out[tree.groups:]=torch.stack([low,high],-1)
    level=tree.groups//2
    while level:
        ids=torch.arange(level,2*level,device=stats.device)
        out[ids,:,0]=torch.minimum(out[ids*2,:,0],out[ids*2+1,:,0])
        out[ids,:,1]=torch.maximum(out[ids*2,:,1],out[ids*2+1,:,1])
        level//=2
    return out.contiguous()
