"""Learnable radial interval bound for unbanded Euclidean-sum DTW.

L = <counts_A,u> + <counts_B,v> + DTW(G-u-v), with u,v>=0,
u_s+v_t<=G_st. G_st lower-bounds every raw point pair in the two blocks.
The weighted dual charge and residual ordered path never charge the same
part of a local cost twice. Probe positions may move without invalidating it.
"""
import torch
import triton
import triton.language as tl


def counts(lengths, bins=4):
    k=lengths.clamp_max(bins)
    s=torch.arange(bins,device=lengths.device)
    n=lengths[...,None];k=k[...,None]
    return torch.where(s<k,((s+1)*n+k-1)//k-(s*n+k-1)//k,0)


def stats_torch(points,lengths,theta,bins=4):
    radial=(points[:,:,None]-theta[None,None]).square().sum(-1).clamp_min(1e-24).sqrt()
    i=torch.arange(points.shape[1],device=points.device)[None]
    k=lengths.clamp_max(bins)
    block=i*k[:,None]//lengths[:,None]
    valid=i<lengths[:,None]
    output=[]
    for s in range(bins):
        mask=(valid&(block==s))[...,None]
        lo=radial.masked_fill(~mask,torch.inf).amin(1)
        hi=radial.masked_fill(~mask,-torch.inf).amax(1)
        exists=(s<k)[:,None]
        output.append(torch.stack([torch.where(exists,lo,0.),torch.where(exists,hi,0.)],-1))
    return torch.stack(output,1)


def small_dtw(cost):
    k=cost.shape[-1]
    inf=torch.full_like(cost[...,0,0],torch.inf)
    previous=[inf]*(k+1);previous[0]=torch.zeros_like(inf)
    for s in range(k):
        current=[inf]
        for t in range(k):
            current.append(cost[...,s,t]+torch.minimum(previous[t],torch.minimum(previous[t+1],current[t])))
        previous=current
    return previous[-1]


def bound_torch(a,b,na,nb,guarded=True,details=False):
    bins=a.shape[-3]
    alo,ahi=a[...,0],a[...,1]
    blo,bhi=b[...,0],b[...,1]
    g=torch.maximum(alo[..., :,None,:]-bhi[...,None,:,:],
                    blo[...,None,:,:]-ahi[..., :,None,:]).clamp_min(0).amax(-1)
    ca,cb=counts(na,bins),counts(nb,bins)
    valid=(ca[..., :,None]>0)&(cb[...,None,:]>0)
    g=torch.where(valid,g,0.)
    u=g.amin(-1)
    v=(g-u[..., :,None]).amin(-2).clamp_min(0)
    left=(ca*u).sum(-1)+(cb*v).sum(-1)+small_dtw((g-u[..., :,None]-v[...,None,:]).clamp_min(0))
    v2=g.amin(-2)
    u2=(g-v2[...,None,:]).amin(-1).clamp_min(0)
    right=(ca*u2).sum(-1)+(cb*v2).sum(-1)+small_dtw((g-u2[..., :,None]-v2[...,None,:]).clamp_min(0))
    pure=small_dtw(g)
    result=torch.maximum(pure,torch.maximum(left,right))
    if guarded:
        scale=ahi.amax((-1,-2))+bhi.amax((-1,-2))
        guard=64*torch.finfo(a.dtype).eps*(na+nb+2*bins)*scale
        result=(result-guard).clamp_min(0)
    if details:
        return result,{'gap':g,'row_dual':left,'col_dual':right,'ordered_only':pure,
                      'coverage_only':torch.maximum((ca*g.amin(-1)).sum(-1),(cb*g.amin(-2)).sum(-1))}
    return result


@triton.jit
def encode_kernel(P,O,TH,OUT,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr):
    row=tl.program_id(0);f=tl.program_id(1)
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
def encode(index,theta,bins=4):
    if index.points.dtype == torch.float32 and len(index.offsets) - 1 >= 4096:
        from leap.dtw._summary_encoding import ordered
        return ordered(index, theta, bins)
    return _encode_reference(index, theta, bins)


@torch.no_grad()
def _encode_reference(index,theta,bins=4):
    n=len(index.offsets)-1;m=len(theta)
    block=triton.next_power_of_2(int((index.offsets[1:]-index.offsets[:-1]).max()))
    out=torch.empty((n,bins,m,2),device=theta.device,dtype=index.points.dtype)
    encode_kernel[(n,m)](index.points,index.offsets,theta,out,m,bins,block,num_warps=4,enable_fp_fusion=False)
    return out


@triton.jit
def grid_dtw(G,K:tl.constexpr):
    j=tl.arange(0,K)
    prev=tl.full((K,),float('inf'),G.dtype)
    for s in tl.static_range(K):
        current=tl.full((K,),float('inf'),G.dtype)
        left=tl.full((),float('inf'),G.dtype)
        cross=tl.full((),float('inf'),G.dtype)
        for t in tl.static_range(K):
            up=tl.sum(tl.where(j==t,prev,0.),0)
            c=tl.sum(tl.sum(tl.where((j[:,None]==s)&(j[None,:]==t),G,0.),1),0)
            pred=tl.minimum(cross,tl.minimum(up,left))
            if s==0 and t==0:pred=tl.full((),0.,G.dtype)
            value=c+pred
            current=tl.where(j==t,value,current)
            left=value;cross=up
        prev=current
    return left


@triton.jit
def pair_kernel(A,B,AO,BO,Q,D,OUT,M:tl.constexpr,K:tl.constexpr,LIMIT,PRUNE:tl.constexpr=False):
    lane=tl.program_id(0)
    q=tl.load(Q+lane);d=tl.load(D+lane)
    na=tl.load(AO+q+1)-tl.load(AO+q);nb=tl.load(BO+d+1)-tl.load(BO+d)
    s=tl.arange(0,K);f=tl.arange(0,M)
    ax=((q*K+s[:,None])*M+f[None,:])*2
    bx=((d*K+s[:,None])*M+f[None,:])*2
    alo=tl.load(A+ax);ahi=tl.load(A+ax+1)
    blo=tl.load(B+bx);bhi=tl.load(B+bx+1)
    gap=tl.maximum(0.,tl.maximum(alo[:,None,:]-bhi[None,:,:],blo[None,:,:]-ahi[:,None,:]))
    g=tl.max(gap,2)
    ka=tl.minimum(na,K);kb=tl.minimum(nb,K)
    ca=tl.where(s<ka,((s+1)*na+ka-1)//ka-(s*na+ka-1)//ka,0)
    cb=tl.where(s<kb,((s+1)*nb+kb-1)//kb-(s*nb+kb-1)//kb,0)
    g=tl.where((ca[:,None]>0)&(cb[None,:]>0),g,0.)
    u=tl.min(g,1);v=tl.maximum(tl.min(g-u[:,None],0),0.)
    left=tl.sum(ca*u,0)+tl.sum(cb*v,0)+grid_dtw(tl.maximum(g-u[:,None]-v[None,:],0.),K)
    scale=tl.max(tl.max(ahi,1),0)+tl.max(tl.max(bhi,1),0)
    guard=64.*1.1920928955078125e-7*(na+nb+2*K)*scale
    limit=tl.full((),float('inf'),A.dtype.element_ty)
    if PRUNE:
        limit=tl.load(LIMIT+lane)
    result=left
    # One component above the guarded threshold already proves rejection.
    if not PRUNE or tl.maximum(left-guard,0.)<=limit:
        v2=tl.min(g,0);u2=tl.maximum(tl.min(g-v2[None,:],1),0.)
        right=tl.sum(ca*u2,0)+tl.sum(cb*v2,0)+grid_dtw(tl.maximum(g-u2[:,None]-v2[None,:],0.),K)
        result=tl.maximum(grid_dtw(g,K),tl.maximum(left,right))
    lower=tl.maximum(result-guard,0.)
    if PRUNE:
        tl.store(OUT+lane,lower<=limit)
    else:
        tl.store(OUT+lane,lower)


@torch.no_grad()
def pairs(a,b,ao,bo,q,d):
    out=torch.empty(q.shape,device=q.device,dtype=a.dtype)
    if len(q):
        pair_kernel[(len(q),)](a,b,ao,bo,q.contiguous(),d.contiguous(),out,a.shape[-2],a.shape[-3],out,
                                           num_warps=4,enable_fp_fusion=False)
    return out


@torch.no_grad()
def keep_pairs(a,b,ao,bo,q,d,limits):
    """Match pairs(...) <= limits without computing unused rejected scores."""
    if limits.shape!=q.shape or limits.device!=q.device or limits.dtype!=a.dtype:
        raise ValueError('One same-device, same-dtype threshold is required per pair')
    out=torch.empty(q.shape,device=q.device,dtype=torch.bool)
    if len(q):
        pair_kernel[(len(q),)](a,b,ao,bo,q.contiguous(),d.contiguous(),out,a.shape[-2],a.shape[-3],
                               limits.contiguous(),True,num_warps=4,enable_fp_fusion=False)
    return out
