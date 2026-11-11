"""Four disjoint subtree waves with exact Top-k radius feedback."""
import torch
import triton
import triton.language as tl

from leap.common.interval_index import lex_keys
from leap.common.range_buffers import enumerate_candidates
from leap.dfd.dfd_tree import endpoints, native_enum
from leap.dfd.dfd_refinement import cached_pairs
from leap.dtw.dtw_ragged import topk as ragged_topk


@triton.jit
def root_bounds(Q, VERT, EQ, ENDS, OUT, R: tl.constexpr):
    row=tl.program_id(0);root=R+tl.arange(0,R);f=tl.arange(0,64)
    qa=tl.load(Q+row*128+f)
    high=tl.load(VERT+root[:,None]*128+f[None,:])
    low=tl.load(VERT+root[:,None]*128+64+f[None,:])
    bound=tl.maximum(0.,tl.max(tl.maximum(qa[None,:]-high,low-qa[None,:]),1))
    for end in range(2):
        x=tl.load(EQ+row*4+end*2);y=tl.load(EQ+row*4+end*2+1)
        ux=tl.load(ENDS+root*8+end*2);uy=tl.load(ENDS+root*8+end*2+1)
        lx=tl.load(ENDS+root*8+4+end*2);ly=tl.load(ENDS+root*8+4+end*2+1)
        dx=tl.maximum(0.,tl.maximum(x-ux,lx-x));dy=tl.maximum(0.,tl.maximum(y-uy,ly-y))
        bound=tl.maximum(bound,tl.sqrt(dx*dx+dy*dy))
    tl.store(OUT+row*R+tl.arange(0,R),bound)


@triton.jit
def remove_seed_duplicates(VALUES, ROWS, IDS, SEEDS, N: tl.constexpr, S: tl.constexpr, C: tl.constexpr):
    p=tl.program_id(0)*32+tl.arange(0,32);valid=p<N;f=tl.arange(0,C)
    row=tl.load(ROWS+p,valid,other=0);d=tl.load(IDS+p,valid,other=-1)
    seeds=tl.load(SEEDS+row[:,None]*S+f[None,:],valid[:,None]&(f[None,:]<S),other=-2)
    hit=tl.sum((seeds==d[:,None]).to(tl.int32),1)>0
    tl.store(VALUES+p,float('inf'),valid&hit)


class SubtreeEnumeration:
    def __init__(self, roots):
        self.roots=roots

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return native_enum[grid](*args,ROOTS=self.roots,**kwargs)
        return launch


@torch.no_grad()
def search(query, database, tree, qids, seeds, top_k=20, *, raw_initial=None):
    from leap.dfd.dfd_range import initial_distances
    raw=initial_distances(query,database,seeds) if raw_initial is None else raw_initial
    if raw.shape!=seeds.shape or raw.device!=query.points.device or raw.dtype!=query.points.dtype:
        raise ValueError('Initial distances must match seed shape, geometry dtype and device')
    n=len(seeds);waves=min(4,tree.groups)
    initial_radius=raw.topk(top_k,largest=False).values[:,-1].contiguous()
    radius=initial_radius
    best=lex_keys(raw,seeds).topk(top_k,largest=False).values
    slack=(2e-5*(1+query.coordinates.abs().amax(1)+tree.database_scale)).contiguous()
    ep=endpoints(query);extra=tree.dfd_geometry
    lower=torch.empty((n,waves),device=raw.device,dtype=raw.dtype)
    root_bounds[(n,)](query.coordinates,extra['native'][0],ep,extra['native'][1],lower,
                     waves,num_warps=4,enable_fp_fusion=False)
    order=lower.argsort(dim=1,stable=True)
    lower=lower.gather(1,order);order=order+waves
    counts=torch.zeros(n,device=raw.device,dtype=torch.int32)
    visits=torch.zeros((n,2),device=raw.device,dtype=torch.int32)
    for wave in range(waves):
        roots=torch.where(lower[:,wave]<=radius+slack,order[:,wave],0).contiguous()
        args=(query.coordinates,database.coordinates,ep,extra['endpoints'],tree.members,
              *extra['native'],radius,qids,slack)
        # Ragged Top-k has no n*maximum allocation; retain the actual-pair cap.
        count,visited,ids,rows,offsets,_=enumerate_candidates(
            SubtreeEnumeration(roots),args,n,raw.device,tree.groups,tree.block,0,
            n*8_000_000,max_pairs=8_000_000)
        values,_=cached_pairs(query,database,rows,ids,seeds,raw,(radius+slack).contiguous())
        if len(ids):
            remove_seed_duplicates[(triton.cdiv(len(ids),32),)](
                values,rows,ids,seeds,len(ids),seeds.shape[1],triton.next_power_of_2(seeds.shape[1]),num_warps=4)
        old=(best>>32).int().contiguous().view(torch.float32)
        best=ragged_topk(best&0xffffffff,old,ids,values,offsets,count,top_k)
        radius=(best[:,-1]>>32).int().contiguous().view(torch.float32)
        counts+=count;visits+=visited
    return dict(indices=best&0xffffffff,distances=(best>>32).int().contiguous().view(torch.float32),
                counts=counts,radius=initial_radius,visits=visits)
