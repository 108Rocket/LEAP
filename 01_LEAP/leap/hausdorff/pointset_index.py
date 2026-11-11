"""Point-set HD near/far fields and exact bounded-memory search kernels."""
from dataclasses import replace
import numpy as np
import torch
import triton
import triton.language as tl
from leap.common.model import TrajectoryIndex
from leap.common.data import checked_pack
from leap.common.raw import normalize_points
from leap.common.interval_index import partition, lex_keys
from leap.common.range_buffers import enumerate_candidates, top_keys
from leap.common.raw_geometry_refinement import hausdorff_distances
from leap.hausdorff.cached_refinement import cached_pairs
from leap.common.candidate_policy import MAX_TREE_CANDIDATES


def joint_partition(features, block=64):
    """Median split over both near and far fields, retaining original row IDs."""
    depth=max(0,int(np.ceil(np.log2(max(1,len(features)/block)))))
    leaves=[]
    def split(ids,level):
        if level==depth:
            leaves.append(ids)
            return
        axis=np.ptp(features[ids],axis=0).argmax()
        order=np.argsort(features[ids,axis],kind='stable')
        middle=len(ids)//2
        split(ids[order[:middle]],level+1)
        split(ids[order[middle:]],level+1)
    split(np.arange(len(features)),0)
    members=np.full((len(leaves),block),-1,np.int64)
    for i,ids in enumerate(leaves):
        members[i,:len(ids)]=ids
    return members


@triton.jit
def field_kernel(P, O, C, Z, NF: tl.constexpr, M: tl.constexpr, PB: tl.constexpr=8):
    row=tl.program_id(0)
    f=tl.program_id(1)*PB+tl.arange(0,PB)
    lane=tl.arange(0,32)
    lo=tl.load(O+row); hi=tl.load(O+row+1)
    cx=tl.load(C+f*2,f<M,other=0.); cy=tl.load(C+f*2+1,f<M,other=0.)
    near=tl.full((PB,),float('inf'),tl.float32)
    far=tl.full((PB,),0.,tl.float32)
    for start in range(lo,hi,32):
        good=start+lane<hi
        x=tl.load(P+(start+lane)*2,good,other=0.)
        y=tl.load(P+(start+lane)*2+1,good,other=0.)
        dx=x[:,None]-cx[None,:]; dy=y[:,None]-cy[None,:]
        d=dx*dx+dy*dy
        near=tl.minimum(near,tl.min(tl.where(good[:,None],d,float('inf')),0))
        if NF:
            far=tl.maximum(far,tl.max(tl.where(good[:,None],d,0.),0))
    mn=tl.sqrt(tl.maximum(near,1e-24))
    mx=tl.sqrt(tl.maximum(far,1e-24)) if NF else mn
    tl.store(Z+row*(2*M)+f,mn,f<M)
    tl.store(Z+row*(2*M)+M+f,mx,f<M)


def fields(points,offsets,anchors,nearfar=True):
    out=torch.empty((len(offsets)-1,2*len(anchors)),device=points.device,dtype=torch.float32)
    field_kernel[(len(out),triton.cdiv(len(anchors),8))](points,offsets,anchors,out,nearfar,len(anchors),num_warps=4,enable_fp_fusion=False)
    return out


def prepare(paths,model):
    p,o=checked_pack(paths)
    p=normalize_points(p,model.mean,model.std)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite normalized geometry')
    return torch.from_numpy(p).to(model.device),torch.from_numpy(o).to(model.device)


def index_from_geometry(p,o,model,nearfar=True):
    z=fields(p,o,model.anchors,nearfar)
    if not torch.isfinite(z).all():
        raise ValueError('Nonfinite fields')
    return TrajectoryIndex(z,p,o,model.asset_id+(':nearfar-v1' if nearfar else ':near-v1'))


def subset(index,lo,hi):
    return TrajectoryIndex(index.coordinates[lo:hi],index.points,index.offsets[lo:hi+1],index.asset_id)


@triton.jit
def leaf_kernel(D, MEM, TREE, G:tl.constexpr, B:tl.constexpr, DIM:tl.constexpr):
    group=tl.program_id(0)
    f=tl.program_id(1)*32+tl.arange(0,32)
    lane=tl.arange(0,B)
    ids=tl.load(MEM+group*B+lane)
    z=tl.load(D+ids[:,None]*128+f[None,:],ids[:,None]>=0,other=0.)
    low=tl.min(tl.where(ids[:,None]>=0,z,float('inf')),0)
    high=tl.max(tl.where(ids[:,None]>=0,z,-float('inf')),0)
    node=group+G
    tl.store(TREE+node*(2*DIM)+f,low)
    tl.store(TREE+node*(2*DIM)+DIM+f,high)


@triton.jit
def parent_kernel(TREE, START:tl.constexpr, DIM:tl.constexpr):
    node=START+tl.program_id(0)
    f=tl.program_id(1)*32+tl.arange(0,32)
    l0=tl.load(TREE+(node*2)*(2*DIM)+f)
    l1=tl.load(TREE+(node*2+1)*(2*DIM)+f)
    u0=tl.load(TREE+(node*2)*(2*DIM)+DIM+f)
    u1=tl.load(TREE+(node*2+1)*(2*DIM)+DIM+f)
    tl.store(TREE+node*(2*DIM)+f,tl.minimum(l0,l1))
    tl.store(TREE+node*(2*DIM)+DIM+f,tl.maximum(u0,u1))


@triton.jit
def node_bound(Q,TREE,q,node,DIM:tl.constexpr):
    f=tl.arange(0,DIM)
    value=tl.load(Q+q*128+f)
    low=tl.load(TREE+node[:,None]*(2*DIM)+f[None,:])
    high=tl.load(TREE+node[:,None]*(2*DIM)+DIM+f[None,:])
    return tl.maximum(0.,tl.max(tl.maximum(low-value[None,:],value[None,:]-high),1))


@triton.jit
def member_bound(Q,D,q,ids,valid,B:tl.constexpr,DIM:tl.constexpr):
    f=tl.arange(0,32)
    score=tl.full((B,),0.,tl.float32)
    # Bound tile size limits live per-candidate temporaries.
    for start in range(0,DIM,32):
        query=tl.load(Q+q*128+start+f)
        value=tl.load(D+ids[:,None]*128+start+f[None,:],valid[:,None],other=0.)
        score=tl.maximum(score,tl.max(tl.abs(value-query[None,:]),1))
    return score


@triton.jit
def member_keep(Q,D,q,ids,valid,threshold,B:tl.constexpr,DIM:tl.constexpr):
    f=tl.arange(0,32)
    alive=valid
    for start in range(0,DIM,32):
        if tl.sum(alive.to(tl.int32),0)>0:
            query=tl.load(Q+q*128+start+f)
            value=tl.load(D+ids[:,None]*128+start+f[None,:],alive[:,None],other=0.)
            partial=tl.max(tl.abs(value-query[None,:]),1)
            alive=alive&(partial<=threshold)
    return alive


@triton.jit(do_not_specialize=['K'])
def candidate_kernel(Q,D,MEM,TREE,QIDS,SLACK,OUT,VISITS,
                     G:tl.constexpr,B:tl.constexpr,K,DIM:tl.constexpr,C:tl.constexpr=0,
                     HAS_AFTER:tl.constexpr=False,AFTER=None,FAST:tl.constexpr=False):
    row=tl.program_id(0)
    WIDTH:tl.constexpr=B if C==0 else C
    retained=tl.arange(0,WIDTH)
    lane=tl.arange(0,B); slots=tl.arange(0,32); sides=tl.arange(0,2)
    self_id=tl.load(QIDS+row); slack=tl.load(SLACK+row)
    inf=tl.full((),9187343244130779135,tl.int64)
    best=tl.full((WIDTH,),9187343244130779135,tl.int64)
    stack=tl.where(slots==0,1,0); stack_lb=tl.full((32,),0.,tl.float32)
    sp=0; threshold=tl.full((),float('inf'),tl.float32)
    cutoff=inf
    nodes=0; leaves=0; examined=0
    f=tl.arange(0,128)
    finite=tl.max(tl.abs(tl.load(Q+row*128+f)),0)<float('inf')
    while (sp>=0)&finite:
        node=tl.sum(tl.where(slots==sp,stack,0),0)
        lower=tl.min(tl.where(slots==sp,stack_lb,float('inf')),0)
        sp-=1
        if lower<=threshold:
            nodes+=1
            if node<G:
                children=node*2+sides
                bound=node_bound(Q,TREE,row,children,DIM)-slack
                l0=tl.sum(tl.where(sides==0,bound,0.),0)
                l1=tl.sum(tl.where(sides==1,bound,0.),0)
                near=tl.where(l0<=l1,node*2,node*2+1)
                far=tl.where(l0<=l1,node*2+1,node*2)
                if tl.maximum(l0,l1)<=threshold:
                    sp+=1; stack=tl.where(slots==sp,far,stack)
                    stack_lb=tl.where(slots==sp,tl.maximum(l0,l1),stack_lb)
                if tl.minimum(l0,l1)<=threshold:
                    sp+=1; stack=tl.where(slots==sp,near,stack)
                    stack_lb=tl.where(slots==sp,tl.minimum(l0,l1),stack_lb)
            else:
                ids=tl.load(MEM+(node-G)*B+lane)
                valid=(ids>=0)&(ids!=self_id)
                if FAST:
                    score=tl.full((B,),0.,tl.float32)
                    alive=valid
                    chunk=tl.arange(0,32)
                    for start in range(0,DIM,32):
                        if tl.sum(alive.to(tl.int32),0)>0:
                            value=tl.load(D+ids[:,None]*128+start+chunk[None,:],alive[:,None],other=0.)
                            query=tl.load(Q+row*128+start+chunk)
                            score=tl.maximum(score,tl.max(tl.abs(value-query[None,:]),1))
                            alive=alive&(score<=threshold)
                    score=tl.where(alive,score,float('inf'))
                else:
                    score=member_bound(Q,D,row,ids,valid,B,DIM)
                key=(score.to(tl.int32,bitcast=True).to(tl.int64)<<32)|ids.to(tl.int64)
                key=tl.where(valid,key,inf)
                if HAS_AFTER:
                    key=tl.where(key>tl.load(AFTER+row),key,inf)
                if WIDTH>B:
                    key=tl.reshape(tl.where(tl.arange(0,WIDTH//B)[:,None]==0,key[None,:],inf),(WIDTH,))
                # One bitonic split discards the larger half before sorting.
                if not FAST or tl.min(key,0)<=cutoff:
                    reverse=tl.sort(key,descending=True)
                    best=tl.sort(tl.minimum(best,reverse),descending=False)
                    cutoff=tl.min(tl.where(retained==K-1,best,inf),0)
                    threshold=(cutoff>>32).to(tl.int32).to(tl.float32,bitcast=True)
                leaves+=1; examined+=tl.sum(valid.to(tl.int32),0)
    tl.store(OUT+row*K+retained,tl.where(finite,best,0),retained<K)
    tl.store(VISITS+row*3,nodes); tl.store(VISITS+row*3+1,leaves)
    tl.store(VISITS+row*3+2,examined)


@triton.jit
def enumerate_kernel(Q,D,MEM,TREE,RADIUS,QIDS,SLACK,BASE,OUTD,OUTQ,COUNTS,VISITS,
                     G:tl.constexpr,B:tl.constexpr,WRITE:tl.constexpr,
                     STAGE_CAP:tl.constexpr=0,DIM:tl.constexpr=128):
    row=tl.program_id(0)
    lane=tl.arange(0,B); slots=tl.arange(0,32); sides=tl.arange(0,2)
    self_id=tl.load(QIDS+row); threshold=tl.load(RADIUS+row)+tl.load(SLACK+row)
    stack=tl.where(slots==0,1,0); sp=0
    count=tl.full((),0,tl.int32); nodes=0; examined=0
    while sp>=0:
        node=tl.sum(tl.where(slots==sp,stack,0),0); sp-=1; nodes+=1
        if node<G:
            node_bounds=node_bound(Q,TREE,row,node*2+sides,DIM)
            l0=tl.sum(tl.where(sides==0,node_bounds,0.),0)
            l1=tl.sum(tl.where(sides==1,node_bounds,0.),0)
            if l1<=threshold:
                sp+=1; stack=tl.where(slots==sp,node*2+1,stack)
            if l0<=threshold:
                sp+=1; stack=tl.where(slots==sp,node*2,stack)
        else:
            ids=tl.load(MEM+(node-G)*B+lane)
            valid=(ids>=0)&(ids!=self_id)
            keep=member_keep(Q,D,row,ids,valid,threshold,B,DIM)
            if WRITE or STAGE_CAP>0:
                rank=tl.cumsum(keep.to(tl.int32),0)-1
                if STAGE_CAP>0:
                    tl.store(OUTD+row*STAGE_CAP+count+rank,ids,keep&(count+rank<STAGE_CAP))
                else:
                    base=tl.load(BASE+row)
                    tl.store(OUTD+base+count+rank,ids,keep)
                    tl.store(OUTQ+base+count+rank,row,keep)
            count+=tl.sum(keep.to(tl.int32),0)
            examined+=tl.sum(valid.to(tl.int32),0)
    tl.store(COUNTS+row,count)
    tl.store(VISITS+row*2,nodes); tl.store(VISITS+row*2+1,examined)


class SpecializedEnumeration:
    def __init__(self,dim):
        self.dim=dim

    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            return enumerate_kernel[grid](*args,DIM=self.dim,**kwargs)
        return launch


class SearchTree:
    def __init__(self,database,nearfar=True,members=None,*,fast=False):
        self.database=database; self.d=database.coordinates
        self.search_engine='fast' if fast else 'original'
        self.dim=128 if nearfar else 64
        self.block=64; self.version=self.d._version
        if members is None:
            from leap.common.tree_build import median_members
            # Preserve the original add/multiply rounding, including FP32 overflow.
            keys=self.d if fast else (self.d[:,:64]+self.d[:,:64])*.5
            members=median_members(keys,self.block)
        self.members=members; self.groups=len(members)
        assert self.groups & (self.groups-1)==0 and self.groups<2**25
        self.tree=torch.empty((2*self.groups,2*self.dim),device=self.d.device)
        leaf_kernel[(self.groups,self.dim//32)](self.d,members,self.tree,self.groups,self.block,self.dim,
                                               num_warps=4,enable_fp_fusion=False)
        start=self.groups//2
        while start:
            parent_kernel[(start,self.dim//32)](self.tree,start,self.dim,num_warps=4)
            start//=2
        self.database_scale=self.d.abs().amax()
        self.kernel=SpecializedEnumeration(self.dim)
        self.signature=self._signature()

    def _signature(self):
        values=(self.database.coordinates,self.database.points,self.database.offsets,self.members,self.tree)
        return tuple((v.data_ptr(),v._version,tuple(v.shape),v.dtype,v.device) for v in values)

    def validate(self,database=None):
        if database is not None and database is not self.database:
            raise ValueError('Search index belongs to a different database')
        if self._signature()!=self.signature:
            raise ValueError('Database or tree tensors changed; rebuild the search index')

    def candidates(self,query,k=50,qids=None,slack=None):
        self.validate()
        q=query.coordinates if isinstance(query,TrajectoryIndex) else query
        if isinstance(k,bool) or not isinstance(k,int) or not 1<=k<=min(MAX_TREE_CANDIDATES,len(self.d)):
            raise ValueError('Invalid candidate count')
        if qids is None:
            qids=torch.full((len(q),),-1,device=q.device,dtype=torch.int64)
        if slack is None:
            slack=2e-5*(1+q.abs().amax(1)+self.database_scale)
        from leap.common.candidate_pages import collect
        def launch(count,width,after,out,visits):
            candidate_kernel[(len(q),)](q.contiguous(),self.d,self.members,self.tree,qids,slack.contiguous(),
                out,visits,self.groups,self.block,count,self.dim,C=width,
                HAS_AFTER=after is not None,AFTER=qids if after is None else after,
                FAST=self.search_engine=='fast',
                num_warps=8 if width>256 else 4,enable_fp_fusion=False)
        out,visits=collect(len(q),k,q.device,self.block,3,launch)
        return out&0xffffffff,visits

    def enumerate(self,query,radius,slack,qids=None,top_k=20):
        if qids is None:
            qids=torch.full((len(query.coordinates),),-1,device=self.d.device,dtype=torch.int64)
        args=(query.coordinates.contiguous(),self.d,self.members,self.tree,radius,qids,slack)
        return enumerate_candidates(self.kernel,args,len(query.coordinates),self.d.device,
                                    self.groups,self.block,top_k,26_000_000,max_pairs=26_000_000)


@torch.no_grad()
def full(query,database,tree,seed_k=50,top_k=20,batch=512,initial=None,slack=None,qids=None,progress=None,tree_initial=True):
    if not 1<=top_k<=seed_k<=len(database.coordinates):
        raise ValueError('Invalid top-k or initial pool')
    tree.validate()
    results=[]
    for lo in range(0,len(query.coordinates),batch):
        q=subset(query,lo,lo+batch); n=len(q.coordinates)
        eps=(2e-5*(1+q.coordinates.abs().amax(1)+tree.database_scale)
             if slack is None else slack[lo:lo+n]).contiguous()
        self_ids=None if qids is None else qids[lo:lo+n]
        if initial is not None:
            seeds=initial[lo:lo+n]
            seed_visits=torch.zeros((n,3),device=tree.d.device,dtype=torch.int32)
        elif tree_initial and seed_k<=MAX_TREE_CANDIDATES:
            seeds,seed_visits=tree.candidates(q,seed_k,qids=self_ids,slack=eps)
        else:
            seeds=large_candidates(q,database,seed_k,tree.dim==128,self_ids)
            seed_visits=torch.zeros((n,3),device=tree.d.device,dtype=torch.int32)
        qi=torch.arange(n,device=tree.d.device)[:,None].expand_as(seeds).reshape(-1)
        values=hausdorff_distances(q,database,qi,seeds.reshape(-1),point=True).reshape(n,seed_k)
        radius=values.topk(top_k,largest=False).values[:,-1].contiguous()
        counts,visits,ids,rows,offsets,maximum=tree.enumerate(q,radius,eps,qids=self_ids,top_k=top_k)
        values=cached_pairs(q,database,rows,ids,seeds,values,point=True)
        keys=top_keys(values,ids,rows,offsets,n,maximum,top_k)
        results.append(dict(indices=keys&0xffffffff,
            distances=(keys>>32).int().contiguous().view(torch.float32),counts=counts,
            visits=visits,seed_visits=seed_visits,radius=radius,seed_ids=seeds))
        if progress is not None:
            progress(lo+n,len(query.coordinates))
    return {key:torch.cat([r[key] for r in results]) for key in results[0]}


@triton.jit
def scan_kernel(Q,D,OUT,N:tl.constexpr,DIM:tl.constexpr):
    # Flatten the grid: large databases exceed CUDA's 65,535 grid-y limit.
    tiles=tl.cdiv(N,16)
    q=tl.program_id(0)//tiles
    ids=(tl.program_id(0)%tiles)*16+tl.arange(0,16)
    score=member_bound(Q,D,q,ids,ids<N,16,DIM)
    tl.store(OUT+q*N+ids,score,ids<N)


def scan(query,database,nearfar=True):
    q=query.coordinates if isinstance(query,TrajectoryIndex) else query
    d=database.coordinates if isinstance(database,TrajectoryIndex) else database
    out=torch.empty((len(q),len(d)),device=q.device)
    scan_kernel[(len(q)*triton.cdiv(len(d),16),)](q,d,out,len(d),128 if nearfar else 64,
                                             num_warps=4,enable_fp_fusion=False)
    return out


def large_candidates(query,database,k=1000,nearfar=True,qids=None):
    if len(query.coordinates)*len(database.coordinates)>32_000_000:
        raise MemoryError('Split the query batch for bounded full feature scan')
    scores=scan(query,database,nearfar)
    if qids is not None:
        valid=qids>=0
        scores[torch.arange(len(qids),device=scores.device)[valid],qids[valid]]=torch.inf
    ids=torch.arange(len(database.coordinates),device=scores.device)[None].expand_as(scores)
    return (lex_keys(scores,ids).topk(k,largest=False).values&0xffffffff).contiguous()


def native_near_tree(database,tree):
    """Independent legacy traversal on exactly the same near-field groups."""
    from leap.common.interval_index import IntervalSearchIndex
    result=object.__new__(IntervalSearchIndex)
    result.database=database; result.d=database.coordinates
    result.version=result.d._version; result.block=tree.block
    result.members=tree.members; result.groups=tree.groups
    result.tree=torch.cat((tree.tree[:,64:128],tree.tree[:,:64]),dim=1).contiguous()
    result.database_scale=tree.database_scale
    return result
