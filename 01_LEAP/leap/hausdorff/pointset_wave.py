import torch
import triton
import triton.language as tl
from leap.hausdorff.pointset_index import node_bound, member_keep
from leap.common.range_buffers import compact_candidates, scatter_keys


@triton.jit
def traverse(Q,D,MEM,TREE,RADIUS,EPS,SEEDS,SELF,STACK,LOWER,SP,
             BASE,OUT,ROWS,COUNTS,VISITS,LEAVES,
             G:tl.constexpr,B:tl.constexpr=64,FIRST:tl.constexpr=False,
             CAP:tl.constexpr=2048,WRITE:tl.constexpr=False,LIMIT:tl.constexpr=4):
    row=tl.program_id(0)
    lane=tl.arange(0,B);slot=tl.arange(0,32);side=tl.arange(0,2)
    ss=tl.arange(0,64)
    seed=tl.load(SEEDS+row*50+ss,ss<50,other=-2)
    self_id=tl.load(SELF+row)
    threshold=tl.load(RADIUS+row)+tl.load(EPS+row)
    if FIRST:
        tl.store(LEAVES+row*LIMIT+tl.arange(0,LIMIT),-1)
        stack=tl.where(slot==0,1,0)
        lbs=tl.full((32,),0.,tl.float32)
        sp=tl.full((),0,tl.int32)
    else:
        sp=tl.load(SP+row)
        stack=tl.load(STACK+row*32+slot,slot<=sp,other=0)
        lbs=tl.load(LOWER+row*32+slot,slot<=sp,other=0.)
    count=tl.full((),0,tl.int32)
    nodes=tl.full((),0,tl.int32)
    examined=tl.full((),0,tl.int32)
    leaves=tl.full((),0,tl.int32)
    active=sp>=0
    while active:
        node=tl.sum(tl.where(slot==sp,stack,0),0)
        lower=tl.sum(tl.where(slot==sp,lbs,0.),0)
        sp-=1
        if lower<=threshold:
            nodes+=1
            if node<G:
                nb=node_bound(Q,TREE,row,node*2+side,128)
                l0=tl.sum(tl.where(side==0,nb,0.),0)
                l1=tl.sum(tl.where(side==1,nb,0.),0)
                near=tl.where(l0<=l1,node*2,node*2+1)
                far=tl.where(l0<=l1,node*2+1,node*2)
                if tl.maximum(l0,l1)<=threshold:
                    sp+=1;stack=tl.where(slot==sp,far,stack)
                    lbs=tl.where(slot==sp,tl.maximum(l0,l1),lbs)
                if tl.minimum(l0,l1)<=threshold:
                    sp+=1;stack=tl.where(slot==sp,near,stack)
                    lbs=tl.where(slot==sp,tl.minimum(l0,l1),lbs)
            else:
                ids=tl.load(MEM+(node-G)*B+lane)
                valid=(ids>=0)&(ids!=self_id)
                already=tl.sum((ids[:,None]==seed[None,:]).to(tl.int32),1)>0
                keep=member_keep(Q,D,row,ids,valid&(~already),threshold,B,128)
                rank=tl.cumsum(keep.to(tl.int32),0)-1
                if WRITE:
                    base=tl.load(BASE+row)
                    tl.store(OUT+base+count+rank,ids,keep)
                    tl.store(ROWS+base+count+rank,row,keep)
                else:
                    tl.store(OUT+row*CAP+count+rank,ids,keep&(count+rank<CAP))
                count+=tl.sum(keep.to(tl.int32),0)
                examined+=tl.sum(valid.to(tl.int32),0)
                if FIRST:
                    tl.store(LEAVES+row*LIMIT+leaves,node-G)
                leaves+=1
        active=sp>=0
        if FIRST:
            active=active&(leaves<LIMIT)
    if FIRST:
        tl.store(STACK+row*32+slot,stack)
        tl.store(LOWER+row*32+slot,lbs)
        tl.store(SP+row,sp)
    tl.store(COUNTS+row,count)
    tl.store(VISITS+row*3,nodes)
    tl.store(VISITS+row*3+1,examined)
    tl.store(VISITS+row*3+2,leaves)


def enumerate_wave(q,db,tree,radius,eps,seeds,self_ids,frontier,first,capacity=None):
    n=len(q.coordinates)
    capacity=(256 if first else 2048) if capacity is None else capacity
    counts=torch.empty(n,device=eps.device,dtype=torch.int32)
    visits=torch.empty((n,3),device=eps.device,dtype=torch.int32)
    stage=torch.empty((n,capacity),device=eps.device,dtype=torch.int64)
    stack,lower,sp,leaves=frontier
    args=(q.coordinates,db.coordinates,tree.members,tree.tree,radius,eps,
          seeds,self_ids,stack,lower,sp)
    compiled=traverse[(n,)](*args,counts,stage,counts,counts,visits,leaves,
        tree.groups,FIRST=first,CAP=capacity,num_warps=4,enable_fp_fusion=False)
    maximum,total=map(int,torch.stack((counts.max(),counts.sum())).tolist())
    if n*max(maximum,1)>8_000_000:
        raise MemoryError('Two-wave candidate allocation exceeds fixed bound')
    offsets=counts.long().cumsum(0)-counts
    ids=torch.empty(total,device=eps.device,dtype=torch.int64)
    rows=torch.empty_like(ids)
    if maximum>capacity:
        if first:
            raise RuntimeError('First-wave leaf budget exceeded its exact capacity')
        written=torch.empty_like(counts)
        traverse[(n,)](*args,offsets,ids,rows,written,visits,leaves,
            tree.groups,FIRST=False,WRITE=True,num_warps=4,enable_fp_fusion=False)
        if not torch.equal(written,counts):
            raise RuntimeError('Saved frontier changed across overflow replay')
    elif total:
        compact_candidates[(n,triton.cdiv(maximum,128))](
            stage,counts,offsets,ids,rows,capacity,128)
    return dict(ids=ids,rows=rows,counts=counts,visits=visits,offsets=offsets,
                maximum=maximum,overflow=int(maximum>capacity),compiled=compiled)


def make_frontier(n,device):
    return (torch.empty((n,32),device=device,dtype=torch.int32),
            torch.empty((n,32),device=device),
            torch.empty(n,device=device,dtype=torch.int32),
            torch.full((n,4),-1,device=device,dtype=torch.int32))


def merge(base_keys,wave,values,k=20):
    n=len(base_keys)
    width=max(k,wave['maximum'])
    keys=torch.full((n,width),9187343244130779135,device=values.device,dtype=torch.int64)
    if len(values):
        scatter_keys[(triton.cdiv(len(values),256),)](values,wave['ids'],wave['rows'],
            wave['offsets'],keys,len(values),width,256,num_warps=4)
    return torch.cat((base_keys,keys),dim=1).topk(k,largest=False).values.contiguous()


def keys_to_distances(keys):
    return (keys>>32).int().contiguous().view(torch.float32)
