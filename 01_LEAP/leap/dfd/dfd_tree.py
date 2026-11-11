"""Exact PSE TopK using vertex and endpoint group envelopes, no full scan."""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from leap.common.candidate_policy import MAX_TREE_CANDIDATES


@triton.jit(do_not_specialize=['K'])
def candidate_kernel(QZ,DZ,EQ,ED,QIDS,MEM,VERT,ENDS,OUT,VISITS,SCALE,
                     G:tl.constexpr,B:tl.constexpr,K,C:tl.constexpr,
                     HAS_AFTER:tl.constexpr=False,AFTER=None,FAST:tl.constexpr=False):
    row=tl.program_id(0); q=tl.load(QIDS+row)
    lane=tl.arange(0,B); retained=tl.arange(0,C)
    f=tl.arange(0,64); slot=tl.arange(0,32); side=tl.arange(0,2)
    qa=tl.load(QZ+row*128+f)
    # Same numerical slack as the existing range search.
    qb=tl.load(QZ+row*128+64+f)
    slack=2e-5*(1.+tl.maximum(tl.max(tl.abs(qa),0),tl.max(tl.abs(qb),0))+tl.load(SCALE))
    best=tl.full((C,),9187343244130779135,tl.int64)
    stack=tl.where(slot==0,1,0); lbs=tl.full((32,),0.,tl.float32); sp=0
    threshold=tl.full((),float('inf'),tl.float32)
    cutoff=tl.full((),9187343244130779135,tl.int64)
    nodes=tl.full((),0,tl.int32); examined=tl.full((),0,tl.int32)
    while sp>=0:
        node=tl.sum(tl.where(slot==sp,stack,0),0)
        node_lb=tl.min(tl.where(slot==sp,lbs,float('inf')),0)
        sp-=1
        if node_lb<=threshold+slack:
            nodes+=1
            if node<G:
                children=node*2+side
                upper=tl.load(VERT+children[:,None]*128+f[None,:])
                lower=tl.load(VERT+children[:,None]*128+64+f[None,:])
                bound=tl.maximum(0.,tl.max(tl.maximum(qa[None,:]-upper,lower-qa[None,:]),1))
                for end in range(2):
                    x=tl.load(EQ+row*4+end*2); y=tl.load(EQ+row*4+end*2+1)
                    ux=tl.load(ENDS+children*8+end*2); uy=tl.load(ENDS+children*8+end*2+1)
                    lx=tl.load(ENDS+children*8+4+end*2); ly=tl.load(ENDS+children*8+4+end*2+1)
                    dx=tl.maximum(0.,tl.maximum(x-ux,lx-x)); dy=tl.maximum(0.,tl.maximum(y-uy,ly-y))
                    bound=tl.maximum(bound,tl.sqrt(dx*dx+dy*dy))
                l0=tl.sum(tl.where(side==0,bound,0.),0); l1=tl.sum(tl.where(side==1,bound,0.),0)
                near=tl.where(l0<=l1,node*2,node*2+1); far=tl.where(l0<=l1,node*2+1,node*2)
                nl=tl.minimum(l0,l1); fl=tl.maximum(l0,l1)
                if fl<=threshold+slack:
                    sp+=1; stack=tl.where(slot==sp,far,stack); lbs=tl.where(slot==sp,fl,lbs)
                if nl<=threshold+slack:
                    sp+=1; stack=tl.where(slot==sp,near,stack); lbs=tl.where(slot==sp,nl,lbs)
            else:
                ids=tl.load(MEM+(node-G)*B+lane,lane<B,other=-1)
                valid=(ids>=0)&(ids!=q)&(lane<B)
                if FAST:
                    score=tl.full((B,),0.,tl.float32)
                else:
                    da=tl.load(DZ+ids[:,None]*128+f[None,:],valid[:,None],other=0.)
                    score=tl.max(tl.abs(qa[None,:]-da),1)
                for end in range(2):
                    dx=tl.load(EQ+row*4+end*2)-tl.load(ED+ids*4+end*2,valid,other=0.)
                    dy=tl.load(EQ+row*4+end*2+1)-tl.load(ED+ids*4+end*2+1,valid,other=0.)
                    # Match the scan scorer's FP32 rounding before score/ID ties.
                    score=tl.maximum(score,libdevice.sqrt_rn(dx*dx+dy*dy))
                if FAST:
                    # Maxima only increase. Equal-distance candidates remain for ID ties.
                    alive=valid&(score<=threshold)
                    chunk=tl.arange(0,32)
                    for start in range(0,64,32):
                        if tl.sum(alive.to(tl.int32),0)>0:
                            qpart=tl.load(QZ+row*128+start+chunk)
                            da=tl.load(DZ+ids[:,None]*128+start+chunk[None,:],alive[:,None],other=0.)
                            score=tl.maximum(score,tl.max(tl.abs(qpart[None,:]-da),1))
                            alive=alive&(score<=threshold)
                    score=tl.where(alive,score,float('inf'))
                keys=(score.to(tl.int32,bitcast=True).to(tl.int64)<<32)|ids
                keys=tl.where(valid,keys,9187343244130779135)
                if HAS_AFTER:
                    keys=tl.where(keys>tl.load(AFTER+row),keys,9187343244130779135)
                if C>B:
                    keys=tl.reshape(tl.where(tl.arange(0,C//B)[:,None]==0,keys[None,:],9187343244130779135),(C,))
                if FAST:
                    if tl.min(keys,0)<=cutoff:
                        best=tl.sort(tl.minimum(best,tl.sort(keys,descending=True)),descending=False)
                        cutoff=tl.min(tl.where(retained==K-1,best,9187343244130779135),0)
                        threshold=(cutoff>>32).to(tl.int32).to(tl.float32,bitcast=True)
                elif C>B:
                    best=tl.sort(tl.minimum(best,tl.sort(keys,descending=True)),descending=False)
                else:
                    ordered=tl.sort(tl.reshape(tl.join(best,keys),(2*C,)),descending=False)
                    best,discard=tl.split(tl.trans(tl.reshape(ordered,(2,C))))
                if not FAST:
                    kth=tl.min(tl.where(retained==K-1,best,9187343244130779135),0)
                    threshold=(kth>>32).to(tl.int32).to(tl.float32,bitcast=True)
                examined+=tl.sum(valid.to(tl.int32),0)
    tl.store(OUT+row*K+retained,best,retained<K)
    tl.store(VISITS+row*2,nodes); tl.store(VISITS+row*2+1,examined)


@triton.jit
def native_enum(Q, D, EQ, ED, MEM, VERT, ENDS, RADIUS, QIDS, SLACK, BASE, OUTD, OUTQ, COUNTS, VISITS,
                G: tl.constexpr, B: tl.constexpr, WRITE: tl.constexpr, STAGE_CAP: tl.constexpr=0,
                ROOTS=None):
    row = tl.program_id(0)
    f = tl.arange(0, 64); lane = tl.arange(0, B); slot = tl.arange(0, 32); side = tl.arange(0, 2)
    qa = tl.load(Q+row*128+f)
    self_id = tl.load(QIDS+row)
    threshold = tl.load(RADIUS+row)+tl.load(SLACK+row)
    root = 1 if ROOTS is None else tl.load(ROOTS+row)
    stack = tl.where(slot==0, root, 0); sp=tl.where(root>0,0,-1)
    count=tl.full((),0,tl.int32); nodes=tl.full((),0,tl.int32); examined=tl.full((),0,tl.int32)
    while sp>=0:
        node=tl.sum(tl.where(slot==sp,stack,0),0); sp-=1; nodes+=1
        if node<G:
            children=node*2+side
            upper=tl.load(VERT+children[:,None]*128+f[None,:])
            lower=tl.load(VERT+children[:,None]*128+64+f[None,:])
            lb=tl.maximum(0.,tl.max(tl.maximum(qa[None,:]-upper,lower-qa[None,:]),1))
            for end in range(2):
                qx=tl.load(EQ+row*4+end*2); qy=tl.load(EQ+row*4+end*2+1)
                ux=tl.load(ENDS+children*8+end*2); uy=tl.load(ENDS+children*8+end*2+1)
                lx=tl.load(ENDS+children*8+4+end*2); ly=tl.load(ENDS+children*8+4+end*2+1)
                dx=tl.maximum(0.,tl.maximum(qx-ux,lx-qx))
                dy=tl.maximum(0.,tl.maximum(qy-uy,ly-qy))
                lb=tl.maximum(lb,tl.sqrt(dx*dx+dy*dy))
            l0=tl.sum(tl.where(side==0,lb,0.),0); l1=tl.sum(tl.where(side==1,lb,0.),0)
            if l1<=threshold:
                sp+=1; stack=tl.where(slot==sp,node*2+1,stack)
            if l0<=threshold:
                sp+=1; stack=tl.where(slot==sp,node*2,stack)
        else:
            ids=tl.load(MEM+(node-G)*B+lane)
            valid=(ids>=0)&(ids!=self_id)
            da=tl.load(D+ids[:,None]*128+f[None,:],valid[:,None],other=0.)
            score=tl.max(tl.abs(qa[None,:]-da),1)
            for end in range(2):
                dx=tl.load(EQ+row*4+end*2)-tl.load(ED+ids*4+end*2,valid,other=0.)
                dy=tl.load(EQ+row*4+end*2+1)-tl.load(ED+ids*4+end*2+1,valid,other=0.)
                score=tl.maximum(score,tl.sqrt(dx*dx+dy*dy))
            keep=valid&(score<=threshold)
            if WRITE or STAGE_CAP > 0:
                rank=tl.cumsum(keep.to(tl.int32),0)-1
                if STAGE_CAP > 0:
                    tl.store(OUTD+row*STAGE_CAP+count+rank,ids,keep&(count+rank<STAGE_CAP))
                else:
                    base=tl.load(BASE+row)
                    tl.store(OUTD+base+count+rank,ids,keep); tl.store(OUTQ+base+count+rank,row,keep)
            count+=tl.sum(keep.to(tl.int32),0); examined+=tl.sum(valid.to(tl.int32),0)
    tl.store(COUNTS+row,count)
    tl.store(VISITS+row*2,nodes); tl.store(VISITS+row*2+1,examined)


@torch.no_grad()
def make_native(index, tree, ep):
    from leap.common.tree_build import envelopes
    return envelopes(index.coordinates[:,:64],tree.members), envelopes(ep,tree.members)



def endpoints(index):
    from leap.common.runtime.context import current
    plan = current()
    if plan is not None and len(index.coordinates) == 1:
        return plan.endpoint(index)
    return torch.cat((index.points[index.offsets[:-1]],index.points[index.offsets[1:]-1]),1).contiguous()


@torch.no_grad()
def prepare(index,tree):
    ep=endpoints(index)
    return {'endpoints':ep,'native':make_native(index,tree,ep)}


@torch.no_grad()
def candidate_pool(queries,index,tree,qids,k=200):
    """Exact Top-k by eq:dfd-bound, using GEF envelopes instead of a full scan."""
    if isinstance(k,bool) or not isinstance(k,int) or not 1<=k<=min(MAX_TREE_CANDIDATES,len(index.coordinates)):
        raise ValueError('DFD candidate pool must be between 1 and min(8192,N)')
    n=len(queries.coordinates)
    from leap.common.candidate_pages import collect
    ep=endpoints(queries)
    def launch(count,width,after,top,visits):
        if n:
            candidate_kernel[(n,)](queries.coordinates,index.coordinates,ep,
                tree.dfd_geometry['endpoints'],qids,tree.members,*tree.dfd_geometry['native'],
                top,visits,tree.database_scale,tree.groups,tree.block,count,width,
                HAS_AFTER=after is not None,AFTER=qids if after is None else after,
                FAST=getattr(tree,'search_engine','original')=='fast',
                num_warps=8 if width>=256 else 4,enable_fp_fusion=False)
    top,visits=collect(n,k,index.points.device,tree.block,2,launch)
    return top&0xffffffff,(top>>32).int().contiguous().view(torch.float32),visits


@torch.no_grad()
def initial_candidates(queries,index,tree,qids,k=32,pool=None,strategy='light'):
    """eq:dfd-progressive: L_F -> retained8 -> S_q, with s=32 by default.

    The light path scores64 candidates. The optional progressive reference
    scores200, then strengthens the first64 with32 retained vertices.
    """
    from leap.dfd.dfd_geometry import witness_bound, ordered_bound
    from leap.common.interval_index import lex_keys
    from leap.common.runtime.context import current
    plan = current()
    if plan is not None:
        result = plan.dfd_candidates(queries, index, tree, qids, k, pool, strategy)
        if result is not None:
            return result
    same=bool((qids>=0).any())
    if strategy not in ('light','progressive'):
        raise ValueError('Unknown DFD initial-ranking strategy')
    if pool is None:
        pool=max(k,64) if strategy=='light' else 200
    pool=min(pool,len(index.coordinates)-int(same))
    limit=64 if strategy=='progressive' else MAX_TREE_CANDIDATES
    if not 1<=k<=min(limit,pool):
        raise ValueError('initial_k exceeds the selected strategy or available pool')
    ids,lower,_=candidate_pool(queries,index,tree,qids,pool)
    rows=torch.arange(len(ids),device=ids.device)[:,None].expand_as(ids).reshape(-1)
    geometric=witness_bound(queries,index,rows,ids.reshape(-1),8).reshape_as(lower)
    ordered=ordered_bound(queries,index,rows,ids.reshape(-1),8).reshape_as(lower)
    score=torch.maximum(torch.maximum(lower,geometric),ordered)
    if strategy=='light':
        return (lex_keys(score,ids).topk(k,largest=False).values&0xffffffff).contiguous()
    keys=lex_keys(score,ids).topk(min(64,pool),largest=False).values
    ids=keys&0xffffffff; score=(keys>>32).int().contiguous().view(torch.float32)
    rows=torch.arange(len(ids),device=ids.device)[:,None].expand_as(ids).reshape(-1)
    ordered=ordered_bound(queries,index,rows,ids.reshape(-1),32).reshape_as(score)
    top=lex_keys(torch.maximum(score,ordered),ids).topk(k,largest=False).values
    return (top&0xffffffff).contiguous()
