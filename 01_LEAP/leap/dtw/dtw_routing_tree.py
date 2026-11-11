"""Tree Top-K for ordered-mass routing (eq:dtw-seed-scan).

For each of eight ordered XY means, a node stores an enclosing rectangle.
sum_b dist(mu_b(q), rect_b(U))*max(n_q,min_{T in U} n_T)/8 bounds
every member's routing score. It is not claimed to bound raw DTW.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from leap.common.candidate_policy import MAX_TREE_CANDIDATES


class RoutingTree:
    @torch.no_grad()
    def __init__(self, database, block=128, *, fast=False):
        if block != 128:
            raise ValueError('Routing tree uses 128-member leaves')
        self.database = database
        self.fast = fast
        from leap.dtw.ordered_dtw import signature
        self.version = signature(database.points,database.offsets,database.ordered_summary)
        from leap.common.tree_build import median_members, envelopes
        x = database.ordered_summary.detach()
        lengths = database.offsets[1:]-database.offsets[:-1]
        self.members=median_members(x,block)
        self.groups,self.block=len(self.members),block
        bounds=envelopes(x,self.members,upper_first=False)
        self.low=bounds[:,:16].contiguous()
        self.high=bounds[:,16:].contiguous()
        self.low[0]=torch.inf
        self.high[0]=-torch.inf
        self.nmin=torch.full((2*self.groups,),torch.iinfo(torch.int32).max,
                            device=x.device,dtype=torch.int32)
        self.nmin[self.groups:]=lengths[self.members.clamp_min(0)].masked_fill(
            self.members<0,torch.iinfo(torch.int32).max).amin(1).to(torch.int32)
        level=self.groups//2
        while level:
            self.nmin[level:2*level]=torch.minimum(self.nmin[2*level:4*level:2],
                                                 self.nmin[2*level+1:4*level:2])
            level//=2
        self.last_visits=None
        self.signal_plan=None
        if fast and getattr(database,'dtw_probe_summary',None) is not None:
            from leap.dtw.dtw_signal_batch import SignalPairPlan
            self.signal_plan=SignalPairPlan(database)

    @torch.no_grad()
    def candidates(self,queries,k,qids=None):
        from leap.dtw.ordered_dtw import signature
        db=self.database
        if self.version != signature(db.points,db.offsets,db.ordered_summary):
            raise ValueError('Routing tree data changed; rebuild')
        if isinstance(k,bool) or not isinstance(k,int) or not 1<=k<=min(MAX_TREE_CANDIDATES,len(db.coordinates)-int(qids is not None)):
            raise ValueError('Routing tree Top-K must be <=8192 and fit available trajectories')
        nq=len(queries.coordinates)
        ids=torch.full((nq,),-1,device=queries.points.device,dtype=torch.int64) if qids is None else qids
        if ids.shape!=(nq,) or ids.dtype!=torch.int64 or ids.device!=db.points.device:
            raise ValueError('Self IDs must be one int64 per query on the index device')
        from leap.common.candidate_pages import collect
        def launch(count,width,after,out,visits):
            routing_kernel[(nq,)](queries.ordered_summary,self.database.ordered_summary,
                queries.offsets,self.database.offsets,self.members,self.low,self.high,self.nmin,ids,
                out,visits,self.groups,count,C=width,RETURN_KEYS=True,
                HAS_AFTER=after is not None,AFTER=ids if after is None else after,
                FAST=self.fast,
                num_warps=8 if width>256 else 4,enable_fp_fusion=False)
        out,visits=collect(nq,k,ids.device,128,3,launch)
        self.last_visits=visits
        return out&0xffffffff


@triton.jit(do_not_specialize=['K'])
def routing_kernel(Q,D,QO,DO,MEM,LOW,HIGH,NMIN,QIDS,OUT,VISITS,G:tl.constexpr,K,C:tl.constexpr=128,
                   HAS_AFTER:tl.constexpr=False,AFTER=None,RETURN_KEYS:tl.constexpr=False,
                   FAST:tl.constexpr=False):
    q=tl.program_id(0)
    lane=tl.arange(0,128);slot=tl.arange(0,32);sides=tl.arange(0,2)
    retained=tl.arange(0,C)
    n=tl.load(QO+q+1)-tl.load(QO+q)
    self_id=tl.load(QIDS+q)
    inf_key=tl.full((),9187343244130779135,tl.int64)
    best=tl.full((C,),9187343244130779135,tl.int64)
    stack=tl.where(slot==0,1,0);stack_lb=tl.full((32,),0.,tl.float32);sp=0
    threshold=tl.full((),float('inf'),tl.float32)
    cutoff=inf_key
    nodes=0;leaves=0;examined=0
    while sp>=0:
        node=tl.sum(tl.where(slot==sp,stack,0),0)
        lower=tl.min(tl.where(slot==sp,stack_lb,float('inf')),0);sp-=1
        if lower<=threshold:
            nodes+=1
            if node<G:
                child=node*2+sides
                node_total=tl.full((2,),0.,tl.float32)
                error_scale=tl.full((2,),0.,tl.float32)
                for b in range(8):
                    x=tl.load(Q+q*16+b*2);y=tl.load(Q+q*16+b*2+1)
                    xmin=tl.load(LOW+child*16+b*2);ymin=tl.load(LOW+child*16+b*2+1)
                    xmax=tl.load(HIGH+child*16+b*2);ymax=tl.load(HIGH+child*16+b*2+1)
                    node_dx=tl.maximum(0.,tl.maximum(xmin-x,x-xmax))
                    node_dy=tl.maximum(0.,tl.maximum(ymin-y,y-ymax))
                    node_total+=tl.sqrt(node_dx*node_dx+node_dy*node_dy)
                    error_scale+=tl.abs(x)+tl.abs(y)+tl.maximum(tl.abs(xmin),tl.abs(xmax))+tl.maximum(tl.abs(ymin),tl.abs(ymax))
                scale=tl.maximum(n,tl.load(NMIN+child))
                child_lower=scale*(node_total/8.)-64.*1.1920928955078125e-7*scale*(1.+error_scale)
                l0=tl.sum(tl.where(sides==0,child_lower,0.),0)
                l1=tl.sum(tl.where(sides==1,child_lower,0.),0)
                near=tl.where(l0<=l1,node*2,node*2+1)
                far=tl.where(l0<=l1,node*2+1,node*2)
                if tl.maximum(l0,l1)<=threshold:
                    sp+=1;stack=tl.where(slot==sp,far,stack)
                    stack_lb=tl.where(slot==sp,tl.maximum(l0,l1),stack_lb)
                if tl.minimum(l0,l1)<=threshold:
                    sp+=1;stack=tl.where(slot==sp,near,stack)
                    stack_lb=tl.where(slot==sp,tl.minimum(l0,l1),stack_lb)
            else:
                ids=tl.load(MEM+(node-G)*128+lane)
                valid=(ids>=0)&(ids!=self_id)
                m=tl.load(DO+ids+1,valid,other=0)-tl.load(DO+ids,valid,other=0)
                total=tl.full((128,),0.,tl.float32)
                for b in range(8):
                    dx=tl.load(Q+q*16+b*2)-tl.load(D+ids*16+b*2,valid,other=0.)
                    dy=tl.load(Q+q*16+b*2+1)-tl.load(D+ids*16+b*2+1,valid,other=0.)
                    total+=tl.sqrt(dx*dx+dy*dy)
                score=tl.maximum(n,m)*(total/8.)
                keys=(score.to(tl.int32,bitcast=True).to(tl.int64)<<32)|ids
                keys=tl.where(valid,keys,inf_key)
                if HAS_AFTER:
                    keys=tl.where(keys>tl.load(AFTER+q),keys,inf_key)
                if C>128:
                    keys=tl.reshape(tl.where(tl.arange(0,C//128)[:,None]==0,keys[None,:],inf_key),(C,))
                if FAST:
                    if tl.min(keys,0)<=cutoff:
                        best=tl.sort(tl.minimum(best,tl.sort(keys,descending=True)),descending=False)
                        cutoff=tl.min(tl.where(retained==K-1,best,inf_key),0)
                        threshold=(cutoff>>32).to(tl.int32).to(tl.float32,bitcast=True)
                elif C>128:
                    best=tl.sort(tl.minimum(best,tl.sort(keys,descending=True)),descending=False)
                else:
                    ordered=tl.sort(tl.reshape(tl.join(best,keys),(256,)),descending=False)
                    best,_=tl.split(tl.trans(tl.reshape(ordered,(2,128))))
                if not FAST:
                    kth=tl.min(tl.where(retained==K-1,best,inf_key),0)
                    threshold=(kth>>32).to(tl.int32).to(tl.float32,bitcast=True)
                leaves+=1;examined+=tl.sum(valid.to(tl.int32),0)
    tl.store(OUT+q*K+retained,best if RETURN_KEYS else best&0xffffffff,retained<K)
    tl.store(VISITS+q*3,leaves)
    tl.store(VISITS+q*3+1,nodes)
    tl.store(VISITS+q*3+2,examined)
