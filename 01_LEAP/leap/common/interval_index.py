"""Exact interval-box candidate search for the frozen dual-field model.

This is not a Hausdorff-exact database index. Worst-case traversal is linear.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from leap.common.group_envelope_filter import node_envelope, node_bound_kernel
from leap.common.trajectory_bound_filter import cross_field_bound_kernel
from leap.common.candidate_policy import MAX_TREE_CANDIDATES

def partition(x, block=128):
    """Median kd partition of interval centers, original IDs retained."""
    interval_centers = (x[:, :64] + x[:, 64:]) * .5
    leaves = []
    depth = max(0, int(np.ceil(np.log2(max(1, len(x)/block)))))
    def split(ids, level):
        if level == depth:
            leaves.append(ids)
            return
        axis = np.ptp(interval_centers[ids], axis=0).argmax()
        order = np.argsort(interval_centers[ids, axis], kind='stable')
        half = len(ids) // 2
        split(ids[order[:half]], level+1)
        split(ids[order[half:]], level+1)
    split(np.arange(len(x)), 0)
    ids = np.full((len(leaves), block), -1, np.int64)
    for i, leaf in enumerate(leaves):
        ids[i, :len(leaf)] = leaf
    return ids


class IntervalSearchIndex:
    """Immutable database-bound tree for the 64-probe dual-field score."""

    @torch.no_grad()
    def __init__(self, database, block=64):
        z_T = database.coordinates
        if block not in (64, 128):
            raise ValueError('leaf capacity must be 64 or 128')
        if z_T.ndim != 2 or z_T.shape[1] != 128 or not len(z_T):
            raise ValueError('Expected nonempty 64-probe dual-field coordinates')
        if not z_T.is_cuda or z_T.dtype != torch.float32:
            raise ValueError('Interval search requires CUDA float32 coordinates')
        if not torch.isfinite(z_T).all():
            raise ValueError('Coordinates must be finite')
        if (z_T[:,64:] > z_T[:,:64]+1e-5).any():
            raise ValueError('Invalid dual-field intervals')
        self.database = database
        self.d = z_T.contiguous()
        self.version = z_T._version
        self.block = block
        self.database_scale = self.d.abs().amax()
        from leap.common.tree_build import median_members
        self.members = median_members((self.d[:, :64] + self.d[:, 64:]) * .5, block)
        self.groups = len(self.members)
        if self.groups >= 2**25:
            raise ValueError('Tree exceeds supported traversal stack depth')
        packed = self.d[self.members.clamp_min(0)]
        valid = (self.members >= 0)[...,None]
        tree = torch.zeros((2*self.groups,128),device=self.d.device)
        # GEF: (a_bar_U, b_under_U), eq:node-bound.
        tree[self.groups:] = node_envelope(packed, valid)
        level = self.groups // 2
        while level:
            tree[level:2*level,:64] = torch.maximum(tree[2*level:4*level:2,:64],tree[2*level+1:4*level:2,:64])
            tree[level:2*level,64:] = torch.minimum(tree[2*level:4*level:2,64:],tree[2*level+1:4*level:2,64:])
            level //= 2
        self.tree = tree

    @property
    def extra_bytes(self):
        return self.members.numel()*self.members.element_size()+self.tree.numel()*self.tree.element_size()

    def validate(self, database):
        if database is not self.database or database.coordinates._version != self.version:
            raise ValueError('Search tree belongs to a different or modified database')

    @torch.no_grad()
    def candidates(self, q, qids, k=50):
        self.validate(self.database)
        if q.ndim != 2 or q.shape[1] != 128 or q.dtype != torch.float32 or q.device != self.d.device:
            raise ValueError('Queries must match the database CUDA float32 feature format')
        if qids.shape != (len(q),) or qids.dtype != torch.int64 or qids.device != q.device:
            raise ValueError('Query IDs must be an int64 vector on the query device')
        if not len(q) or isinstance(k,bool) or not isinstance(k,int) or not 1 <= k <= min(MAX_TREE_CANDIDATES,len(self.d)):
            raise ValueError('Tree candidates require nonempty queries and 1 <= k <= min(8192,N)')
        slack = (2e-5*(1+q.abs().amax(1)+self.database_scale)).contiguous()
        from leap.common.candidate_pages import collect
        def launch(count,width,after,out,visits):
            tree_kernel[(len(q),)](q.contiguous(),q.contiguous(),self.d,self.members,self.tree,
                qids.contiguous(),slack,out,visits,self.groups,self.block,count,64,C=width,
                HAS_AFTER=after is not None,AFTER=qids if after is None else after,
                num_warps=8 if width>256 else 4,enable_fp_fusion=False)
        out,visits=collect(len(q),k,q.device,self.block,3,launch)
        return (out & 0xffffffff).contiguous(), visits


def lex_keys(scores, ids):
    return (scores.contiguous().view(torch.int32).long() << 32) | ids.long()


@triton.jit(do_not_specialize=['K'])
def tree_kernel(Q,TQ,D,MEMBERS,TREE,QIDS,SLACK,OUT,VISITS,
                G:tl.constexpr,B:tl.constexpr,K,M:tl.constexpr,
                SINGLE_QUERY:tl.constexpr=False,C:tl.constexpr=0,
                HAS_AFTER:tl.constexpr=False,AFTER=None):
    q=tl.program_id(0)
    WIDTH:tl.constexpr=B if C==0 else C
    retained=tl.arange(0,WIDTH)
    lane=tl.arange(0,B)
    f=tl.arange(0,64)
    mf=tl.arange(0,triton.next_power_of_2(M))
    slot=tl.arange(0,32)
    sides=tl.arange(0,2)
    a_q=tl.load(Q+q*128+f);b_q=tl.load(Q+q*128+64+f)
    a_tree_q=tl.load(TQ+q*(2*M)+mf,mf<M,other=0.)
    b_tree_q=tl.load(TQ+q*(2*M)+M+mf,mf<M,other=0.)
    if SINGLE_QUERY:
        # SLACK supplies the precomputed database scale in this specialization.
        self_id=tl.full((),-1,tl.int64)
        finite=tl.min(((tl.abs(a_q)<float('inf'))&(tl.abs(b_q)<float('inf'))).to(tl.int32),0)>0
        scale=tl.maximum(tl.max(tl.abs(a_q),0),tl.max(tl.abs(b_q),0))
        slack=((1.+scale)+tl.load(SLACK))*2e-5
    else:
        self_id=tl.load(QIDS+q)
        slack=tl.load(SLACK+q)
        finite=tl.full((),True,tl.int1)
    inf_key=tl.full((),9187343244130779135,tl.int64)
    best=tl.full((WIDTH,),9187343244130779135,tl.int64)
    stack=tl.where(slot==0,1,0)
    stack_lb=tl.full((32,),0.,tl.float32)
    sp=tl.where(finite,0,-1)
    threshold=tl.full((),float('inf'),tl.float32)
    leaves=0;nodes=0;examined=0
    while sp>=0:
        node=tl.sum(tl.where(slot==sp,stack,0),0)
        lower=tl.min(tl.where(slot==sp,stack_lb,float('inf')),0)
        sp-=1
        if lower<=threshold:
            nodes+=1
            if node<G:
                child=node*2+sides
                a_bar_U=tl.load(TREE+child[:,None]*(2*M)+mf[None,:],mf[None,:]<M,other=0.)
                b_under_U=tl.load(TREE+child[:,None]*(2*M)+M+mf[None,:],mf[None,:]<M,other=0.)
                ell_qU=node_bound_kernel(a_tree_q[None,:],b_tree_q[None,:],a_bar_U,b_under_U,1)-slack
                l0=tl.sum(tl.where(sides==0,ell_qU,0.),0)
                l1=tl.sum(tl.where(sides==1,ell_qU,0.),0)
                near=tl.where(l0<=l1,node*2,node*2+1)
                far=tl.where(l0<=l1,node*2+1,node*2)
                near_lb=tl.minimum(l0,l1);far_lb=tl.maximum(l0,l1)
                if far_lb<=threshold:
                    sp+=1
                    stack=tl.where(slot==sp,far,stack)
                    stack_lb=tl.where(slot==sp,far_lb,stack_lb)
                if near_lb<=threshold:
                    sp+=1
                    stack=tl.where(slot==sp,near,stack)
                    stack_lb=tl.where(slot==sp,near_lb,stack_lb)
            else:
                group=node-G
                ids=tl.load(MEMBERS+group*B+lane)
                valid=(ids>=0)&(ids!=self_id)
                a_T=tl.load(D+ids[:,None]*128+f[None,:],valid[:,None],other=0.)
                b_T=tl.load(D+ids[:,None]*128+64+f[None,:],valid[:,None],other=0.)
                L_qT=cross_field_bound_kernel(a_q[None,:],b_q[None,:],a_T,b_T,1)
                keys=(L_qT.to(tl.int32,bitcast=True).to(tl.int64)<<32)|ids.to(tl.int64)
                keys=tl.where(valid,keys,inf_key)
                if HAS_AFTER:
                    keys=tl.where(keys>tl.load(AFTER+q),keys,inf_key)
                if WIDTH>B:
                    keys=tl.reshape(tl.where(tl.arange(0,WIDTH//B)[:,None]==0,keys[None,:],inf_key),(WIDTH,))
                    best=tl.sort(tl.minimum(best,tl.sort(keys,descending=True)),descending=False)
                else:
                    ordered=tl.sort(tl.reshape(tl.join(best,keys),(2*B,)),descending=False)
                    best,discarded=tl.split(tl.trans(tl.reshape(ordered,(2,B))))
                kth=tl.min(tl.where(retained==K-1,best,inf_key),0)
                threshold=(kth>>32).to(tl.int32).to(tl.float32,bitcast=True)
                leaves+=1
                examined+=tl.sum(valid.to(tl.int32),0)
    if SINGLE_QUERY:
        # Invalid fields never enter traversal and become a host-visible error.
        tl.store(OUT+q*K+retained,tl.where(finite,best&0xffffffff,-1),retained<K)
    else:
        tl.store(OUT+q*K+retained,best,retained<K)
    tl.store(VISITS+q*3,leaves)
    tl.store(VISITS+q*3+1,nodes)
    tl.store(VISITS+q*3+2,examined)


@triton.jit
def reference_kernel(Q,D,QIDS,O,N:tl.constexpr):
    q=tl.program_id(0)
    j=tl.program_id(1)*16+tl.arange(0,16)
    f=tl.arange(0,32)
    L_qT=tl.full((16,),0.,tl.float32)
    for start in range(0,64,32):
        a_q=tl.load(Q+q*128+start+f)
        b_q=tl.load(Q+q*128+64+start+f)
        a_T=tl.load(D+j[:,None]*128+start+f[None,:],j[:,None]<N,other=0.)
        b_T=tl.load(D+j[:,None]*128+64+start+f[None,:],j[:,None]<N,other=0.)
        L_qT=tl.maximum(L_qT,tl.max(tl.maximum(b_q[None,:]-a_T,b_T-a_q[None,:]),1))
    self_id=tl.load(QIDS+q)
    L_qT=tl.where(self_id==j,float('inf'),L_qT)
    tl.store(O+q*N+j,L_qT,j<N)


def reference(q,d,qids,k=50,deterministic=True):
    scores=torch.empty((len(q),len(d)),device=q.device,dtype=torch.float32)
    reference_kernel[(len(q),triton.cdiv(len(d),16))](q,d,qids,scores,len(d),num_warps=4,enable_fp_fusion=False)
    if deterministic:
        return lex_keys(scores,torch.arange(len(d),device=q.device)[None,:]).topk(k,largest=False).values
    return scores.topk(k,largest=False)
