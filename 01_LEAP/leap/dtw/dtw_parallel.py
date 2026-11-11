"""Same DTW group/member bounds, separate traversal from parallel leaf evaluation."""
import torch
import triton
import triton.language as tl
from leap.dtw.dtw_range import box_distance
from leap.common.group_envelope_filter import node_bound_kernel, dtw_node_bound_kernel
from leap.common.trajectory_bound_filter import dtw_individual_bound_kernel
from leap.dtw.dtw_shortcircuit import target_first


MAX_LEAF_CELLS=32_000_000
MAX_MEMBER_CELLS=64_000_000


def bucket_tile(width, leaf_size, fallback):
    from leap.common.runtime import options as opt
    if not opt.dtw_bucket_tiles:
        return fallback
    # Bound the point-axis temporary without over-splitting short buckets.
    return min(leaf_size, max(4, 4096 // width))


@triton.jit
def leaf_traversal(QP,QOFF,QZ,TREE,QBOX,TBOX,RADIUS,SLACK,RF,RT,LEAVES,COUNTS,VISITS,
                   G:tl.constexpr,N:tl.constexpr,RADIAL:tl.constexpr,
                   PARTS:tl.constexpr=1,ROOTS=None,ROOT_COUNTS=None):
    row=tl.program_id(0)
    part=tl.program_id(1)
    root=1
    active=True
    if PARTS>1:
        active=part<tl.load(ROOT_COUNTS+row)
        root=PARTS+tl.load(ROOTS+row*PARTS+part,active,other=0)
    j=tl.arange(0,N);f=tl.arange(0,64);slot=tl.arange(0,32);sides=tl.arange(0,2)
    lo=tl.load(QOFF+row);nq=tl.load(QOFF+row+1)-lo
    x=tl.load(QP+(lo+j)*2,j<nq,other=0.)
    y=tl.load(QP+(lo+j)*2+1,j<nq,other=0.)
    a_q=tl.load(QZ+row*128+f);b_q=tl.load(QZ+row*128+64+f)
    threshold=tl.load(RADIUS+row)+tl.load(SLACK+row)
    stack=tl.where(slot==0,root,0);sp=tl.where(active,0,-1)
    count=tl.full((),0,tl.int32);nodes=tl.full((),0,tl.int32)
    while sp>=0:
        node=tl.sum(tl.where(slot==sp,stack,0),0);sp-=1;nodes+=1
        if node<G:
            child=node*2+sides
            cx0=tl.load(TBOX+child*4);cy0=tl.load(TBOX+child*4+1)
            cx1=tl.load(TBOX+child*4+2);cy1=tl.load(TBOX+child*4+3)
            distance=box_distance(x[None,:],y[None,:],cx0[:,None],cy0[:,None],cx1[:,None],cy1[:,None])
            query_node_box_sum=tl.sum(tl.where(j[None,:]<nq,distance,0.),1)
            a_bar_U=tl.load(TREE+child[:,None]*128+f[None,:])
            b_under_U=tl.load(TREE+child[:,None]*128+64+f[None,:])
            ell_qU=node_bound_kernel(a_q[None,:],b_q[None,:],a_bar_U,b_under_U,1)
            ell_W_qU=dtw_node_bound_kernel(ell_qU,query_node_box_sum)
            if RADIAL:
                ff=tl.arange(0,16)
                rn_min=tl.load(RT+(child[:,None]*16+ff[None,:])*2)
                rn_max=tl.load(RT+(child[:,None]*16+ff[None,:])*2+1)
                mass=tl.full((2,16),0.,QP.dtype.element_ty)
                for bin in range(4):
                    mu=tl.load(RF+(row*16+ff)*6+2+bin)
                    size=((bin+1)*nq+3)//4-(bin*nq+3)//4
                    mass+=size*tl.maximum(0.,tl.maximum(rn_min-mu[None,:],mu[None,:]-rn_max))
                qmax=tl.max(tl.load(RF+(row*16+ff)*6+1),0)
                guard=64.*1.1920928955078125e-7*nq*(tl.max(rn_max,1)+qmax)
                ell_W_qU=tl.maximum(ell_W_qU,tl.max(mass,1)-guard-2e-4*(1+threshold))
            l0=tl.sum(tl.where(sides==0,ell_W_qU,0.),0)
            l1=tl.sum(tl.where(sides==1,ell_W_qU,0.),0)
            if l1<=threshold:
                sp+=1;stack=tl.where(slot==sp,node*2+1,stack)
            if l0<=threshold:
                sp+=1;stack=tl.where(slot==sp,node*2,stack)
        else:
            tl.store(LEAVES+row*G+part*(G//PARTS)+count,node-G)
            count+=1
    tl.store(COUNTS+row*PARTS+part,count)
    tl.store(VISITS+(row*PARTS+part)*2,nodes)


@triton.jit
def compact_leaves(SOURCE,COUNT,OFFSET,LIDS,QIDS,G:tl.constexpr,BLOCK:tl.constexpr,
                   ROW_DIVISOR:tl.constexpr=1):
    row=tl.program_id(0);j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    n=tl.load(COUNT+row);base=tl.load(OFFSET+row)
    leaf=tl.load(SOURCE+row*G+j,j<n,other=0)
    tl.store(LIDS+base+j,leaf,j<n)
    tl.store(QIDS+base+j,row//ROW_DIVISOR,j<n)


@triton.jit
def merge_subtree_visits(PREFIX,FRONTIER_COUNT,SUBTREE,VISITS,PARTS:tl.constexpr):
    row=tl.program_id(0);j=tl.arange(0,PARTS)
    # Frontier roots were visited by both stages, but count each only once.
    nodes=tl.load(PREFIX+row*2)-tl.load(FRONTIER_COUNT+row)
    nodes+=tl.sum(tl.load(SUBTREE+(row*PARTS+j)*2),0)
    tl.store(VISITS+row*2,nodes)
    tl.store(VISITS+row*2+1,0)


@torch.no_grad()
def enumerate_leaves(queries,tree,extra,qgeom,radius,slack,probe_fields=None,probe_tree=None):
    from leap.common.runtime import options as opt
    n=len(radius);g=tree.groups;device=radius.device
    block=qgeom['block'] if opt.dtw_query_width else max(extra['block'],qgeom['block'])
    parts=128 if opt.dtw_subtree_parallel and n==1 and g>=256 and g&(g-1)==0 else 1
    leaves=torch.empty((n,g),device=device,dtype=torch.int32)
    counts=torch.empty(n*parts,device=device,dtype=torch.int32)
    visits=torch.zeros((n,2),device=device,dtype=torch.int32)
    args=(queries.points,queries.offsets,queries.coordinates,tree.tree,qgeom['box'],
          extra['treebox'],radius,slack,queries.coordinates if probe_tree is None else probe_fields[0],
          tree.tree if probe_tree is None else probe_tree)
    if parts==1:
        leaf_traversal[(n,)](*args,leaves,counts,visits,g,block,probe_tree is not None,
                             num_warps=8,enable_fp_fusion=False)
    else:
        # Each equal-depth subtree has exactly g/parts leaf slots. This bounded
        # allocation covers even an all-pass query without dropping candidates.
        roots=torch.empty((n,parts),device=device,dtype=torch.int32)
        root_counts=torch.empty(n,device=device,dtype=torch.int32)
        prefix_visits=torch.empty((n,2),device=device,dtype=torch.int32)
        subtree_visits=torch.empty((n*parts,2),device=device,dtype=torch.int32)
        leaf_traversal[(n,)](*args,roots,root_counts,prefix_visits,parts,block,probe_tree is not None,
                             num_warps=8,enable_fp_fusion=False)
        leaf_traversal[(n,parts)](*args,leaves,counts,subtree_visits,g,block,probe_tree is not None,
                                  parts,roots,root_counts,num_warps=8,enable_fp_fusion=False)
        merge_subtree_visits[(n,)](prefix_visits,root_counts,subtree_visits,visits,parts)
    return leaves,counts,visits,parts


@triton.jit
def check_leaf_tiles(P,OFF,QP,QOFF,MEM,BOX,QBOX,RADIUS,SLACK,SELF,SEEDS,LIDS,QIDS,KEEP,EXAMINED,
                     B:tl.constexpr,N:tl.constexpr,S:tl.constexpr,TILE:tl.constexpr,SB:tl.constexpr,
                     SORTED:tl.constexpr=False,STEPS:tl.constexpr=0,
                     SHORT_CIRCUIT:tl.constexpr=False,COUNTER=None,RECORD:tl.constexpr=False):
    task=tl.program_id(0);lane=tl.program_id(1)*TILE+tl.arange(0,TILE)
    j=tl.arange(0,N);ss=tl.arange(0,SB)
    row=tl.load(QIDS+task);leaf=tl.load(LIDS+task)
    self_id=tl.load(SELF+row)
    ids=tl.load(MEM+leaf*B+lane,lane<B,other=-1)
    if SORTED:
        # Binary membership avoids a leaf-width x large-seed-set temporary.
        left=tl.full((TILE,),0,tl.int32);right=tl.full((TILE,),S,tl.int32)
        for _ in range(STEPS):
            mid=(left+right)//2
            value=tl.load(SEEDS+row*S+mid,mid<S,other=2147483647)
            left=tl.where(value<ids,mid+1,left)
            right=tl.where(value>=ids,mid,right)
        already=tl.load(SEEDS+row*S+left,left<S,other=-2)==ids
    else:
        initial=tl.load(SEEDS+row*S+ss,ss<S,other=-2)
        already=tl.sum((ids[:,None]==initial[None,:]).to(tl.int32),1)>0
    valid=(lane<B)&(ids>=0)&(ids!=self_id)&(~already)
    tl.store(EXAMINED+task*B+lane,valid.to(tl.int32),lane<B)
    threshold=tl.load(RADIUS+row)+tl.load(SLACK+row)
    if SHORT_CIRCUIT:
        keep,first_checks,second_checks,second_slots=target_first(
            P,OFF,QP,QOFF,BOX,QBOX,row,ids,valid,threshold,N,TILE)
    else:
        lo=tl.load(QOFF+row);nq=tl.load(QOFF+row+1)-lo
        x=tl.load(QP+(lo+j)*2,j<nq,other=0.)
        y=tl.load(QP+(lo+j)*2+1,j<nq,other=0.)
        xmin=tl.load(QBOX+row*4);ymin=tl.load(QBOX+row*4+1)
        xmax=tl.load(QBOX+row*4+2);ymax=tl.load(QBOX+row*4+3)
        bx0=tl.load(BOX+ids*4,valid,other=0.);by0=tl.load(BOX+ids*4+1,valid,other=0.)
        bx1=tl.load(BOX+ids*4+2,valid,other=0.);by1=tl.load(BOX+ids*4+3,valid,other=0.)
        dist=box_distance(x[None,:],y[None,:],bx0[:,None],by0[:,None],bx1[:,None],by1[:,None])
        query_box_sum=tl.sum(tl.where(j[None,:]<nq,dist,0.),1)
        d0=tl.load(OFF+ids,valid,other=0)
        nd=tl.load(OFF+ids+1,valid,other=0)-d0
        u=tl.load(P+(d0[:,None]+j[None,:])*2,valid[:,None]&(j[None,:]<nd[:,None]),other=0.)
        v=tl.load(P+(d0[:,None]+j[None,:])*2+1,valid[:,None]&(j[None,:]<nd[:,None]),other=0.)
        trajectory_box_sum=tl.sum(tl.where(j[None,:]<nd[:,None],box_distance(u,v,xmin,ymin,xmax,ymax),0.),1)
        lower=dtw_individual_bound_kernel(query_box_sum,trajectory_box_sum)
        keep=valid&(lower<=threshold)
        if RECORD:
            first_checks=tl.sum(valid.to(tl.int64),0)*nq
            second_checks=tl.sum(tl.where(valid,nd,0).to(tl.int64),0)
            second_slots=tl.full((),N*TILE,tl.int64)
    tl.store(KEEP+task*B+lane,keep.to(tl.int32),lane<B)
    if RECORD:
        counter=COUNTER+(task*B+tl.program_id(1)*TILE)*4
        tl.store(counter,first_checks);tl.store(counter+1,second_checks)
        tl.store(counter+2,N*TILE);tl.store(counter+3,second_slots)


@triton.jit
def write_pairs(KEEP,LIDS,QIDS,MEM,OFF,DIDS,ROWS,B:tl.constexpr):
    task=tl.program_id(0);lane=tl.arange(0,B)
    keep=tl.load(KEEP+task*B+lane)>0
    leaf=tl.load(LIDS+task);row=tl.load(QIDS+task);base=tl.load(OFF+task)
    ids=tl.load(MEM+leaf*B+lane)
    rank=tl.cumsum(keep.to(tl.int32),0)-1
    tl.store(DIDS+base+rank,ids,keep)
    tl.store(ROWS+base+rank,row,keep)


@torch.no_grad()
def member_task_widths(offsets,query_offsets,members,lids,lrows,b,leaf_max=None):
    if leaf_max is None:
        ids=members.reshape(-1,b)
        safe=ids.clamp_min(0).long()
        lengths=offsets[safe+1]-offsets[safe]
        leaf_max=torch.where(ids>=0,lengths,0).amax(1)
    query_lengths=query_offsets[lrows.long()+1]-query_offsets[lrows.long()]
    return torch.maximum(leaf_max[lids.long()],query_lengths)


@torch.no_grad()
def check_leaf_buckets(points,offsets,query_points,query_offsets,members,boxes,query_boxes,
                       radius,slack,self_ids,seeds,lids,lrows,keep,examined,
                       b,block,s,tile,leaf_max=None):
    if not len(lids):
        return
    from leap.common.runtime.context import current
    from leap.common.runtime import options as opt
    plan = current()
    if plan is not None and s <= 64:
        return plan.check_leaves(points, offsets, query_points, query_offsets, members, boxes,
                                 query_boxes, radius, slack, self_ids, seeds, lids, lrows,
                                 keep, examined, b, block, s, tile)
    widths=member_task_widths(offsets,query_offsets,members,lids,lrows,b,leaf_max)
    lower=0
    for width in [2**i for i in range(min(6,block.bit_length()-1),block.bit_length())]:
        selected=((widths>lower)&(widths<=width)).nonzero().flatten()
        lower=width
        size=len(selected)
        if not size:
            continue
        bucket_keep=torch.empty((size,b),device=keep.device,dtype=keep.dtype)
        bucket_examined=torch.empty_like(bucket_keep)
        bucket_width=bucket_tile(width,b,tile)
        check_leaf_tiles[(size,triton.cdiv(b,bucket_width))](
            points,offsets,query_points,query_offsets,members,boxes,query_boxes,
            radius,slack,self_ids,seeds,lids[selected].contiguous(),lrows[selected].contiguous(),
            bucket_keep,bucket_examined,b,width,s,bucket_width,64,s>64,s.bit_length(),
            SHORT_CIRCUIT=opt.dtw_box_shortcircuit,num_warps=8,enable_fp_fusion=False)
        # Restore the original leaf order before candidate compaction and ranking.
        keep[selected]=bucket_keep
        examined[selected]=bucket_examined


@torch.no_grad()
def enumerate_parallel(queries,index,tree,extra,qgeom,qids,initial,radius,slack,
                       probe_fields=None,probe_tree=None,ragged=True,mark=lambda name:None,
                       *,leaf_enumerator=None):
    from leap.common.runtime import options as opt
    n,s=initial.shape;g=tree.groups;b=tree.block;device=qids.device
    if opt.dtw_length_cache and extra.get('length_signature')!=tuple(
            (t.data_ptr(),t._version) for t in (index.points,index.offsets,tree.members)):
        raise ValueError('DTW length cache changed; rebuild the search index')
    membership=initial.sort(1).values.contiguous() if s>64 else initial
    if n*g>MAX_LEAF_CELLS:
        raise MemoryError('Bounded leaf task buffer; split query batch')
    block=max(extra['block'],qgeom['block'])
    leaves,leaf_counts,visits,parts=(enumerate_leaves if leaf_enumerator is None else leaf_enumerator)(
        queries,tree,extra,qgeom,radius,slack,probe_fields,probe_tree)
    mark('group_traversal')
    if opt.dtw_single_layout and n==1 and g<=32768:
        from leap.common.runtime.context import current
        plan=current()
        if plan is not None:
            from leap.dtw.dtw_single_layout import enumerate_members
            return enumerate_members(plan,leaves,leaf_counts,visits,parts,queries,
                index,tree,extra,qgeom,qids,membership,radius,slack,ragged,mark)
    tasks=int(leaf_counts.sum());max_leaves=int(leaf_counts.max())
    if tasks*b>MAX_MEMBER_CELLS:
        raise MemoryError('Bounded parallel leaf workspace; split query batch')
    leaf_offsets=leaf_counts.long().cumsum(0)-leaf_counts
    lids=torch.empty(tasks,device=device,dtype=torch.int32);lrows=torch.empty_like(lids)
    if tasks:
        compact_leaves[(n*parts,triton.cdiv(max_leaves,128))](
            leaves,leaf_counts,leaf_offsets,lids,lrows,g//parts,128,parts)
    del leaves
    keep=torch.empty((tasks,b),device=device,dtype=torch.int32)
    examined=torch.empty_like(keep)
    mark('leaf_task_layout')
    if tasks:
        # Long point axes otherwise spill large 64 x N temporaries to local memory.
        tile=4 if block>=4096 else b
        check_leaf_buckets(index.points,index.offsets,queries.points,queries.offsets,
            tree.members,extra['box'],qgeom['box'],radius,slack,qids,membership,lids,lrows,keep,examined,
            b,block,s,tile,extra.get('leaf_max') if opt.dtw_length_cache else None)
    mark('parallel_member_filter')
    per_leaf=keep.sum(1,dtype=torch.int32)
    count=torch.zeros(n,device=device,dtype=torch.int32)
    count.scatter_add_(0,lrows.long(),per_leaf)
    visits[:,1].scatter_add_(0,lrows.long(),examined.sum(1,dtype=torch.int32))
    total=int(count.sum());maximum=int(count.max())
    if (total>8_000_000 if ragged else n*(s+maximum)>8_000_000):
        raise MemoryError('DTW range batch exceeds native reduction limit')
    off=count.long().cumsum(0)-count
    pair_offsets=per_leaf.long().cumsum(0)-per_leaf
    dids=torch.empty(total,device=device,dtype=torch.int64);rows=torch.empty_like(dids)
    mark('candidate_layout')
    if tasks:
        write_pairs[(tasks,)](keep,lids,lrows,tree.members,pair_offsets,dids,rows,b,num_warps=4)
    mark('candidate_write')
    return count,visits,dids,rows,off,total,maximum,0
