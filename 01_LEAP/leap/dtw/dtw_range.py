"""DTW instantiation of GEF -> TBF -> RGR (eq:dtw-node / eq:dtw-individual)."""
import torch
import triton
import triton.language as tl
from leap.common.raw_geometry_refinement import dtw_distances
from leap.common.interval_index import lex_keys
from leap.common.group_envelope_filter import node_bound_kernel, dtw_node_bound_kernel
from leap.common.trajectory_bound_filter import dtw_individual_bound_kernel


@triton.jit
def boxes_kernel(P,OFF,BOX,LENGTH,N:tl.constexpr):
    row=tl.program_id(0)
    lo=tl.load(OFF+row);n=tl.load(OFF+row+1)-lo
    j=tl.arange(0,N)
    x=tl.load(P+(lo+j)*2,j<n,other=0.)
    y=tl.load(P+(lo+j)*2+1,j<n,other=0.)
    tl.store(BOX+row*4,tl.min(tl.where(j<n,x,float('inf')),0))
    tl.store(BOX+row*4+1,tl.min(tl.where(j<n,y,float('inf')),0))
    tl.store(BOX+row*4+2,tl.max(tl.where(j<n,x,-float('inf')),0))
    tl.store(BOX+row*4+3,tl.max(tl.where(j<n,y,-float('inf')),0))
    tl.store(LENGTH+row,n)


@torch.no_grad()
def sidecars(index,tree=None):
    from leap.common.runtime.context import current
    plan = current()
    n=len(index.coordinates)
    maximum=plan.query_length if plan is not None else int((index.offsets[1:]-index.offsets[:-1]).max())
    box=plan.storage('query_box', (n,4)) if plan is not None else torch.empty((n,4),device=index.points.device,dtype=index.points.dtype)
    lengths=plan.storage('query_lengths',(n,),torch.int32) if plan is not None else torch.empty(n,device=index.points.device,dtype=torch.int32)
    boxes_kernel[(n,)](index.points,index.offsets,box,lengths,triton.next_power_of_2(maximum),num_warps=4)
    if tree is None:
        return {'box':box,'block':triton.next_power_of_2(maximum)}
    member=tree.members
    b=box[member.clamp_min(0)]
    valid=(member>=0)[...,None]
    bmin=b[:,:,:2].masked_fill(~valid,torch.inf).amin(1)
    bmax=b[:,:,2:].masked_fill(~valid,-torch.inf).amax(1)
    boxes=torch.empty((tree.groups*2,4),device=index.points.device,dtype=index.points.dtype)
    boxes[tree.groups:]=torch.cat((bmin,bmax),1)
    level=tree.groups//2
    while level:
        ids=torch.arange(level,2*level,device=index.points.device)
        boxes[ids,:2]=torch.minimum(boxes[ids*2,:2],boxes[ids*2+1,:2])
        boxes[ids,2:]=torch.maximum(boxes[ids*2,2:],boxes[ids*2+1,2:])
        level//=2
    leaf_max=torch.where(member>=0,lengths[member.clamp_min(0)],0).amax(1)
    return {'box':box,'treebox':boxes,'lengths':lengths,'block':triton.next_power_of_2(maximum),
            'leaf_max':leaf_max,'length_signature':tuple((t.data_ptr(),t._version)
                for t in (index.points,index.offsets,tree.members))}


@triton.jit
def box_distance(x,y,xmin,ymin,xmax,ymax):
    dx=tl.maximum(tl.maximum(xmin-x,x-xmax),0.)
    dy=tl.maximum(tl.maximum(ymin-y,y-ymax),0.)
    return tl.sqrt(dx*dx+dy*dy)


@triton.jit
def enumerate_sum(P,OFF,QP,QOFF,QZ,MEM,TREE,BOX,QBOX,TBOX,RADIUS,SLACK,QIDS,SEEDS,RF,RT,BASE,
                  OUTD,OUTQ,COUNTS,VISITS,G:tl.constexpr,B:tl.constexpr,
                  N:tl.constexpr,SEED_K:tl.constexpr,WRITE:tl.constexpr,RADIAL:tl.constexpr,
                  CAPACITY:tl.constexpr=0,MAP_ROWS:tl.constexpr=False,ROWMAP=None,
                  INITIAL_BLOCK:tl.constexpr=64):
    row=tl.program_id(0)
    if MAP_ROWS:
        row=tl.load(ROWMAP+row)
    qid=tl.load(QIDS+row)
    j=tl.arange(0,N);f=tl.arange(0,64);lane=tl.arange(0,B)
    slot=tl.arange(0,32);sides=tl.arange(0,2);ss=tl.arange(0,INITIAL_BLOCK)
    lo=tl.load(QOFF+row);nq=tl.load(QOFF+row+1)-lo
    x=tl.load(QP+(lo+j)*2,j<nq,other=0.)
    y=tl.load(QP+(lo+j)*2+1,j<nq,other=0.)
    xmin=tl.load(QBOX+row*4);ymin=tl.load(QBOX+row*4+1)
    xmax=tl.load(QBOX+row*4+2);ymax=tl.load(QBOX+row*4+3)
    a_q=tl.load(QZ+row*128+f);b_q=tl.load(QZ+row*128+64+f)
    initial_ids=tl.load(SEEDS+row*SEED_K+ss,ss<SEED_K,other=-2)
    threshold=tl.load(RADIUS+row)+tl.load(SLACK+row)
    stack=tl.where(slot==0,1,0);sp=0
    count=tl.full((),0,tl.int32);nodes=tl.full((),0,tl.int32);examined=tl.full((),0,tl.int32)
    while sp>=0:
        node=tl.sum(tl.where(slot==sp,stack,0),0);sp-=1;nodes+=1
        if node<G:
            child=node*2+sides
            # Union boxes contain every member vertex: the query-to-union sum is admissible.
            cx0=tl.load(TBOX+child*4);cy0=tl.load(TBOX+child*4+1)
            cx1=tl.load(TBOX+child*4+2);cy1=tl.load(TBOX+child*4+3)
            distance=box_distance(x[None,:],y[None,:],cx0[:,None],cy0[:,None],cx1[:,None],cy1[:,None])
            query_node_box_sum=tl.sum(tl.where(j[None,:]<nq,distance,0.),1)
            a_bar_U=tl.load(TREE+child[:,None]*128+f[None,:])
            b_under_U=tl.load(TREE+child[:,None]*128+64+f[None,:])
            # GEF, eq:dtw-node: ell_W(q,U).
            ell_qU=node_bound_kernel(a_q[None,:],b_q[None,:],a_bar_U,b_under_U,1)
            ell_W_qU=dtw_node_bound_kernel(ell_qU,query_node_box_sum)
            if RADIAL:
                ff=tl.arange(0,16)
                rn_min=tl.load(RT+(child[:,None]*16+ff[None,:])*2)
                rn_max=tl.load(RT+(child[:,None]*16+ff[None,:])*2+1)
                mass=tl.full((2,16),0.,tl.float32)
                for bin in range(4):
                    mu=tl.load(RF+(row*16+ff)*6+2+bin)
                    size=((bin+1)*nq+3)//4-(bin*nq+3)//4
                    mass+=size*tl.maximum(0.,tl.maximum(rn_min-mu[None,:],mu[None,:]-rn_max))
                # A node interval contains every member's radial interval.
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
            ids=tl.load(MEM+(node-G)*B+lane)
            valid=(ids>=0)&(ids!=qid)
            already=tl.sum((ids[:,None]==initial_ids[None,:]).to(tl.int32),1)>0
            valid=valid&(~already)
            bx0=tl.load(BOX+ids*4,valid,other=0.)
            by0=tl.load(BOX+ids*4+1,valid,other=0.)
            bx1=tl.load(BOX+ids*4+2,valid,other=0.)
            by1=tl.load(BOX+ids*4+3,valid,other=0.)
            dist=box_distance(x[None,:],y[None,:],bx0[:,None],by0[:,None],bx1[:,None],by1[:,None])
            query_box_sum=tl.sum(tl.where(j[None,:]<nq,dist,0.),1)
            d0=tl.load(OFF+ids,valid,other=0)
            nd=tl.load(OFF+ids+1,valid,other=0)-d0
            u=tl.load(P+(d0[:,None]+j[None,:])*2,valid[:,None]&(j[None,:]<nd[:,None]),other=0.)
            v=tl.load(P+(d0[:,None]+j[None,:])*2+1,valid[:,None]&(j[None,:]<nd[:,None]),other=0.)
            trajectory_box_sum=tl.sum(tl.where(j[None,:]<nd[:,None],box_distance(u,v,xmin,ymin,xmax,ymax),0.),1)
            # TBF, eq:dtw-individual: L_W(q,T).
            L_W_qT=dtw_individual_bound_kernel(query_box_sum,trajectory_box_sum)
            keep=valid&(L_W_qT<=threshold)
            if WRITE:
                base=tl.load(BASE+row)
                rank=tl.cumsum(keep.to(tl.int32),0)-1
                writable=keep
                if CAPACITY:
                    writable=writable&(count+rank<CAPACITY)
                tl.store(OUTD+base+count+rank,ids,writable)
                tl.store(OUTQ+base+count+rank,row,writable)
            count+=tl.sum(keep.to(tl.int32),0)
            examined+=tl.sum(valid.to(tl.int32),0)
    tl.store(COUNTS+row,count)
    tl.store(VISITS+row*2,nodes);tl.store(VISITS+row*2+1,examined)


@triton.jit
def compact_rows(SOURCE,COUNTS,OFF,OUTD,OUTQ,CAP:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    count=tl.load(COUNTS+row)
    valid=(count<=CAP)&(j<count)&(j<CAP)
    ids=tl.load(SOURCE+row*CAP+j,valid,other=0)
    base=tl.load(OFF+row)
    tl.store(OUTD+base+j,ids,valid)
    tl.store(OUTQ+base+j,row,valid)


@torch.no_grad()
def initial_distances(queries,index,initial,*,bucket=False,check=8,maximum=None):
    rows=torch.arange(len(initial),device=initial.device)[:,None].expand_as(initial).reshape(-1)
    ids=initial.reshape(-1)
    if bucket:
        from leap.dtw.dtw_bounded import pairs
        values=pairs(queries,index,rows,ids,bucket=True,check=check,maximum=maximum)
    else:
        values=dtw_distances(queries,index,rows,ids)
    return values.reshape_as(initial)


@torch.no_grad()
def search(queries,index,tree,extra,qgeom,qids,initial,top_k=20,probe_fields=None,probe_tree=None,profile=False,ordered_fields=None,ragged=False,
           bounded_dtw=False,bucket_dtw=False,one_pass_capacity=0,check_interval=8,parallel_leaves=False,
           large_k_parallel=False,raw_initial=None,dynamic_threshold=False):
    """initial is S(q); r_hat_q = kth_{T in S(q)} D_W(q,T), eq:search-radius."""
    n,s=initial.shape
    if large_k_parallel and (not parallel_leaves or tree.block!=64):
        raise ValueError('Large-k parallel execution requires parallel_leaves and leaf_capacity=64')
    if dynamic_threshold and n>1 and top_k==20 and s<=64 and ragged and bounded_dtw and bucket_dtw and parallel_leaves and not profile:
        from leap.common.runtime.context import current
        if current() is None and queries.points.dtype==torch.float32:
            from leap.dtw.dynamic_search import search as dynamic_search
            return dynamic_search(queries,index,tree,extra,qgeom,qids,initial,top_k,
                probe_fields=probe_fields,probe_tree=probe_tree,ordered_fields=ordered_fields,
                check_interval=check_interval,raw_initial=raw_initial)
    stamps=[]
    def mark(name):
        if profile:
            ev=torch.cuda.Event(enable_timing=True);ev.record();stamps.append((name,ev))
    mark('start')
    qrows=torch.arange(n,device=qids.device)
    def raw(qr,dr,limits=None):
        if bucket_dtw or limits is not None:
            from leap.common.runtime import options as opt
            from leap.dtw.dtw_bounded import pairs
            return pairs(queries,index,qr,dr,limits,bucket=bucket_dtw,check=check_interval,
                         maximum=extra['block'] if opt.dtw_length_cache else None)
        return dtw_distances(queries,index,qr,dr)
    if raw_initial is None:
        from leap.common.runtime import options as opt
        D_S_q=initial_distances(queries,index,initial,bucket=bucket_dtw,check=check_interval,
                               maximum=extra['block'] if opt.dtw_length_cache else None)
    else:
        if (raw_initial.shape!=initial.shape or raw_initial.device!=initial.device
                or raw_initial.dtype!=queries.points.dtype):
            raise ValueError('Cached initial DTW values must match the initial candidate matrix')
        D_S_q=raw_initial
    r_hat_q=D_S_q.topk(top_k,largest=False).values[:,-1].contiguous()
    mark('initial_raw')
    slack=(2e-5*(1+r_hat_q+queries.coordinates.abs().amax(1)+tree.database_scale)).contiguous()
    used_parallel=parallel_leaves and tree.block==64 and (s<=64 or large_k_parallel)
    if used_parallel:
        from leap.dtw.dtw_parallel import enumerate_parallel
        count,visits,dids,rows,off,total,maximum,overflow_rows=enumerate_parallel(
            queries,index,tree,extra,qgeom,qids,initial,r_hat_q,slack,
            probe_fields,probe_tree,ragged,mark)
    else:
        count=torch.empty(n,dtype=torch.int32,device=qids.device)
        visits=torch.empty((n,2),dtype=torch.int32,device=qids.device)
        args=(index.points,index.offsets,queries.points,queries.offsets,queries.coordinates,
              tree.members,tree.tree,extra['box'],qgeom['box'],extra['treebox'],r_hat_q,slack,qids,initial,
              queries.coordinates if probe_tree is None else probe_fields[0],
              tree.tree if probe_tree is None else probe_tree)
        block=max(extra['block'],qgeom['block'])
        if one_pass_capacity:
            if one_pass_capacity<1 or n*one_pass_capacity>8_000_000:
                raise MemoryError('Single-pass candidate storage exceeds bounded allocation')
            staging=torch.empty(n*one_pass_capacity,dtype=torch.int64,device=qids.device)
            staging_rows=torch.empty_like(staging)
            bases=qrows*one_pass_capacity
            enumerate_sum[(n,)](*args,bases,staging,staging_rows,count,visits,tree.groups,tree.block,block,s,True,
                                probe_tree is not None,one_pass_capacity,INITIAL_BLOCK=max(64,triton.next_power_of_2(s)),
                                num_warps=8,enable_fp_fusion=False)
            del staging_rows
        else:
            enumerate_sum[(n,)](*args,count,count,count,count,visits,tree.groups,tree.block,block,s,False,
                                probe_tree is not None,INITIAL_BLOCK=max(64,triton.next_power_of_2(s)),
                                num_warps=8,enable_fp_fusion=False)
        mark('count_pass')
        total=int(count.sum());maximum=int(count.max())
        if (total>8_000_000 if ragged else n*(s+maximum)>8_000_000):
            raise MemoryError('DTW range batch exceeds bounded reduction; split query batch')
        off=count.long().cumsum(0)-count
        dids=torch.empty(total,dtype=torch.int64,device=qids.device);rows=torch.empty_like(dids)
        overflow_rows=0
        if one_pass_capacity:
            compact_rows[(n,triton.cdiv(one_pass_capacity,256))](staging,count,off,dids,rows,one_pass_capacity,256)
            overflow=(count>one_pass_capacity).nonzero().flatten()
            overflow_rows=len(overflow)
            if overflow_rows:
                written=count.clone()
                # Replay only overflowing queries with their full, exactly counted space.
                enumerate_sum[(overflow_rows,)](*args,off,dids,rows,written,visits,tree.groups,tree.block,block,s,True,
                    probe_tree is not None,0,True,overflow,INITIAL_BLOCK=max(64,triton.next_power_of_2(s)),
                    num_warps=8,enable_fp_fusion=False)
                if not torch.equal(count,written):
                    raise RuntimeError('DTW overflow replay counts changed')
            del staging
        else:
            written=torch.empty_like(count)
            enumerate_sum[(n,)](*args,off,dids,rows,written,visits,tree.groups,tree.block,block,s,True,
                                probe_tree is not None,INITIAL_BLOCK=max(64,triton.next_power_of_2(s)),
                                num_warps=8,enable_fp_fusion=False)
            if not torch.equal(count,written):
                raise RuntimeError('DTW range counts changed')
        mark('write_pass')
    # RGR, eq:dtw-target. Initial D_W values are reused below.
    raw_pairs=total
    if probe_fields is None and ordered_fields is None:
        mark('probe_filter')
        budgets=(r_hat_q+slack)[rows] if bounded_dtw else None
        values=raw(rows,dids,budgets)
    else:
        keep=torch.ones_like(rows,dtype=torch.bool)
        if probe_fields is not None:
            from leap.dtw.dtw_radial import pairs
            lower=pairs(probe_fields[0],probe_fields[1],queries.offsets,index.offsets,rows,dids)
            keep=lower<=r_hat_q[rows]+2e-4*(1+r_hat_q[rows])
        values=torch.full((total,),torch.inf,device=qids.device)
        mark('probe_filter')
        if ordered_fields is not None:
            from leap.dtw.dtw_probe_intervals import keep_pairs as ordered_keep
            selected=keep.nonzero().flatten()
            limits=r_hat_q[rows[selected]]+2e-4*(1+r_hat_q[rows[selected]])
            keep[selected]=ordered_keep(ordered_fields[0],ordered_fields[1],queries.offsets,index.offsets,
                                        rows[selected],dids[selected],limits)
            mark('ordered_filter')
        qr,dr=rows[keep],dids[keep]
        budgets=(r_hat_q+slack)[qr] if bounded_dtw else None
        values[keep]=raw(qr,dr,budgets)
        raw_pairs=int(keep.sum())
    mark('raw_dtw')
    if ragged:
        from leap.dtw.dtw_ragged import topk
        top=topk(initial,D_S_q,dids,values,off,count,top_k)
    else:
        keys=torch.full((n,s+maximum),9187343244130779135,dtype=torch.int64,device=qids.device)
        keys[:,:s]=lex_keys(D_S_q,initial)
        column=torch.arange(total,device=qids.device)-off[rows]+s
        keys[rows,column]=lex_keys(values,dids)
        top=keys.topk(top_k,largest=False).values
    mark('sort')
    times={}
    if profile:
        stamps[-1][1].synchronize()
        times={stamps[i][0]:stamps[i-1][1].elapsed_time(stamps[i][1]) for i in range(1,len(stamps))}
    return {'indices':top&0xffffffff,'distances':(top>>32).int().contiguous().view(torch.float32),
            'counts':count+s,'new_counts':count,'visits':visits,'radius':r_hat_q,
            'raw_extra_pairs':raw_pairs,'profile_ms':times,'overflow_rows':overflow_rows,
            'enumeration':'parallel-leaves' if used_parallel else 'serial'}
