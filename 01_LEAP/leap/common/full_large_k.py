"""Explicit large-k global search with bounded workspaces and path reporting.

Initial ranking sets a radius, never a limit on database membership. The same
group/member bounds and original-distance refinement as small-k search follow.
"""
from dataclasses import replace

import torch
import triton
import triton.language as tl

from leap.common.interval_index import lex_keys
from leap.common.large_k import coarse_keys, _positive_int, MAX_SCORE_PAIRS
from leap.common.runtime.batching import resolve_query_batch
from leap.common.candidate_policy import tree_initializer


def initial_pool(config, seed_k, available):
    default = {'dfd': 2000, 'dtw': 6400, 'hausdorff': seed_k}[config.distance]
    pool = default if config.initial_pool is None else config.initial_pool
    if config.initial_pool is not None and pool < seed_k:
        raise ValueError('initial_pool must cover seed_k')
    return min(available, max(seed_k, pool))


@triton.jit
def hausdorff_keys(Q,D,SELF,OUT,ND:tl.constexpr):
    row=tl.program_id(1);j=tl.program_id(0)*16+tl.arange(0,16)
    f=tl.arange(0,32);score=tl.full((16,),0.,tl.float32)
    for start in range(0,64,32):
        aq=tl.load(Q+row*128+start+f);bq=tl.load(Q+row*128+64+start+f)
        ad=tl.load(D+j[:,None]*128+start+f[None,:],j[:,None]<ND,other=0.)
        bd=tl.load(D+j[:,None]*128+64+start+f[None,:],j[:,None]<ND,other=0.)
        score=tl.maximum(score,tl.max(tl.maximum(bq[None,:]-ad,bd-aq[None,:]),1))
    score=tl.where(j==tl.load(SELF+row),float('inf'),score)
    key=(score.to(tl.int32,bitcast=True).to(tl.int64)<<32)|j.to(tl.int64)
    tl.store(OUT+row*ND+j,key,j<ND)


@triton.jit
def dtw_keys(Q,D,QO,DO,SELF,OUT,ND:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    valid=j<ND
    n=tl.load(QO+row+1)-tl.load(QO+row)
    m=tl.load(DO+j+1,valid,other=0)-tl.load(DO+j,valid,other=0)
    total=tl.full((BLOCK,),0.,tl.float32)
    for b in range(8):
        dx=tl.load(Q+row*16+b*2)-tl.load(D+j*16+b*2,valid,other=0.)
        dy=tl.load(Q+row*16+b*2+1)-tl.load(D+j*16+b*2+1,valid,other=0.)
        total+=tl.sqrt(dx*dx+dy*dy)
    score=tl.maximum(n,m)*(total/8.)
    score=tl.where(j==tl.load(SELF+row),float('inf'),score)
    key=(score.to(tl.int32,bitcast=True).to(tl.int64)<<32)|j.to(tl.int64)
    tl.store(OUT+row*ND+j,key,valid)


def subset(index,lo,hi):
    from leap.common.large_k import subset as base_subset
    q=base_subset(index,lo,hi)
    if index.ordered_summary is not None:
        q=replace(q,ordered_summary=index.ordered_summary[lo:hi])
    if index.dtw_probe_summary is not None:
        from leap.dtw.dtw_probes import slice_summary
        q=replace(q,dtw_probe_summary=slice_summary(index.dtw_probe_summary,lo,hi))
    return q


@torch.no_grad()
def initial_candidates(model,q,d,qids,k,self_start=None,*,explicit_self_ids=False,search_index=None):
    nq,nd=len(q.coordinates),len(d.coordinates)
    metric=model.config.distance
    available=nd-int(self_start is not None or (explicit_self_ids and bool((qids>=0).any())))
    pool=initial_pool(model.config,k,available)
    use_tree=tree_initializer(model.config,pool,has_index=search_index is not None)
    if use_tree:
        search_index.validate(d)
        if metric=='hausdorff':
            ids=search_index.candidates(q.coordinates,qids,pool)[0]
        elif metric=='dfd':
            from leap.dfd.dfd_tree import candidate_pool
            ids=candidate_pool(q,d,search_index,qids,pool)[0]
        else:
            routing=getattr(search_index,'dtw_routing_tree',None)
            if routing is None:
                routing=model._routing_tree(d)
            self_ids=qids if self_start is not None or explicit_self_ids else None
            ids=routing.candidates(q,pool,self_ids)
    else:
        if nq*nd>MAX_SCORE_PAIRS:
            raise MemoryError('Split large-k score generation to respect the workspace bound')
        if metric=='dfd' or (metric=='hausdorff' and model.config.hausdorff_target=='pointset'):
            keys=coarse_keys(q,d,metric,self_start=self_start)
        else:
            keys=torch.empty((nq,nd),device=q.points.device,dtype=torch.int64)
            if metric=='hausdorff':
                hausdorff_keys[(triton.cdiv(nd,16),nq)](q.coordinates,d.coordinates,qids,keys,nd,
                    num_warps=4,enable_fp_fusion=False)
            else:
                dtw_keys[(nq,triton.cdiv(nd,128))](q.ordered_summary,d.ordered_summary,
                    q.offsets,d.offsets,qids,keys,nd,128,num_warps=4,enable_fp_fusion=False)
        if explicit_self_ids:
            valid=qids>=0
            rows=torch.arange(nq,device=keys.device)[valid]
            keys[rows,qids[valid]]=torch.iinfo(torch.int64).max
        ids=(keys.topk(pool,largest=False).values&0xffffffff).contiguous()
        del keys
    if metric=='hausdorff':
        return ids
    rows=torch.arange(nq,device=q.points.device)[:,None].expand_as(ids).reshape(-1)
    flat=ids.reshape(-1)
    if metric=='dfd':
        from leap.dfd.dfd import dfd_pair_bounds
        from leap.dfd.dfd_geometry import witness_bound, ordered_bound
        scores=torch.maximum(dfd_pair_bounds(q,d,rows,flat),torch.maximum(
            witness_bound(q,d,rows,flat,8),ordered_bound(q,d,rows,flat,8)))
    else:
        from leap.dtw.dtw_signal_bucket import pairs
        options={}
        if use_tree and getattr(routing,'signal_plan',None) is not None:
            options['batch_plan']=routing.signal_plan
        scores=pairs(q,d,rows,flat,fields=(q.dtw_probe_summary['signal'],d.dtw_probe_summary['signal']),**options)
    return (lex_keys(scores.reshape_as(ids),ids).topk(k,largest=False).values&0xffffffff).contiguous()


@torch.no_grad()
def search(model,queries,database=None,*,search_index=None,top_k=1000,seed_k=None,
           query_batch=None,progress=None):
    if not model.config.full_large_k:
        raise ValueError('Enable full_large_k=True for the explicit optimized global-search path')
    top_k=_positive_int(top_k,'top_k')
    seed_k=top_k if seed_k is None else _positive_int(seed_k,'seed_k')
    if seed_k<top_k:
        raise ValueError('seed_k must cover top_k')
    same=database is None or database is queries
    database=queries if database is None else database
    model._check_index(queries);model._check_index(database)
    if search_index is None:
        raise ValueError('Build the reusable full-database search_index first')
    search_index.validate(database)
    metric=model.config.distance
    if metric=='dtw' and search_index.block!=64:
        raise ValueError('full_large_k DTW requires leaf_capacity=64; no serial fallback')
    nd=len(database.coordinates);nq=len(queries.coordinates)
    if seed_k>nd-int(same):
        raise ValueError('Database has fewer available trajectories than seed_k')
    use_tree=tree_initializer(model.config,initial_pool(model.config,seed_k,nd-int(same)))
    batch_options={} if use_tree else dict(default=128,database_size=nd,max_score_pairs=MAX_SCORE_PAIRS)
    batch=resolve_query_batch(model.config,query_batch,**batch_options)
    if metric=='dfd' and getattr(search_index,'dfd_versions',None)!=database.dfd_signature:
        raise ValueError('DFD tree is missing or has stale retained geometry; rebuild')
    if metric=='dtw':
        from leap.dtw.dtw_range import sidecars
        versions=(database.points.data_ptr(),database.points._version,database.offsets.data_ptr(),database.offsets._version)
        if getattr(search_index,'dtw_versions',None)!=versions or getattr(search_index,'dtw_probe_versions',None)!=database.dtw_probe_signature:
            raise ValueError('DTW tree has missing/stale geometry or probe envelopes; rebuild')
        geometry=search_index.dtw_geometry if same else sidecars(queries)
    outputs=[]
    metadata={'version':'full-large-k-v1','full_database':True,'top_k':top_k,'seed_k':seed_k,
              'initializer':'tree-score-id-keys' if use_tree else 'fused-score-id-keys',
              'initial_pool':initial_pool(model.config,seed_k,nd-int(same)),
              'query_batch_limit':batch,'batches':0,'batch_splits':0,
              'enumeration':'parallel-leaves-sorted-seeds' if metric=='dtw' else 'native-tree',
              'ordered_filter':metric=='dtw','serial_fallback':False}
    metadata['initial_pool']=min(metadata['initial_pool'],nd-int(same))

    def process(lo,hi,dtw_cache=None):
        q=subset(queries,lo,hi)
        ids=torch.arange(lo,hi,device=model.device,dtype=torch.int64) if same else torch.full((hi-lo,),-1,device=model.device,dtype=torch.int64)
        try:
            seeds=(initial_candidates(model,q,database,ids,seed_k,lo if same else None,search_index=search_index)
                   if dtw_cache is None else dtw_cache[0])
            if metric=='dtw':
                from leap.dtw.dtw_range import search as run, initial_distances
                from leap.common.runtime import options as opt
                if dtw_cache is None and hi-lo>1 and opt.dtw_initial_cache:
                    dtw_cache=(seeds,initial_distances(q,database,seeds,bucket=True,
                        maximum=search_index.dtw_geometry['block'] if opt.dtw_length_cache else None))
                a,b=q.dtw_probe_summary,database.dtw_probe_summary
                result=run(q,database,search_index,search_index.dtw_geometry,
                    {'box':geometry['box'][lo:hi],'block':geometry['block']},ids,seeds,top_k,
                    probe_fields=(a['radial'],b['radial']),probe_tree=search_index.dtw_probe_tree,
                    ordered_fields=(a['ordered'],b['ordered']),ragged=True,bounded_dtw=True,
                    bucket_dtw=True,parallel_leaves=True,large_k_parallel=True,
                    dynamic_threshold=model.config.dynamic_threshold,
                    raw_initial=None if dtw_cache is None else dtw_cache[1])
                if result['enumeration']!='parallel-leaves':
                    raise RuntimeError('Requested optimized large-k execution did not run')
            elif metric=='dfd':
                from leap.dfd.dfd_range import search as run
                result=run(q,database,search_index,ids,seeds,top_k,
                           dynamic_threshold=model.config.dynamic_threshold)
            else:
                from leap.hausdorff.range_search import search_chunk
                result=search_chunk(q,database,search_index,ids,seed_k,top_k,initial=seeds,
                                    point=model.config.hausdorff_target=='pointset')
        except MemoryError:
            if hi-lo<=1:
                raise
            metadata['batch_splits']+=1
            mid=(lo+hi)//2
            left=None if dtw_cache is None else tuple(v[:mid-lo] for v in dtw_cache)
            right=None if dtw_cache is None else tuple(v[mid-lo:] for v in dtw_cache)
            process(lo,mid,left);process(mid,hi,right)
            return
        outputs.append({key:result[key] for key in ('indices','distances','counts','radius')})
        metadata['batches']+=1
        if progress is not None:
            progress(hi,nq)

    for lo in range(0,nq,batch):
        process(lo,min(lo+batch,nq))
    result={key:torch.cat([row[key] for row in outputs]) for key in outputs[0]}
    result.update(full_search=True,execution=metadata)
    return result
