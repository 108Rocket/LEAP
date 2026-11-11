"""Opt-in FP64 decision geometry; approximate routing remains the frozen model.

All database candidates still enter the same GEF/TBF/RGR hierarchy. FP32
ranking only initializes a radius; it never certifies exclusion in this path.
The original empirical pruning slack is retained conservatively, not promoted
to a proof of exact real arithmetic. No truth or reference file is read here.
"""
from copy import copy
from dataclasses import replace
import threading
import time

import numpy as np
import torch

from leap.common.model import TrajectoryIndex


def ranking_model(model):
    ranker = copy(model)
    ranker.config = replace(model.config, precision='fp32')
    if not model.asset_id.endswith(':fp64'):
        raise ValueError('FP64 model identity is missing')
    ranker.asset_id = model.asset_id[:-5]
    return ranker


def signature(index):
    tensors = [index.points,index.offsets,index.coordinates]
    if index.dtw_probe_summary:
        tensors.extend(index.dtw_probe_summary.values())
    return tuple((v.data_ptr(),v._version,tuple(v.shape),v.dtype,v.device) for v in tensors)


def validate(model,index):
    if not isinstance(index,TrajectoryIndex) or index.asset_id != model.asset_id:
        raise ValueError('Different model or precision; re-encode and rebuild')
    if (index.points.dtype != torch.float64 or index.coordinates.dtype != torch.float64
            or index.points.device != model.device or index.coordinates.device != model.device
            or index.ranking_index is None or index.precision_signature != signature(index)):
        raise ValueError('Missing, stale or mixed FP64 geometry')
    ranking_model(model)._check_index(index.ranking_index)


@torch.no_grad()
def encode(model,paths):
    from leap.common.data import checked_pack
    from leap.common.dual_field import fields_packed
    paths = list(paths)
    if any(np.asarray(p).dtype.kind == 'f' and np.asarray(p).dtype.itemsize < 8 for p in paths):
        raise ValueError('FP64 requires original float64 input, not previously rounded FP32 coordinates')
    raw, offsets = checked_pack(paths)
    normalized = (raw.astype(np.float64)-model.mean)/model.std
    if not np.isfinite(normalized).all():
        raise ValueError('Nonfinite FP64 normalized geometry')
    ranked = ranking_model(model).encode(paths)
    points = torch.from_numpy(np.ascontiguousarray(normalized)).to(model.device)
    off = ranked.offsets
    fields = torch.cat([fields_packed(points,off[lo:lo+model.batch_size+1],model.anchors.double())
                        for lo in range(0,len(off)-1,model.batch_size)])
    if model.config.hausdorff_target == 'pointset':
        fields = torch.cat((fields[:,:64],fields[:,:64]),1).contiguous()
    if not torch.isfinite(fields).all():
        raise ValueError('Nonfinite FP64 dual fields')
    index = TrajectoryIndex(fields,points,off,model.asset_id,
        dfd_summary=ranked.dfd_summary,ranking_index=ranked)
    if model.config.distance == 'dtw' and model.dtw_probes is not None:
        from leap.dtw import dtw_radial
        from leap.dtw import dtw_probe_intervals
        summary = {'radial':dtw_radial.encode(index,model.dtw_probes['radial'].double()),
                   'ordered':dtw_probe_intervals.encode(index,model.dtw_probes['ordered'].double())}
        if any(not bool(torch.isfinite(t).all()) for t in summary.values()):
            raise ValueError('Nonfinite FP64 pruning summaries')
        index = replace(index,dtw_probe_summary=summary)
    return replace(index,precision_signature=signature(index))


class PrecisionSearchIndex:
    @torch.no_grad()
    def __init__(self,model,database,block):
        from leap.common.group_envelope_filter import node_envelope
        self.database=database
        self.signature=signature(database)
        self.ranking_tree=ranking_model(model).build_search_index(database.ranking_index,block)
        self.members=self.ranking_tree.members
        self.block=block
        self.groups=self.ranking_tree.groups
        self.d=database.coordinates
        self.database_scale=self.d.abs().amax()
        self.tree=torch.zeros((2*self.groups,128),device=self.d.device,dtype=torch.float64)
        # Bound the gathered leaf tensor independently of database size.
        for lo in range(0,self.groups,1024):
            ids=self.members[lo:lo+1024]
            self.tree[self.groups+lo:self.groups+lo+len(ids)]=node_envelope(
                self.d[ids.clamp_min(0)],(ids>=0)[...,None])
        level=self.groups//2
        while level:
            ids=torch.arange(level,2*level,device=self.d.device)
            self.tree[ids,:64]=torch.maximum(self.tree[2*ids,:64],self.tree[2*ids+1,:64])
            self.tree[ids,64:]=torch.minimum(self.tree[2*ids,64:],self.tree[2*ids+1,64:])
            level//=2
        if model.config.distance=='dfd':
            from leap.dfd.dfd_tree import prepare
            self.dfd_geometry=prepare(database,self)
        if model.config.distance=='dtw':
            from leap.dtw.dtw_range import sidecars
            self.dtw_geometry=sidecars(database,self)
            self.dtw_probe_tree=None
            if database.dtw_probe_summary is not None:
                from leap.dtw.dtw_radial import node_envelopes
                self.dtw_probe_tree=node_envelopes(database.dtw_probe_summary['radial'],self)

    def validate(self,database):
        if database is not self.database or signature(database)!=self.signature:
            raise ValueError('FP64 tree belongs to another or modified database')
        self.ranking_tree.validate(database.ranking_index)

    @property
    def extra_bytes(self):
        def tensors(value):
            if isinstance(value,torch.Tensor):
                yield value
            elif isinstance(value,dict):
                for item in value.values():
                    yield from tensors(item)
            elif isinstance(value,(tuple,list)):
                for item in value:
                    yield from tensors(item)
        items=list(tensors([self.tree,getattr(self,'dfd_geometry',None),
                           getattr(self,'dtw_geometry',None),getattr(self,'dtw_probe_tree',None)]))
        unique={v.data_ptr():v for v in items}
        return self.ranking_tree.extra_bytes+sum(v.numel()*v.element_size() for v in unique.values())


def subset(index,lo,hi):
    from leap.common.full_large_k import subset as ranking_subset
    result=replace(index,coordinates=index.coordinates[lo:hi],offsets=index.offsets[lo:hi+1],
                   ranking_index=ranking_subset(index.ranking_index,lo,hi))
    if index.dtw_probe_summary:
        result=replace(result,dtw_probe_summary={k:v[lo:hi] for k,v in index.dtw_probe_summary.items()})
    return replace(result,precision_signature=signature(result))


def initial(model,q,d,tree,self_ids,count):
    ranker=ranking_model(model)
    a,b=q.ranking_index,d.ranking_index
    if count>64 or model.config.full_large_k:
        from leap.common.full_large_k import initial_candidates
        return initial_candidates(ranker,a,b,self_ids,count,explicit_self_ids=True)
    if model.config.distance=='hausdorff':
        return tree.ranking_tree.candidates(a.coordinates,self_ids,count)[0]
    if model.config.distance=='dfd':
        return ranker._dfd_candidates(a,b,tree.ranking_tree,self_ids,count)
    return ranker._dtw_candidates(a,b,count,self_ids,
        routing_tree=getattr(tree.ranking_tree,'dtw_routing_tree',None))


def raw_pairs(metric,q,d,rows,ids,point=False):
    if metric=='hausdorff':
        from leap.common.raw_geometry_refinement import hausdorff_distances
        return hausdorff_distances(q,d,rows,ids,point)
    if metric=='dfd':
        from leap.dfd.dfd import dfd_pairs_bucketed
        return dfd_pairs_bucketed(q,d,rows,ids)
    from leap.dtw.dtw_bounded import pairs
    return pairs(q,d,rows,ids,bucket=True)


def stable_top(values,ids,k):
    if values.dtype != torch.float64:
        raise ValueError('FP64 final scores must never be packed as float32 keys')
    by_id=ids.argsort(stable=True)
    order=values[by_id].argsort(stable=True)[:k]
    selected=by_id[order]
    return ids[selected],values[selected]


def reduce_rows(values,dids,counts,offsets,k,seed_ids=None,seed_values=None):
    output_ids=[];output_values=[]
    # Separate score/ID arrays preserve all 64 score bits and exact-tie ID order.
    for row,(off,count) in enumerate(zip(offsets.tolist(),counts.tolist())):
        ids=dids[off:off+count];v=values[off:off+count]
        if seed_ids is not None:
            ids=torch.cat((ids,seed_ids[row]));v=torch.cat((v,seed_values[row]))
        ids,v=stable_top(v,ids,k)
        if len(ids)!=k or not bool(torch.isfinite(v).all()):
            raise RuntimeError('Not enough finite FP64 candidates after filtering')
        output_ids.append(ids);output_values.append(v)
    return torch.stack(output_ids),torch.stack(output_values)


def expand(model,q,d,tree,self_ids,seeds,k):
    from leap.common.range_buffers import enumerate_candidates
    metric=model.config.distance
    n,s=seeds.shape
    rows=torch.arange(n,device=model.device)[:,None].expand_as(seeds).reshape(-1)
    initial_values=raw_pairs(metric,q,d,rows,seeds.reshape(-1),model.config.hausdorff_target=='pointset').reshape(n,s)
    radius=initial_values.topk(k,largest=False).values[:,-1].contiguous()
    slack=2e-5*(1+q.coordinates.abs().amax(1)+tree.database_scale)
    if metric=='dtw':
        slack=slack+2e-5*radius
    if metric=='hausdorff':
        from leap.hausdorff.range_search import enumerate_kernel
        from leap.common.early_refinement import bounded_hausdorff_distances
        args=(q.coordinates,d.coordinates,tree.members,tree.tree,radius,self_ids,slack)
        count,visits,dids,qr,off,maximum=enumerate_candidates(
            enumerate_kernel,args,n,model.device,tree.groups,tree.block,k,8_000_000,8_000_000)
        values=(raw_pairs(metric,q,d,qr,dids,True) if model.config.hausdorff_target=='pointset' else
                bounded_hausdorff_distances(q,d,qr,dids,(radius+slack).contiguous()))
        ids,distances=reduce_rows(values,dids,count,off,k)
    elif metric=='dfd':
        from leap.dfd.dfd_tree import native_enum, endpoints
        from leap.dfd.dfd_refinement import cached_pairs
        e=tree.dfd_geometry
        args=(q.coordinates,d.coordinates,endpoints(q),e['endpoints'],tree.members,*e['native'],radius,self_ids,slack)
        count,visits,dids,qr,off,maximum=enumerate_candidates(
            native_enum,args,n,model.device,tree.groups,tree.block,k,8_000_000,8_000_000)
        values,_=cached_pairs(q,d,qr,dids,seeds,initial_values,(radius+slack).contiguous())
        ids,distances=reduce_rows(values,dids,count,off,k)
    else:
        from leap.dtw.dtw_range import sidecars
        from leap.dtw.dtw_parallel import enumerate_parallel
        from leap.dtw import dtw_radial
        from leap.dtw import dtw_probe_intervals
        from leap.dtw import dtw_bounded
        qa,da=q.dtw_probe_summary,d.dtw_probe_summary
        radial=None if qa is None else (qa['radial'],da['radial'])
        count,visits,dids,qr,off,total,maximum,_=enumerate_parallel(q,d,tree,
            tree.dtw_geometry,sidecars(q),self_ids,seeds,radius,slack,
            radial,tree.dtw_probe_tree,True)
        keep=torch.ones_like(qr,dtype=torch.bool)
        if qa is not None:
            lower=dtw_radial.pairs(*radial,q.offsets,d.offsets,qr,dids)
            keep=lower<=radius[qr]+2e-4*(1+radius[qr])
            selected=keep.nonzero().flatten()
            keep[selected]=dtw_probe_intervals.keep_pairs(qa['ordered'],da['ordered'],q.offsets,d.offsets,
                qr[selected],dids[selected],radius[qr[selected]]+2e-4*(1+radius[qr[selected]]))
        values=torch.full((total,),torch.inf,device=model.device,dtype=torch.float64)
        values[keep]=dtw_bounded.pairs(q,d,qr[keep],dids[keep],(radius+slack)[qr[keep]],bucket=True)
        ids,distances=reduce_rows(values,dids,count,off,k,seeds,initial_values)
        count=count+s
    return dict(indices=ids,distances=distances,counts=count,radius=radius)


@torch.no_grad()
def search(model,queries,database=None,*,search_index=None,seed_k=None,top_k=20,query_batch=None,progress=None,
           self_ids=None):
    same=database is None or database is queries
    database=queries if database is None else database
    model._check_index(queries);model._check_index(database)
    if search_index is None:
        raise ValueError('FP64 full search requires its rebuilt search index')
    search_index.validate(database)
    if isinstance(top_k,bool) or not isinstance(top_k,int) or not 1<=top_k<=1000:
        raise ValueError('top_k must be an integer in [1,1000]')
    if top_k>64 and not model.config.full_large_k:
        raise ValueError('Full search above64 requires full_large_k=True')
    default={'hausdorff':50,'dfd':32,'dtw':20}[model.config.distance]
    seed_k=max(top_k,default) if seed_k is None else seed_k
    has_self=same or (self_ids is not None and bool((self_ids>=0).any()))
    seed_k=min(seed_k,len(database.coordinates)-int(has_self))
    if isinstance(seed_k,bool) or not isinstance(seed_k,int) or seed_k<top_k:
        raise ValueError('Initial count must cover top_k and fit the database')
    batch=32 if query_batch is None else query_batch
    if isinstance(batch,bool) or not isinstance(batch,int) or batch<1:
        raise ValueError('query_batch must be positive')
    if self_ids is not None and (self_ids.shape!=(len(queries.coordinates),) or self_ids.dtype!=torch.int64
            or self_ids.device!=queries.points.device or bool((self_ids < -1).any())
            or bool((self_ids>=len(database.coordinates)).any())):
        raise ValueError('Invalid explicit self-exclusion IDs')
    outputs=[]
    def process(lo,hi):
        q=subset(queries,lo,hi)
        excluded=(self_ids[lo:hi] if self_ids is not None else
                  torch.arange(lo,hi,device=model.device,dtype=torch.int64) if same else
                  torch.full((hi-lo,),-1,device=model.device,dtype=torch.int64))
        try:
            seeds=initial(model,q,database,search_index,excluded,seed_k)
            value=expand(model,q,database,search_index,excluded,seeds,top_k)
        except MemoryError:
            if hi-lo<=1:
                raise
            mid=(lo+hi)//2
            process(lo,mid);process(mid,hi)
            return
        outputs.append(value)
        if progress:
            progress(hi,len(queries.coordinates))
    for lo in range(0,len(queries.coordinates),batch):
        process(lo,min(lo+batch,len(queries.coordinates)))
    result={key:torch.cat([x[key] for x in outputs]) for key in outputs[0]}
    result['execution']={'precision':'fp64','routing_precision':'fp32','geometry_precision':'fp64',
                         'full_database':True,'empirical_slack_retained':True,'serial_fallback':False}
    result['full_search']=True
    return result


@torch.no_grad()
def refine_candidates(model,q,d,tree,count,k,candidate_search):
    ids=model.candidate_indices(q,d,count,search_index=tree,candidate_search=candidate_search)
    d=q if d is None else d
    if not 1<=k<=ids.shape[1]:
        raise ValueError('top_k must fit candidate count')
    rows=torch.arange(len(ids),device=model.device)[:,None].expand_as(ids).reshape(-1)
    values=raw_pairs(model.config.distance,q,d,rows,ids.reshape(-1),model.config.hausdorff_target=='pointset')
    counts=torch.full((len(ids),),ids.shape[1],device=model.device,dtype=torch.int64)
    off=torch.arange(len(ids),device=model.device)*ids.shape[1]
    selected,distances=reduce_rows(values,ids.reshape(-1),counts,off,k)
    return dict(indices=selected,distances=distances)


class PrecisionQuerySession:
    """FP64 raw-input request path without the FP32 CUDA-graph packet format."""
    def __init__(self,model,database,tree,top_k,initial_capacity):
        if model.config.query_profile!='single_query':
            raise ValueError('A query session requires query_profile=single_query')
        if (model.config.full_large_k or isinstance(top_k,bool) or not isinstance(top_k,int)
                or not 1<=top_k<=min(64,len(database.coordinates))):
            raise ValueError('Single request top_k must be in [1,min(64,N)]')
        model._check_index(database);tree.validate(database)
        self.model,self.database,self.tree,self.top_k=model,database,tree,top_k
        self._lock=threading.RLock()

    def search(self,raw_trajectory,*,mode='full'):
        if mode not in ('full','ranking'):
            raise ValueError('mode must be ranking or full')
        with self._lock:
            q=self.model.encode([raw_trajectory])
            if mode=='ranking':
                ids=self.model.candidate_indices(q,self.database,self.top_k,search_index=self.tree)
            else:
                ids=self.model.search_range(q,self.database,search_index=self.tree,top_k=self.top_k)['indices']
            return ids.cpu().numpy().copy()

    def warmup(self,raw_trajectories,*,modes=('ranking','full')):
        for path in raw_trajectories:
            for mode in modes:
                self.search(path,mode=mode)


@torch.no_grad()
def search_row(model,database,tree,row,top_n,mode,pool_size):
    q=subset(database,row,row+1)
    qids=torch.tensor([row],device=model.device,dtype=torch.int64)
    default={'hausdorff':50,'dfd':32,'dtw':20}[model.config.distance]
    count=top_n if mode=='initial' else pool_size if mode=='pool' else max(default,top_n)
    count=min(count,len(database.coordinates)-1)
    torch.cuda.synchronize(model.device)
    start=time.perf_counter()
    seeds=initial(model,q,database,tree,qids,count)
    torch.cuda.synchronize(model.device)
    coarse=time.perf_counter()-start
    detail={}
    if mode=='initial':
        ids=seeds;values=None;refined=0
    elif mode=='pool':
        values=raw_pairs(model.config.distance,q,database,torch.zeros_like(seeds.flatten()),
                         seeds.flatten(),model.config.hausdorff_target=='pointset')
        selected,v=stable_top(values,seeds.flatten(),top_n)
        ids=selected[None];values=v[None];refined=count
    else:
        detail=expand(model,q,database,tree,qids,seeds,top_n)
        ids=detail['indices'];values=detail['distances'];refined=int(detail['counts'][0])
    torch.cuda.synchronize(model.device)
    elapsed=time.perf_counter()-start
    output=ids[0].tolist()
    if len(output)!=top_n or len(set(output))!=top_n or row in output:
        raise RuntimeError('FP64 viewer count, uniqueness or self-exclusion failed')
    return dict(ids=output,scores=None if values is None else values[0].tolist(),
        coarse_seconds=coarse,search_seconds=elapsed,refinement_seconds=elapsed-coarse,
        initial_count=count,routing_pool=count,route='frozen FP32 ranking',survivor_count=refined,
        scope='database' if mode=='complete' else mode,radius=None if not detail else float(detail['radius'][0]),
        asset_id=model.asset_id,numerical_policy='FP64 decision geometry; FP32 initial ranking; empirical pruning slack')
