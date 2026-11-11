"""Validated near/far HD dispatch, with bounded generic Top-k fallbacks."""
import torch

from leap.hausdorff import pointset_index as nf
from leap.common.runtime.batching import resolve_query_batch
from leap.common.interval_index import lex_keys
from leap.hausdorff.pointset_device import DeviceWave
from leap.hausdorff.pointset_wave import keys_to_distances
from leap.common.raw_geometry_refinement import hausdorff_distances
from leap.common.candidate_policy import MAX_TREE_CANDIDATES, tree_initializer


def positive_k(value,name='top_k'):
    if isinstance(value,bool) or not isinstance(value,int) or value<1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def checked(model,queries,database,tree):
    same=database is None or database is queries
    database=queries if database is None else database
    model._check_index(queries);model._check_index(database)
    if tree is not None:
        if not isinstance(tree,nf.SearchTree) or tree.dim!=128:
            raise ValueError('near_far requires its own search index')
        tree.validate(database)
    return database,same


def batch_size(model,requested,wide,n):
    options=dict(default=128,database_size=n,max_score_pairs=26_000_000) if wide else {}
    return resolve_query_batch(model.config,requested,**options)


@torch.no_grad()
def candidates(model,queries,database=None,candidate_k=None,tree=None,execution='auto'):
    database,same=checked(model,queries,database,tree)
    if execution not in ('auto','tree','full'):
        raise ValueError('candidate_search must be auto, tree or full')
    k=min(positive_k(50 if candidate_k is None else candidate_k,'candidate_k'),len(database.coordinates)-int(same))
    if k<1:raise ValueError('At least one available candidate is required')
    use_tree=execution=='tree' or (execution=='auto' and tree_initializer(model.config,k,has_index=tree is not None))
    if use_tree and (tree is None or k>MAX_TREE_CANDIDATES):
        raise ValueError('Tree candidates require an index and k<=8192')
    batch=batch_size(model,None,not use_tree,len(database.coordinates));out=[]
    for lo in range(0,len(queries.coordinates),batch):
        q=nf.subset(queries,lo,lo+batch)
        ids=torch.arange(lo,lo+len(q.coordinates),device=model.device) if same else None
        if use_tree:part=tree.candidates(q,k,qids=ids)[0]
        else:part=nf.large_candidates(q,database,k,True,ids)
        out.append(part)
    return torch.cat(out)


@torch.no_grad()
def dynamic(query,database,tree,*,qids=None,compact=False):
    """The frozen50-seed, four-leaf first wave. Only Top20 uses this path."""
    n=len(query.coordinates);device=query.points.device
    if qids is None:qids=torch.full((n,),-1,device=device,dtype=torch.int64)
    eps=(2e-5*(1+query.coordinates.abs().amax(1)+tree.database_scale)).contiguous()
    seeds,_=tree.candidates(query,50,qids=qids,slack=eps)
    rows=torch.arange(n,device=device)[:,None].expand(-1,50).reshape(-1)
    values=hausdorff_distances(query,database,rows,seeds.flatten(),point=True).reshape(n,50)
    keys=lex_keys(values,seeds).topk(20,largest=False).values
    radius=keys_to_distances(keys)[:,-1].contiguous()
    if compact:
        from leap.hausdorff.compact_wave import CompactWave
        wave=CompactWave(n,device)
    else:
        wave=DeviceWave(n,device)
    _,r1=wave.first(query,database,tree,radius,eps,seeds,qids,keys)
    tail=wave.tail_filter(query,database,tree,r1,eps,seeds,qids)
    wave.tail_raw(query,database,tail);result=wave.finish(tail)
    return dict(indices=result&0xffffffff,distances=keys_to_distances(result),
                counts=wave.first_counts+tail['counts'],radius=r1)


@torch.no_grad()
def search(model,queries,database=None,tree=None,top_k=20,seed_k=None,query_batch=None,progress=None,*,self_ids=None):
    database,same=checked(model,queries,database,tree)
    if tree is None:raise ValueError('Build a near_far search index before Full search')
    k=positive_k(top_k);available=len(database.coordinates)-int(same or self_ids is not None)
    if not 1<=k<=min(1000,available):raise ValueError('Require top_k<=min(1000,available database rows)')
    seed=min(max(50,k) if seed_k is None else positive_k(seed_k,'seed_k'),available)
    if seed<k:raise ValueError('seed_k must cover top_k')
    use_tree=tree_initializer(model.config,seed)
    batch=batch_size(model,query_batch,not use_tree,len(database.coordinates));outputs=[]
    splits=0
    use_wave=use_tree and k==20 and seed==50 and model.config.hausdorff_execution=='two_wave'
    def process(lo,hi):
        nonlocal splits
        q=nf.subset(queries,lo,hi)
        ids=(self_ids[lo:hi] if self_ids is not None else
             torch.arange(lo,hi,device=model.device) if same else None)
        try:
            adaptive=model.config.dynamic_threshold and use_tree and k==20 and seed==50 and hi-lo>1
            if adaptive or use_wave:result=dynamic(q,database,tree,qids=ids,compact=adaptive)
            else:result=nf.full(q,database,tree,seed_k=seed,top_k=k,batch=hi-lo,qids=ids,tree_initial=use_tree)
        except MemoryError:
            if hi-lo==1:raise
            splits+=1
            mid=(lo+hi)//2;process(lo,mid);process(mid,hi);return
        outputs.append({key:result[key] for key in ('indices','distances','counts','radius')})
    for lo in range(0,len(queries.coordinates),batch):
        process(lo,min(lo+batch,len(queries.coordinates)))
        if progress is not None:progress(min(lo+batch,len(queries.coordinates)),len(queries.coordinates))
    result={key:torch.cat([x[key] for x in outputs]) for key in outputs[0]}
    result['execution']=dict(initializer='tree-score-id-keys' if use_tree else 'fused-score-id-keys',
        query_batch_limit=batch,batches=len(outputs),batch_splits=splits,top_k=k,seed_k=seed,full_database=True)
    return result


@torch.no_grad()
def refine(model,queries,database,candidate_k,top_k,tree,execution):
    k=positive_k(top_k)
    ids=candidates(model,queries,database,candidate_k,tree,execution)
    if k>ids.shape[1]:raise ValueError('top_k exceeds candidate_k')
    database=queries if database is None else database
    rows=torch.arange(len(ids),device=ids.device)[:,None].expand_as(ids)
    values=hausdorff_distances(queries,database,rows.flatten(),ids.flatten(),point=True).reshape_as(ids)
    keys=lex_keys(values,ids).topk(k,largest=False).values
    return dict(indices=keys&0xffffffff,distances=keys_to_distances(keys))
