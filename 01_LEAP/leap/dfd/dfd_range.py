"""DFD instantiation of GEF -> TBF -> RGR; eq:dfd-bound and eq:dfd-subset."""
import torch
from leap.dfd.dfd import dfd_pairs, dfd_pairs_bucketed
from leap.dfd.dfd_tree import endpoints, native_enum
from leap.dfd.dfd_refinement import cached_pairs
from leap.common.range_buffers import enumerate_candidates, top_keys


@torch.no_grad()
def search(queries,index,tree,qids,initial,top_k=20,*,raw_initial=None,dynamic_threshold=False):
    from leap.common.runtime import options as opt
    from leap.common.runtime.context import current
    plan=current()
    if dynamic_threshold and plan is None and len(initial)>1 and top_k==20 and initial.shape[1]<=64 and queries.points.dtype==torch.float32:
        from leap.dfd.dynamic_search import search as dynamic_search
        return dynamic_search(queries,index,tree,qids,initial,top_k,raw_initial=raw_initial)
    if raw_initial is None and opt.dfd_static and plan is not None and top_k==20 and len(initial)==1 and queries.points.dtype==torch.float32:
        from leap.dfd.dfd_single_static import run
        result=run(plan,queries,index,tree,qids,initial,opt.dfd_capacity)
        if result is not None:
            return result
    return reference_search(queries,index,tree,qids,initial,top_k,raw_initial=raw_initial)


@torch.no_grad()
def initial_distances(queries,index,initial):
    """Score seed pairs once, with the same recurrence and no path truncation."""
    from leap.common.runtime.context import current
    rows=torch.arange(len(initial),device=initial.device)
    qr=rows[:,None].expand_as(initial).reshape(-1)
    dr=initial.reshape(-1)
    score=dfd_pairs_bucketed if len(initial)>1 and current() is None else dfd_pairs
    return score(queries,index,qr,dr).reshape_as(initial)


@torch.no_grad()
def reference_search(queries,index,tree,qids,initial,top_k=20,*,raw_initial=None):
    n,s=initial.shape
    if raw_initial is None:
        raw_initial=initial_distances(queries,index,initial)
    elif (raw_initial.shape!=initial.shape or raw_initial.device!=queries.points.device
          or raw_initial.dtype!=queries.points.dtype):
        raise ValueError('Initial distances must match seed shape, geometry dtype and device')
    radius=raw_initial.topk(top_k,largest=False).values[:,-1].contiguous()
    slack=(2e-5*(1+queries.coordinates.abs().amax(1)+tree.database_scale)).contiguous()
    extra=tree.dfd_geometry
    args=(queries.coordinates,index.coordinates,endpoints(queries),extra['endpoints'],
          tree.members,*extra['native'],radius,qids,slack)
    counts,visits,dids,qraw,offsets,maximum=enumerate_candidates(
        native_enum,args,n,qids.device,tree.groups,tree.block,top_k,8_000_000,8_000_000)
    values,flags=cached_pairs(queries,index,qraw,dids,initial,raw_initial,(radius+slack).contiguous())
    top=top_keys(values,dids,qraw,offsets,n,maximum,top_k)
    return {'indices':top&0xffffffff,'distances':(top>>32).int().contiguous().view(torch.float32),
            'counts':counts,'radius':radius,'flags':flags,'visits':visits}
