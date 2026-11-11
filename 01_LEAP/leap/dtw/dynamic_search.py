"""Refine a near-first prefix, then resume its saved frontier at a tighter radius."""
import torch

from leap.common.interval_index import lex_keys
from leap.common.runtime import options as opt
from leap.dtw import dtw_parallel
from leap.dtw.bucketed_refinement import pairs
from leap.dtw.dynamic_frontier import leaf_resume
from leap.dtw.dtw_radial import pairs as radial_pairs
from leap.dtw.dtw_probe_intervals import keep_pairs as ordered_keep
from leap.dtw.dtw_ragged import topk as ragged_topk


@torch.no_grad()
def search(queries,index,tree,extra,qgeom,qids,initial,top_k=20,*,probe_fields=None,
           probe_tree=None,ordered_fields=None,check_interval=8,raw_initial=None):
    from leap.dtw.dtw_range import initial_distances
    n,s=initial.shape;device=initial.device
    raw=initial_distances(queries,index,initial,bucket=True,check=check_interval,
        maximum=extra['block'] if opt.dtw_length_cache else None) if raw_initial is None else raw_initial
    if raw.shape!=initial.shape or raw.device!=queries.points.device or raw.dtype!=queries.points.dtype:
        raise ValueError('Initial distances must match seed shape, geometry dtype and device')
    initial_radius=raw.topk(top_k,largest=False).values[:,-1].contiguous()
    radius=initial_radius
    # Keep the original (larger) floating-point guard across both phases.
    slack=(2e-5*(1+initial_radius+queries.coordinates.abs().amax(1)+tree.database_scale)).contiguous()
    best=lex_keys(raw,initial).topk(top_k,largest=False).values
    stack=torch.empty((n,32),device=device,dtype=torch.int32)
    lowers=torch.empty((n,32),device=device,dtype=raw.dtype)
    sp=torch.empty(n,device=device,dtype=torch.int32)
    counts=torch.zeros(n,device=device,dtype=torch.int32)
    visits_sum=torch.zeros((n,2),device=device,dtype=torch.int32)
    raw_calls=0
    for first in (True,False):
        def leaves(q,t,e,qg,rad,guard,pf=None,pt=None):
            g=t.groups;block=qg['block'] if opt.dtw_query_width else max(e['block'],qg['block'])
            leaf=torch.empty((n,g),device=device,dtype=torch.int32)
            count=torch.empty(n,device=device,dtype=torch.int32)
            visits=torch.zeros((n,2),device=device,dtype=torch.int32)
            leaf_resume[(n,)](q.points,q.offsets,q.coordinates,t.tree,qg['box'],e['treebox'],
                rad,guard,q.coordinates if pt is None else pf[0],t.tree if pt is None else pt,
                leaf,count,visits,stack,lowers,sp,g,block,pt is not None,first,512,
                num_warps=8,enable_fp_fusion=False)
            return leaf,count,visits,1
        count,visits,ids,rows,offsets,total,_,_=dtw_parallel.enumerate_parallel(
            queries,index,tree,extra,qgeom,qids,initial,radius,slack,
            probe_fields,probe_tree,True,leaf_enumerator=leaves)
        keep=torch.ones_like(rows,dtype=torch.bool)
        if probe_fields is not None:
            lb=radial_pairs(probe_fields[0],probe_fields[1],queries.offsets,index.offsets,rows,ids)
            keep=lb<=radius[rows]+2e-4*(1+radius[rows])
        if ordered_fields is not None:
            selected=keep.nonzero().flatten();qr=rows[selected]
            keep[selected]=ordered_keep(ordered_fields[0],ordered_fields[1],queries.offsets,index.offsets,
                qr,ids[selected],radius[qr]+2e-4*(1+radius[qr]))
        qr,dr=rows[keep],ids[keep]
        values=torch.full((total,),torch.inf,device=device,dtype=raw.dtype)
        values[keep]=pairs(queries,index,qr,dr,(radius+slack)[qr],bucket=True,
            check=check_interval,maximum=extra['block'] if opt.dtw_length_cache else None)
        old=(best>>32).int().contiguous().view(torch.float32)
        best=ragged_topk(best&0xffffffff,old,ids,values,offsets,count,top_k)
        radius=(best[:,-1]>>32).int().contiguous().view(torch.float32)
        counts+=count;visits_sum+=visits;raw_calls+=len(qr)
    return dict(indices=best&0xffffffff,distances=(best>>32).int().contiguous().view(torch.float32),
                counts=counts+s,visits=visits_sum,radius=initial_radius,
                raw_extra_pairs=raw_calls,enumeration='parallel-leaves')
