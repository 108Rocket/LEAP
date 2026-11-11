# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Parallel DTW candidate checks preserve neighbor order.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Exact ordered-candidate regression tests for parallel DTW leaf checks."""
from unittest.mock import patch

import numpy as np
import pytest
import torch
import triton

from leap import LEAP
from leap import RetrievalConfig
from leap.dtw import dtw_range
from leap.dtw import dtw_parallel
from leap.dtw.dtw_bounded import pairs


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA retrieval tests')


@pytest.fixture(scope='module',params=[33,137])
def database(request):
    model=LEAP.from_asset(config=RetrievalConfig('dtw','signal',dtw_routing='learned'))
    rng=np.random.default_rng(929)
    lengths=(2,3,4,7,8,31,32,33,64,65,127,128,129,200,257)
    paths=[model.mean+20*rng.normal(size=(lengths[i%len(lengths)],2)).cumsum(0)
           for i in range(request.param)]
    paths[2]=paths[1].copy()
    paths[3]=np.repeat(model.mean[None],7,axis=0)
    db=model.encode(paths)
    return model,paths,db,model.build_search_index(db)


def serial_candidates(q,db,tree,qgeom,qids,initial,radius,slack,radial):
    n,s=initial.shape
    count=torch.empty(n,device=qids.device,dtype=torch.int32)
    visits=torch.empty((n,2),device=qids.device,dtype=torch.int32)
    extra=tree.dtw_geometry
    args=(db.points,db.offsets,q.points,q.offsets,q.coordinates,tree.members,tree.tree,
          extra['box'],qgeom['box'],extra['treebox'],radius,slack,qids,initial,
          q.dtw_probe_summary['radial'] if radial else q.coordinates,
          tree.dtw_probe_tree if radial else tree.tree)
    dims=(tree.groups,tree.block,max(extra['block'],qgeom['block']),s)
    options=dict(INITIAL_BLOCK=max(64,triton.next_power_of_2(s)),num_warps=8,enable_fp_fusion=False)
    dtw_range.enumerate_sum[(n,)](*args,count,count,count,count,visits,*dims,False,radial,**options)
    off=count.long().cumsum(0)-count
    total=int(count.sum())
    dids=torch.empty(total,device=qids.device,dtype=torch.int64);rows=torch.empty_like(dids)
    written=torch.empty_like(count)
    dtw_range.enumerate_sum[(n,)](*args,off,dids,rows,written,visits,*dims,True,radial,**options)
    assert torch.equal(written,count)
    return count,visits,dids,rows,off,total,int(count.max()),0


@pytest.mark.parametrize('same',[False,True])
@pytest.mark.parametrize('radial',[False,True])
@torch.no_grad()
def test_ordered_candidates_and_boundary_cases(database,same,radial):
    model,paths,db,tree=database
    q=db if same else model.encode(paths[:17])
    n=len(q.coordinates)
    qids=torch.arange(n,device='cuda') if same else torch.full((n,),-1,device='cuda',dtype=torch.int64)
    initial=model._dtw_candidates(q,db,20,qids,routing_tree=tree.dtw_routing_tree)
    qr=torch.arange(n,device='cuda')[:,None].expand_as(initial).reshape(-1)
    normal=pairs(q,db,qr,initial.reshape(-1),bucket=True).reshape_as(initial).amax(1)
    qgeom=dtw_range.sidecars(q)
    for radius in (normal,normal*.1,torch.nextafter(normal,torch.zeros_like(normal)),
                   torch.zeros_like(normal),torch.full_like(normal,1e8),torch.full_like(normal,-1e8)):
        slack=(2e-5*(1+radius+q.coordinates.abs().amax(1)+tree.database_scale)).contiguous()
        expected=serial_candidates(q,db,tree,qgeom,qids,initial,radius,slack,radial)
        got=dtw_parallel.enumerate_parallel(q,db,tree,tree.dtw_geometry,qgeom,qids,initial,radius,slack,
            (q.dtw_probe_summary['radial'],db.dtw_probe_summary['radial']) if radial else None,
            tree.dtw_probe_tree if radial else None)
        for a,b in zip(got,expected):
            assert torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b
        if torch.all(radius<0):
            assert got[5]==0


@pytest.mark.parametrize('limit',['MAX_LEAF_CELLS','MAX_MEMBER_CELLS'])
@torch.no_grad()
def test_workspace_limits_split_without_recomputing_initial(database,limit):
    model,_,db,tree=database
    expected=model.search_range(db,search_index=tree,query_batch=2000)
    budget=tree.groups*8*(tree.block if limit=='MAX_MEMBER_CELLS' else 1)
    with patch.object(dtw_parallel,limit,budget):
        with patch.object(model,'_dtw_candidates',wraps=model._dtw_candidates) as initial:
            with patch.object(dtw_parallel,'enumerate_parallel',wraps=dtw_parallel.enumerate_parallel) as enum:
                got=model.search_range(db,search_index=tree,query_batch=2000)
    assert initial.call_count==1
    assert enum.call_count>1
    for name in expected:
        torch.testing.assert_close(got[name],expected[name],rtol=0,atol=0)


@pytest.mark.parametrize('leaf_capacity',[128])
@torch.no_grad()
def test_other_leaf_widths_keep_serial_path(database,leaf_capacity):
    model,_,db,_=database
    tree=model.build_search_index(db,leaf_capacity=leaf_capacity)
    expected=model.search_range(db,search_index=tree)
    with patch.object(dtw_parallel,'enumerate_parallel',side_effect=AssertionError('Unvalidated leaf width')):
        got=model.search_range(db,search_index=tree)
    for name in expected:
        torch.testing.assert_close(got[name],expected[name],rtol=0,atol=0)


@torch.no_grad()
def test_reference_mode_does_not_call_parallel(database):
    _,paths,_,_=database
    model=LEAP.from_asset(config=RetrievalConfig('dtw','signal',dtw_routing='learned',dtw_execution='reference'))
    db=model.encode(paths);tree=model.build_search_index(db)
    with patch.object(dtw_parallel,'enumerate_parallel',side_effect=AssertionError('Reference must stay serial')):
        result=model.search_range(db,search_index=tree)
    assert result['indices'].shape==(len(paths),20)
