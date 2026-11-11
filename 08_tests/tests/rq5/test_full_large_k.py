# Test group: RQ5.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search; RQ5: Initialization Budget and Result Count.
# Checks: Large-k Full search agrees with exhaustive original-distance results.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Large-k must use the declared path and preserve original-distance results."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from leap import LEAP
from leap import RetrievalConfig
from leap.common import full_large_k
from leap.dtw import dtw_parallel
from leap.dtw import dtw_range
from leap.common.interval_index import lex_keys, reference


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA retrieval tests')


def make_model(metric):
    distance='hausdorff' if metric=='pointset' else metric
    cfg=RetrievalConfig(distance=distance,full_large_k=True,dtw_seed='signal',
        dtw_routing='learned' if distance=='dtw' else 'fixed',
        hausdorff_target='pointset' if metric=='pointset' else 'polyline')
    return LEAP.from_asset(config=cfg)


def paths_for(model,n):
    rng=np.random.default_rng(925)
    paths=[model.mean+20*rng.normal(size=(2+i%12,2)).cumsum(0) for i in range(n)]
    paths[1]=paths[0].copy()
    paths[2]=np.repeat(model.mean[None],9,axis=0)
    return paths


def exhaustive(model,q,d,k,same=False):
    nr,nd=len(q.coordinates),len(d.coordinates)
    rows=torch.arange(nr,device='cuda')[:,None].expand(nr,nd).reshape(-1)
    ids=torch.arange(nd,device='cuda')[None].expand(nr,nd).reshape(-1)
    if model.config.distance=='hausdorff':
        from leap.common.raw_geometry_refinement import hausdorff_distances
        values=hausdorff_distances(q,d,rows,ids,point=model.config.hausdorff_target=='pointset')
    elif model.config.distance=='dfd':
        from leap.dfd.dfd import dfd_pairs_bucketed
        values=dfd_pairs_bucketed(q,d,rows,ids)
    else:
        from leap.dtw.dtw_bounded import pairs
        values=pairs(q,d,rows,ids,bucket=True)
    values=values.reshape(nr,nd)
    if same:
        values.fill_diagonal_(torch.inf)
    return lex_keys(values,torch.arange(nd,device='cuda')[None]).topk(k,largest=False).values


@pytest.mark.parametrize('metric',['hausdorff','pointset','dfd','dtw'])
@pytest.mark.parametrize('same',[False,True])
@torch.no_grad()
def test_public_full_matches_exhaustive(metric,same):
    model=make_model(metric)
    paths=paths_for(model,70 if same else 1031)
    d=model.encode(paths);q=d if same else model.encode([paths[i] for i in (0,2,17)])
    tree=model.build_search_index(d)
    k=65 if same else 1000
    expected=exhaustive(model,q,d,k,same)
    progress=[]
    got=model.search_range(q,d,search_index=tree,top_k=k,query_batch=16,
                           progress=lambda done,total:progress.append((done,total)))
    actual=lex_keys(got['distances'],got['indices'])
    assert torch.equal(actual,expected)
    assert got['full_search'] and got['execution']['full_database']
    assert not got['execution']['serial_fallback']
    assert progress[-1]==(len(q.coordinates),len(q.coordinates))
    assert torch.isfinite(got['distances']).all()


@torch.no_grad()
def test_fused_initial_ids_match_previous():
    for metric in ('hausdorff','pointset','dfd','dtw'):
        model=make_model(metric);d=model.encode(paths_for(model,1041));q=full_large_k.subset(d,13,16)
        ids=torch.arange(13,16,device='cuda')
        actual=full_large_k.initial_candidates(model,q,d,ids,1000,13)
        if metric=='hausdorff':
            expected=reference(q.coordinates,d.coordinates,ids,1000)&0xffffffff
        elif metric=='dtw':
            from leap.dtw.dtw_probes import candidates
            expected=candidates(q,d,1000,ids,pool=6400,bucket=True)
        else:
            from leap.common.large_k import coarse_keys
            distance='hausdorff' if metric=='pointset' else metric
            keys=coarse_keys(q,d,distance,execution='reference',self_start=13)
            if metric=='pointset':
                expected=keys.topk(1000,largest=False).values&0xffffffff
            else:
                from leap.dfd.dfd import dfd_pair_bounds
                from leap.dfd.dfd_geometry import witness_bound, ordered_bound
                pool=keys.topk(1040,largest=False).values&0xffffffff
                rows=torch.arange(3,device='cuda')[:,None].expand_as(pool).reshape(-1)
                flat=pool.reshape(-1)
                score=torch.maximum(dfd_pair_bounds(q,d,rows,flat),torch.maximum(
                    witness_bound(q,d,rows,flat,8),ordered_bound(q,d,rows,flat,8)))
                expected=lex_keys(score.reshape_as(pool),pool).topk(1000,largest=False).values&0xffffffff
        assert torch.equal(actual,expected),metric


@torch.no_grad()
def test_large_seed_parallel_candidate_parity():
    from tests.rq1.test_dtw_parallel import serial_candidates
    model=make_model('dtw');paths=paths_for(model,1041)
    d=model.encode(paths);q=model.encode(paths[:3]);tree=model.build_search_index(d)
    qids=torch.full((3,),-1,device='cuda',dtype=torch.int64)
    geom=dtw_range.sidecars(q)
    for s in (65,1000):
        initial=full_large_k.initial_candidates(model,q,d,qids,s)
        for value in (0.,1e8,-1e8):
            radius=torch.full((3,),value,device='cuda')
            slack=(2e-5*(1+radius+q.coordinates.abs().amax(1)+tree.database_scale)).contiguous()
            expected=serial_candidates(q,d,tree,geom,qids,initial,radius,slack,True)
            actual=dtw_parallel.enumerate_parallel(q,d,tree,tree.dtw_geometry,geom,qids,
                initial,radius,slack,(q.dtw_probe_summary['radial'],d.dtw_probe_summary['radial']),tree.dtw_probe_tree)
            for a,b in zip(actual,expected):
                assert torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b


@torch.no_grad()
def test_split_does_not_change_results_or_fallback():
    model=make_model('dtw');d=model.encode(paths_for(model,75));q=model.encode(paths_for(model,5))
    tree=model.build_search_index(d)
    expected=model.search_range(q,d,search_index=tree,top_k=65)
    with patch.object(dtw_parallel,'MAX_LEAF_CELLS',tree.groups*2):
        got=model.search_range(q,d,search_index=tree,top_k=65)
    assert got['execution']['batch_splits']>0
    assert not got['execution']['serial_fallback']
    for key in ('indices','distances','counts','radius'):
        assert torch.equal(got[key],expected[key])
    tree128=model.build_search_index(d,leaf_capacity=128)
    with pytest.raises(ValueError,match='no serial fallback'):
        model.search_range(q,d,search_index=tree128,top_k=65)
    model.config=replace(model.config,full_large_k=False)
    with pytest.raises(ValueError,match='full_large_k=True'):
        model.search_range(q,d,search_index=tree,top_k=65)


def test_config_and_parameter_guards():
    with pytest.raises(ValueError,match='OR large_k'):
        RetrievalConfig(full_large_k=True,large_k=True)
    with pytest.raises(ValueError,match='signal ranking'):
        RetrievalConfig(distance='dtw',full_large_k=True)
    with pytest.raises(ValueError,match='boolean'):
        RetrievalConfig(full_large_k='yes')
