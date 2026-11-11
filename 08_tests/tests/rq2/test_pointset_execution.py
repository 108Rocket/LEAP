# Test group: RQ2.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search; RQ3: Contributions of C1--C3.
# Checks: Point-set near/far fields, Ranking/Full execution and session safety.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Public near/far HD adoption, exact outputs and session safety on CUDA."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from leap import RetrievalConfig
from leap import LEAP


def config(**kwargs):
    return RetrievalConfig(hausdorff_target='pointset',hausdorff_fields='near_far',**kwargs)


def test_profiles_and_guards():
    root=Path(__file__).parents[3]/'02_configs'/'hausdorff'
    for name in ('batch_query_pointset_hd','single_query_pointset_hd'):
        c=RetrievalConfig.from_json(root/(name+'.json'))
        assert c.hausdorff_fields=='near_far' and c.hausdorff_execution=='auto'
    assert RetrievalConfig().hausdorff_fields=='default'
    for kwargs in (dict(distance='dfd'),dict(distance='dtw'),dict(large_k=True)):
        with pytest.raises(ValueError):config(**kwargs)
    with pytest.raises(ValueError):RetrievalConfig(hausdorff_fields='near_far')
    with pytest.raises(ValueError):RetrievalConfig(hausdorff_execution='two_wave')


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='L20 CUDA required')


@pytest.fixture(params=('auto','two_wave'))
def example(request):
    if not torch.cuda.is_available():pytest.skip('L20 CUDA required')
    model=LEAP.from_asset(config=config(query_profile='single_query',hausdorff_execution=request.param))
    g=np.random.default_rng(1251)
    paths=[model.mean+model.std*g.normal(size=(3+i%15,2)).cumsum(0) for i in range(1031)]
    paths[1]=paths[0].copy();paths[2]=np.repeat(model.mean[None],8,axis=0)
    database=model.encode(paths);tree=model.build_search_index(database)
    return model,database,tree,paths


def exact(q,d,k,self_ids=None):
    from leap.common.raw_geometry_refinement import hausdorff_distances
    from leap.common.interval_index import lex_keys
    rows=torch.arange(len(q.coordinates),device=q.points.device)[:,None].expand(-1,len(d.coordinates))
    ids=torch.arange(len(d.coordinates),device=q.points.device)[None].expand_as(rows)
    values=hausdorff_distances(q,d,rows.flatten(),ids.flatten(),point=True).reshape_as(ids)
    if self_ids is not None:values[torch.arange(len(rows),device=values.device),self_ids]=torch.inf
    return lex_keys(values,ids).topk(k,largest=False).values


@cuda
@pytest.mark.parametrize('k',[1,20,1000])
@torch.no_grad()
def test_public_batch_and_single_k(example,k):
    from leap.common.interval_index import lex_keys
    model,d,tree,paths=example;q=model.encode(paths[10:13]);truth=exact(q,d,k)
    original_execution=model.config.hausdorff_execution
    for execution in ('auto','two_wave'):
        model.config=replace(model.config,hausdorff_execution=execution)
        for batch in (1,None):
            got=model.search_range(q,d,search_index=tree,top_k=k,query_batch=batch)
            assert torch.equal(lex_keys(got['distances'],got['indices']),truth)
    model.config=replace(model.config,hausdorff_execution=original_execution)
    session=model.query_session(d,search_index=tree,top_k=k,initial_capacity=2)
    session.warmup(paths[10:11],modes=('full',))
    for i,path in enumerate(paths[10:13]):
        out=session.search_result(path)
        key=(out['distances'].view('int32').astype('int64')<<32)|out['indices']
        np.testing.assert_array_equal(key[0],truth[i].cpu().numpy())


@cuda
@torch.no_grad()
def test_ranking_and_refinement_are_pointset(example):
    from leap.hausdorff import pointset_index as nf
    from leap.common.interval_index import lex_keys
    from leap.common.raw_geometry_refinement import hausdorff_distances
    model,d,tree,paths=example;q=model.encode(paths[3:6])
    for k in (1,20,50,64,1000):
        direct=nf.large_candidates(q,d,k)
        for backend in ('auto','full'):
            assert torch.equal(model.candidate_indices(q,d,k,search_index=tree,candidate_search=backend),direct)
    ids=model.candidate_indices(q,d,50,search_index=tree)
    rows=torch.arange(3,device=model.device)[:,None].expand_as(ids)
    values=hausdorff_distances(q,d,rows.flatten(),ids.flatten(),point=True).reshape_as(ids)
    truth=lex_keys(values,ids).topk(20,largest=False).values
    got=model.search(q,d,search_index=tree)
    assert torch.equal(lex_keys(got['distances'],got['indices']),truth)
    with pytest.raises(ValueError):model.search_large_k(q,d)


@cuda
@torch.no_grad()
def test_self_exclusion_and_small_database(example):
    from leap.common.interval_index import lex_keys
    model,_,_,paths=example
    for n in (2,20,51,70):
        d=model.encode(paths[:n]);tree=model.build_search_index(d);k=min(20,n-1)
        expected=exact(d,d,k,torch.arange(n,device=model.device))
        for execution in ('auto','two_wave'):
            model.config=replace(model.config,hausdorff_execution=execution)
            got=model.search_range(d,search_index=tree,top_k=k)
            assert torch.equal(lex_keys(got['distances'],got['indices']),expected)
            assert not torch.any(got['indices']==torch.arange(n,device=model.device)[:,None])
        session=model.query_session(d,search_index=tree,top_k=min(20,n))
        out=session.search_result(paths[0])
        q=model.encode([paths[0]])
        np.testing.assert_array_equal(out['indices'],(exact(q,d,min(20,n))&0xffffffff).cpu().numpy())


@cuda
@torch.no_grad()
def test_session_growth_owned_outputs_and_concurrency(example):
    from leap.hausdorff.pointset_plan import BufferedSingleFull
    model,d,tree,paths=example;session=model.query_session(d,search_index=tree,initial_capacity=2)
    session.warmup(paths[:1],modes=('full',));first=session.search_result(paths[0]);saved={k:v.copy() for k,v in first.items()}
    assert isinstance(session.plan,BufferedSingleFull)==(model.config.hausdorff_execution=='auto')
    longpath=np.tile(paths[0],(200,1))
    out=session.search_result(longpath)
    for k in first:np.testing.assert_array_equal(out[k],saved[k]);np.testing.assert_array_equal(first[k],saved[k])
    assert session.capacity>=len(longpath) and session.plan.graph is None
    session.warmup([longpath],modes=('full',));assert session.plan.graph is not None
    expected=[session.search(p) for p in paths[:4]]
    with ThreadPoolExecutor(max_workers=2) as pool:actual=list(pool.map(session.search,paths[:4]))
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


@cuda
@torch.no_grad()
def test_overflow_and_invalid_input_recovery(example):
    model,_,_,paths=example
    # Duplicate trajectories force more than four surviving leaves, with exact ties.
    path=paths[0];d=model.encode([path.copy() for _ in range(768)])
    tree=model.build_search_index(d);session=model.query_session(d,search_index=tree)
    expected=session.search_result(path)
    assert int(session.plan.wave.tail_counts[0])>1
    with torch.cuda.stream(session.stream):
        session.plan=session._make_plan(capacity=1)
        session.plan.warmup()
    got=session.search_result(path)
    assert session.plan.fallbacks==1
    for k in got:np.testing.assert_array_equal(got[k],expected[k])
    for raw in (np.zeros((1,2)),np.zeros((3,3)),np.full((3,2),np.nan),np.full((3,2),np.inf)):
        with pytest.raises(ValueError):session.search(raw)
    overflowing=model.mean+np.full((3,2),1e20)*model.std
    with pytest.raises(ValueError,match='overflowed'):session.search(overflowing)
    np.testing.assert_array_equal(session.search(path),expected['indices'])


@cuda
@pytest.mark.parametrize('change',['coordinates','points','offsets','members','tree','anchors','mean','config'])
@torch.no_grad()
def test_mutation_and_foreign_tree_rejected(example,change):
    model,d,tree,paths=example;session=model.query_session(d,search_index=tree)
    if change in ('coordinates','points','offsets'):getattr(d,change).add_(0)
    elif change in ('members','tree'):getattr(tree,change).add_(0)
    elif change=='anchors':model.anchors.add_(0)
    elif change=='mean':model.mean[0]+=1
    else:model.config=replace(model.config,hausdorff_fields='default',hausdorff_execution='auto',full_large_k=True)
    with pytest.raises(ValueError):session.search(paths[0])


@cuda
@torch.no_grad()
def test_field_identity_guards_and_batch_split(example):
    model,d,tree,paths=example
    other=model.encode(paths[:70]);foreign=model.build_search_index(other)
    with pytest.raises(ValueError):model.search_range(other,d,search_index=foreign)
    legacy=LEAP.from_asset(config=RetrievalConfig(hausdorff_target='pointset',full_large_k=True))
    with pytest.raises(ValueError):model.candidate_indices(legacy.encode(paths[:2]),d,search_index=tree)
    q=model.encode(paths[3:7]);expected=model.search_range(q,d,search_index=tree)
    import leap.hausdorff.pointset_index as nf
    original=nf.full
    def limited(query,*args,**kwargs):
        if len(query.coordinates)>1:raise MemoryError('Forced bounded split')
        return original(query,*args,**kwargs)
    with patch.object(nf,'full',limited):got=model.search_range(q,d,search_index=tree)
    for k in ('indices','distances','counts','radius'):
        assert torch.equal(got[k],expected[k])
    assert got['execution']['initializer']==expected['execution']['initializer']


@cuda
@torch.no_grad()
def test_explicit_scan_overrides_two_wave(example):
    import leap.hausdorff.pointset_search as execution
    model,d,tree,paths=example
    model.config=replace(model.config,hausdorff_execution='two_wave',initial_search='full')
    q=model.encode(paths[3:5])
    with patch.object(execution,'dynamic',side_effect=AssertionError('Unexpected tree initializer')):
        got=model.search_range(q,d,search_index=tree,top_k=20)
    assert got['execution']['initializer']=='fused-score-id-keys'
    from leap.common.interval_index import lex_keys
    assert torch.equal(lex_keys(got['distances'],got['indices']),exact(q,d,20))
    session=model.query_session(d,search_index=tree,top_k=20)
    session.warmup(paths[3:4],modes=('full','ranking'))
    assert session.plan is None and session.ranking_plan is None
    np.testing.assert_array_equal(session.search(paths[3]),got['indices'][:1].cpu().numpy())


@cuda
@pytest.mark.parametrize('k',[1,20,50,64,1000])
@torch.no_grad()
def test_ranking_graph_cold_warm_and_changed_query(example,k):
    model,d,tree,paths=example
    session=model.query_session(d,search_index=tree,top_k=k)
    expected=model.candidate_indices(model.encode(paths[:4]),d,k,search_index=tree).cpu().numpy()
    cold=session.search(paths[0],mode='ranking')
    np.testing.assert_array_equal(cold,expected[:1])
    if k<=64:assert session.ranking_plan.graph is None
    session.warmup(paths[:1],modes=('ranking',))
    if k<=64:assert session.ranking_plan.graph is not None
    else:assert session.ranking_plan is None
    for i,path in enumerate(paths[:4]):
        np.testing.assert_array_equal(session.search(path,mode='ranking'),expected[i:i+1])
    np.testing.assert_array_equal(cold,expected[:1])


@cuda
@torch.no_grad()
def test_ranking_graph_growth_overflow_and_buffered_fallback(example):
    model,d,tree,paths=example
    session=model.query_session(d,search_index=tree,initial_capacity=2)
    session.warmup(paths[:1],modes=('ranking','full'))
    assert session.ranking_plan.graph is not None
    longpath=np.tile(paths[0],(100,1))
    expected=model.candidate_indices(model.encode([longpath]),d,20,search_index=tree).cpu().numpy()
    np.testing.assert_array_equal(session.search(longpath,mode='ranking'),expected)
    assert session.ranking_plan.graph is None and session.plan is None
    session.warmup([longpath],modes=('ranking',))
    assert session.ranking_plan.graph is not None
    overflowing=model.mean+np.full((3,2),1e20)*model.std
    with pytest.raises(ValueError,match='overflowed'):session.search(overflowing,mode='ranking')
    np.testing.assert_array_equal(session.search(longpath,mode='ranking'),expected)
    model.config=replace(model.config,single_query_engine='buffered')
    buffered=model.query_session(d,search_index=tree)
    buffered.warmup(paths[:1],modes=('ranking',))
    assert buffered.ranking_plan is None
    got=buffered.search(paths[0],mode='ranking')
    np.testing.assert_array_equal(got,model.candidate_indices(model.encode(paths[:1]),d,20,search_index=tree).cpu().numpy())


@cuda
@torch.no_grad()
def test_viewer_pointset_and_self_exclusion(example):
    from leap.common.query_search import search_row, query_row
    model,d,tree,paths=example
    for k in (1,20,1000):
        got=search_row(model,d,tree,3,k)
        q=query_row(d,3,'hausdorff')
        expected=exact(q,d,k,torch.tensor([3],device=model.device))
        assert got['ids']==(expected[0]&0xffffffff).cpu().tolist()
    for mode in ('initial','pool'):
        got=search_row(model,d,tree,3,20,mode)
        assert len(got['ids'])==20 and 3 not in got['ids']
