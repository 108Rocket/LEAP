# Test group: RQ5.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search; RQ5: Initialization Budget and Result Count.
# Checks: Automatic budgets preserve the resolved manual Full-search outputs.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""L20 validation: auto dispatch must equal the resolved manual Full plan."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from leap import RetrievalConfig
from leap import LEAP
from tests.rq5.test_full_large_k import exhaustive

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def setup(metric, *, precision='fp32', profiles=(), single=False, n=1041):
    cfg=RetrievalConfig(distance='hausdorff' if metric in ('pointset','near_far') else metric,
        parameter_mode='auto',dtw_seed='signal',
        dtw_routing='learned' if metric=='dtw' else 'fixed',
        hausdorff_target='pointset' if metric in ('pointset','near_far') else 'polyline',
        hausdorff_fields='near_far' if metric=='near_far' else 'default',
        query_profile='single_query' if single else 'throughput',
        precision=precision,top_k_profiles=profiles)
    model=LEAP.from_asset(config=cfg)
    rng=np.random.default_rng(1653)
    paths=[model.mean+20*rng.normal(size=(2+i%8,2)).cumsum(0) for i in range(n)]
    paths[1]=paths[0].copy()
    db=model.encode(paths);q=model.encode([paths[2],paths[17]])
    return model,paths,db,q,model.build_search_index(db)


@pytest.mark.parametrize('metric',['hausdorff','pointset','near_far','dfd','dtw'])
@pytest.mark.parametrize('k',[1,20,64,65,1000])
@torch.no_grad()
def test_full_plan_matches_native_and_exhaustive(metric,k):
    model,paths,db,q,tree=setup(metric)
    cfg=model.config
    plan=cfg.resolve(len(db.coordinates),k,query_count=len(q.coordinates))
    from leap.common.runtime.configuration import configured_model
    manual=configured_model(model,plan)
    expected=manual.search_range(q,db,search_index=tree,top_k=k)
    got=model.search(q,db,search_index=tree,top_k=k)
    for key in ('indices','distances','counts','radius'):
        assert torch.equal(got[key],expected[key]),(metric,k,key)
    from leap.common.interval_index import lex_keys
    assert torch.equal(lex_keys(got['distances'],got['indices']),exhaustive(manual,q,db,k))
    assert model.config is cfg
    assert got['configuration']['full_database']


@pytest.mark.parametrize('metric',['hausdorff','near_far','dfd','dtw'])
@torch.no_grad()
def test_self_exclusion(metric):
    model,paths,db,q,tree=setup(metric,n=70)
    result=model.search_range(db,search_index=tree,top_k=65)
    assert not (result['indices']==torch.arange(70,device=model.device)[:,None]).any()
    assert result['configuration']['available']==69
    from leap.common.interval_index import lex_keys
    assert torch.equal(lex_keys(result['distances'],result['indices']),exhaustive(model,db,db,65,True))


@pytest.mark.parametrize('metric',['hausdorff','near_far','dfd','dtw'])
@pytest.mark.parametrize('k',[20,1000])
@torch.no_grad()
def test_single_session_resolves_once(metric,k):
    model,paths,db,q,tree=setup(metric,single=True)
    session=model.query_session(db,search_index=tree,top_k=k)
    expected=model.search_range(model.encode(paths[2:3]),db,search_index=tree,top_k=k)['indices'].cpu().numpy()
    session.warmup(paths[2:3])
    before=session.configuration.copy()
    actual=session.search(paths[2])
    np.testing.assert_array_equal(actual,expected)
    actual[:]=-1
    np.testing.assert_array_equal(session.search(paths[2]),expected)
    assert session.configuration==before
    model.config=replace(model.config,query_batch=3)
    with pytest.raises(ValueError,match='Model changed'):
        session.search(paths[2])


@pytest.mark.parametrize('metric',['near_far','dfd','dtw'])
@torch.no_grad()
def test_manual_profile_and_warm_graph_respect_override(metric):
    profile={'20':{'initial_k':40,'query_batch':2}}
    if metric!='near_far':profile['20']['initial_pool']=80
    model,paths,db,q,tree=setup(metric,profiles=profile,single=True,n=100)
    got=model.search_range(q,db,search_index=tree,top_k=20)
    assert got['configuration']['initial_k']==40
    session=model.query_session(db,search_index=tree)
    session.warmup(paths[2:3])
    np.testing.assert_array_equal(session.search(paths[2]),got['indices'][:1].cpu().numpy())
    if metric=='near_far':
        assert session.native.plan is None


@pytest.mark.parametrize('metric',['hausdorff','dfd','dtw'])
@pytest.mark.parametrize('k',[20,65])
@torch.no_grad()
def test_fp64_dispatch_preserved(metric,k):
    model,paths,db,q,tree=setup(metric,precision='fp64',n=100)
    from leap.common.runtime.configuration import configured_model
    manual=configured_model(model,model.config.resolve(100,k,query_count=2))
    expected=manual.search_range(q,db,search_index=tree,top_k=k)
    got=model.search_range(q,db,search_index=tree,top_k=k)
    assert got['distances'].dtype==torch.float64
    for key in ('indices','distances','counts','radius'):
        assert torch.equal(got[key],expected[key])


@pytest.mark.parametrize('metric,pool',[('dfd',320),('dtw',256)])
@torch.no_grad()
def test_wide_pool_override_selects_tree_initializer(metric,pool):
    model,paths,db,q,tree=setup(metric,profiles={'20':{'initial_pool':pool}},n=400)
    got=model.search_range(q,db,search_index=tree)
    assert got['configuration']['path']=='tree_full'
    assert got['execution']['initial_pool']==pool
    assert got['execution']['initializer']=='tree-score-id-keys'
    from leap.common.interval_index import lex_keys
    assert torch.equal(lex_keys(got['distances'],got['indices']),exhaustive(model,q,db,20))
