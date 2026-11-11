# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Execution paths, length handling and overflow preserve search results.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

from dataclasses import replace
import numpy as np
import pytest
import torch

from leap import LEAP
from leap import RetrievalConfig
from leap.common.runtime import options as opt

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA retrieval test')


@pytest.fixture(autouse=True)
def reset_options():
    saved=(opt.dfd_static,opt.dfd_capacity,opt.dtw_query_width,opt.dtw_length_cache)
    opt.dfd_static=False;opt.dfd_capacity=2048
    opt.dtw_query_width=False;opt.dtw_length_cache=False
    yield
    opt.dfd_static,opt.dfd_capacity,opt.dtw_query_width,opt.dtw_length_cache=saved


@pytest.mark.parametrize('distance',['dfd','dtw'])
@torch.no_grad()
def test_real_lengths_and_exact_results(distance):
    cfg=RetrievalConfig(distance=distance,query_profile='single_query',dtw_seed='signal',dtw_routing='learned') if distance=='dtw' else RetrievalConfig(distance=distance,query_profile='single_query')
    model=LEAP.from_asset(config=cfg)
    rng=np.random.default_rng(1429)
    lengths=[2,3,31,32,33,63,64,65,127,128,129,257]*6
    paths=[model.mean+model.std*rng.normal(size=(n,2)).cumsum(0)*.03 for n in lengths]
    paths[1]=paths[0].copy()
    db=model.encode(paths);tree=model.build_search_index(db)
    selected=[paths[i] for i in (0,5,7,10,11)]
    q=model.encode(selected)
    reference=model.search_range(q,db,search_index=tree)
    opt.dfd_static=True;opt.dtw_query_width=True;opt.dtw_length_cache=True
    actual=model.search_range(q,db,search_index=tree)
    for key in ('indices','distances','counts','radius'):
        assert torch.equal(actual[key],reference[key]),key
    session=model.query_session(db,search_index=tree,initial_capacity=512)
    session.warmup(selected,modes=('full',))
    for i,path in enumerate(selected):
        np.testing.assert_array_equal(session.search(path),reference['indices'][i:i+1].cpu().numpy())
    new=selected[0]+model.std*np.array([.02,-.03])
    opt.dfd_static=False;opt.dtw_query_width=False;opt.dtw_length_cache=False
    expected=model.search_range(model.encode([new]),db,search_index=tree)['indices'].cpu().numpy()
    opt.dfd_static=True;opt.dtw_query_width=True;opt.dtw_length_cache=True
    np.testing.assert_array_equal(session.search(new),expected)
    if distance=='dfd':
        assert session.plan.stats['graph_replays']>0
        assert session.plan.dfd_fallbacks==0
    else:
        tree.members.add_(0)
        with pytest.raises(ValueError,match='length cache'):
            model.search_range(q,db,search_index=tree)


@torch.no_grad()
def test_dfd_overflow_growth_and_other_k():
    cfg=RetrievalConfig(distance='dfd',query_profile='single_query')
    model=LEAP.from_asset(config=cfg)
    path=model.mean+model.std*np.array([[0.,0.],[.1,.1],[.3,.4]])
    db=model.encode([path.copy() for _ in range(128)])
    tree=model.build_search_index(db)
    q=model.encode([path])
    expected=model.search_range(q,db,search_index=tree)['indices'].cpu().numpy()
    opt.dfd_static=True;opt.dfd_capacity=32
    session=model.query_session(db,search_index=tree,initial_capacity=2)
    session.warmup([path],modes=('full',))
    np.testing.assert_array_equal(session.search(path),expected)
    assert session.plan.dfd_fallbacks>0
    long=np.tile(path,(90,1))
    opt.dfd_static=False
    ref=model.search_range(model.encode([long]),db,search_index=tree)['indices'].cpu().numpy()
    opt.dfd_static=True
    np.testing.assert_array_equal(session.search(long),ref)
    with pytest.raises(ValueError):session.search(np.zeros((1,2)))
    np.testing.assert_array_equal(session.search(path),expected)
    for k in (1,):
        s=model.query_session(db,search_index=tree,top_k=k)
        ref=model.search_range(q,db,search_index=tree,top_k=k,seed_k=max(k,32))['indices']
        np.testing.assert_array_equal(s.search(path),ref.cpu().numpy())
