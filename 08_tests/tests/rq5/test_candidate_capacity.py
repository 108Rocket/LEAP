# Test group: RQ5.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search; RQ5: Initialization Budget and Result Count.
# Checks: Candidate capacity and top-k ordering across leaf sizes.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Leaf size must not limit requested Top-k or change its score/ID order."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from leap import RetrievalConfig
from leap import LEAP
from leap.common import full_large_k
from leap.common.interval_index import lex_keys
from tests.rq5.test_full_large_k import exhaustive, paths_for

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='L20 CUDA validation')


@pytest.fixture(scope='module', params=['near_far', 'polyline', 'dfd', 'dtw'])
def context(request):
    name = request.param
    metric = 'hausdorff' if name in ('near_far', 'polyline') else name
    config = RetrievalConfig(distance=metric, full_large_k=True,
        hausdorff_target='pointset' if name == 'near_far' else 'polyline',
        hausdorff_fields='near_far' if name == 'near_far' else 'default',
        dtw_seed='signal', dtw_routing='learned' if name == 'dtw' else 'fixed')
    model = LEAP.from_asset(config=config)
    paths = paths_for(model, 8201)
    database = model.encode(paths)
    query = model.encode(paths[:2])
    tree = model.build_search_index(database)
    yield name, model, paths, database, query, tree
    torch.cuda.synchronize()


@pytest.mark.parametrize('width', [1,20,63,64,65,125,128,129,256,320,640,1000,2048,6400,8192])
@torch.no_grad()
def test_candidate_tree_matches_full_score_scan(context, width):
    name, model, _, database, query, tree = context
    qids = torch.tensor([0, 1], device='cuda', dtype=torch.int64)
    if name == 'near_far':
        from leap.hausdorff.pointset_index import large_candidates
        got = tree.candidates(query, width, qids=qids)[0]
        expected = large_candidates(query, database, width, True, qids)
    elif name == 'polyline':
        from leap.common.interval_index import reference
        got = tree.candidates(query.coordinates, qids, width)[0]
        expected = reference(query.coordinates, database.coordinates, qids, width) & 0xffffffff
    elif name == 'dfd':
        from leap.dfd.dfd_tree import candidate_pool
        from leap.common.large_k import coarse_keys
        got = candidate_pool(query, database, tree, qids, width)[0]
        expected = coarse_keys(query, database, 'dfd', self_start=0).topk(width, largest=False).values & 0xffffffff
    else:
        from leap.dtw.ordered_dtw import candidates
        got = tree.dtw_routing_tree.candidates(query, width, qids)
        expected = candidates(query, database, width, qids, max_work=True)
    assert torch.equal(got, expected), (name, width)
    assert not (got == qids[:, None]).any()
    assert all(len(torch.unique(row)) == width for row in got)


@pytest.mark.parametrize('k', [1,20,50,64,65,100,1000])
@torch.no_grad()
def test_full_auto_matches_scan_and_original_distance(context, k):
    name, model, _, database, query, tree = context
    metric = model.config.distance
    seed = max(k, (5*k+1)//2) if metric == 'hausdorff' else ((8*k+4)//5 if metric == 'dfd' else k)
    pool = (16*k+4)//5 if metric == 'dfd' else ((32*k+4)//5 if metric == 'dtw' else None)
    model.config = replace(model.config, initial_k=seed, initial_pool=pool, initial_search='auto')
    got = model.search_range(query, database, search_index=tree, top_k=k, query_batch=2)
    model.config = replace(model.config, initial_search='full')
    scan = model.search_range(query, database, search_index=tree, top_k=k, query_batch=2)
    for key in ('indices', 'distances', 'radius', 'counts'):
        assert torch.equal(got[key], scan[key]), (name, k, key)
    assert torch.equal(lex_keys(got['distances'], got['indices']), exhaustive(model, query, database, k))
    model.config = replace(model.config, initial_search='auto')


@torch.no_grad()
def test_page_boundary_ties_and_self_exclusion(context):
    name, model, paths, _, _, _ = context
    database = model.encode([paths[0]] * 1050)
    query = model.encode([paths[0]] * 2)
    tree = model.build_search_index(database)
    qids = torch.tensor([0, 1049], device='cuda', dtype=torch.int64)
    for k in (1023, 1024, 1025, 1049):
        if name == 'near_far':
            got = tree.candidates(query, k, qids=qids)[0]
        elif name == 'polyline':
            got = tree.candidates(query.coordinates, qids, k)[0]
        elif name == 'dfd':
            from leap.dfd.dfd_tree import candidate_pool
            got = candidate_pool(query, database, tree, qids, k)[0]
        else:
            got = tree.dtw_routing_tree.candidates(query, k, qids)
        expected = torch.stack((torch.arange(1, k+1, device='cuda'),
                                torch.arange(k, device='cuda')))
        assert torch.equal(got, expected), (name, k)


@pytest.mark.parametrize('metric', ['hausdorff', 'dfd', 'dtw'])
def test_automatic_budget_does_not_impose_dense_batch_cap(metric):
    config = RetrievalConfig(distance=metric, parameter_mode='auto', dtw_seed='signal')
    for k in (50, 65, 100, 1000):
        plan = config.resolve(1_597_579, k, query_count=500)
        assert plan.path == 'tree_full'
        assert plan.config.query_batch == 500
    plan = replace(config, initial_search='full').resolve(1_597_579, 1000, query_count=500)
    assert plan.path == 'dense_full' and plan.config.query_batch == 16


@pytest.mark.parametrize('k', [65,100,1000])
@torch.no_grad()
def test_wide_dfd_ranking_preserves_final_scorer(k):
    config = RetrievalConfig(distance='dfd',full_large_k=True,initial_pool=(16*k+4)//5)
    model = LEAP.from_asset(config=config)
    paths = paths_for(model,8201)
    database = model.encode(paths)
    query = model.encode(paths[:2])
    tree = model.build_search_index(database)
    with patch('leap.dfd.dfd_tree.initial_candidates',side_effect=AssertionError('Unexpected small-k reranker')):
        actual = model.candidate_indices(query,database,k,search_index=tree,candidate_search='tree')
    expected = model.candidate_indices(query,database,k,search_index=tree,candidate_search='full')
    assert torch.equal(actual,expected)
