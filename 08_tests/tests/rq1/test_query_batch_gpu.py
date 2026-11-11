# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Batch-size changes preserve exhaustive-search results.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Batch changes must preserve complete native-precision results on CUDA."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from leap import LEAP
from leap import RetrievalConfig
from tests.rq5.test_full_large_k import paths_for, exhaustive


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA retrieval tests')


@pytest.mark.parametrize('metric',['hausdorff','pointset','dfd','dtw'])
@pytest.mark.parametrize('k',[1,20,1000])
@torch.no_grad()
def test_default_batch_matches_32_and_exhaustive(metric,k):
    distance = 'hausdorff' if metric == 'pointset' else metric
    large = k > 64 or metric == 'pointset'
    config = RetrievalConfig(distance=distance,full_large_k=large,
        hausdorff_target='pointset' if metric == 'pointset' else 'polyline',
        dtw_seed='signal',dtw_routing='learned' if distance == 'dtw' else 'fixed')
    model = LEAP.from_asset(config=config)
    paths = paths_for(model,1031)
    database = model.encode(paths)
    queries = model.encode(paths[10:45])
    tree = model.build_search_index(database)
    expected = exhaustive(model,queries,database,k)
    results = []
    for batch in (32,None):
        got = model.search_range(queries,database,search_index=tree,top_k=k,query_batch=batch)
        from leap.common.interval_index import lex_keys
        assert torch.equal(lex_keys(got['distances'],got['indices']),expected)
        results.append(got)
    for key in ('indices','distances','counts','radius'):
        assert torch.equal(results[0][key],results[1][key]),key
    model.config = replace(model.config,query_batch=512)
    configured = model.search_range(queries,database,search_index=tree,top_k=k)
    for key in ('indices','distances','counts','radius'):
        assert torch.equal(configured[key],results[1][key]),key

