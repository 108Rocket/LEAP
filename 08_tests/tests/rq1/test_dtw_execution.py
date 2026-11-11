# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Optimized DTW execution preserves public Full-search outputs.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Public API integration of output-preserving DTW execution."""
from unittest.mock import patch
import numpy as np
import pytest
import torch
from leap import LEAP
from leap import RetrievalConfig


@pytest.mark.parametrize('same',[False,True])
@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA retrieval test')
def test_public_execution_equivalence_and_cached_tree(same):
    optimized=LEAP.from_asset(config=RetrievalConfig('dtw','signal',dtw_routing='learned'))
    reference=LEAP.from_asset(config=RetrievalConfig('dtw','signal',dtw_routing='learned',dtw_execution='reference'))
    rng=np.random.default_rng(405)
    paths=[optimized.mean+rng.normal(size=(3+i%34,2)).cumsum(0)*.0001 for i in range(153)]
    paths[2]=paths[1].copy()
    od,rd=optimized.encode(paths),reference.encode(paths)
    oq,rq=(od,rd) if same else (optimized.encode(paths[:9]),reference.encode(paths[:9]))
    ot,rt=optimized.build_search_index(od),reference.build_search_index(rd)
    assert ot.dtw_routing_tree is optimized._dtw_route_cache
    for k in (1,20,64):
        old=reference.candidate_indices(rq,None if same else rd,k)
        got=optimized.candidate_indices(oq,None if same else od,k,search_index=ot)
        torch.testing.assert_close(got,old,rtol=0,atol=0)
        full_scan=optimized.candidate_indices(oq,None if same else od,k,candidate_search='full')
        torch.testing.assert_close(full_scan,old,rtol=0,atol=0)
        old=reference.search_range(rq,None if same else rd,search_index=rt,top_k=k)
        with patch.object(optimized,'_dtw_candidates',wraps=optimized._dtw_candidates) as candidates:
            got=optimized.search_range(oq,None if same else od,search_index=ot,top_k=k,query_batch=5)
        assert candidates.call_count==1
        for key in ('indices','distances','counts','radius'):
            torch.testing.assert_close(got[key],old[key],rtol=0,atol=0)
    assert optimized._routing_tree(od) is ot.dtw_routing_tree


@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA retrieval test')
def test_wide_pool_tree_matches_full_without_truncation():
    model=LEAP.from_asset(config=RetrievalConfig('dtw','signal',dtw_pool=256,dtw_routing='learned'))
    rng=np.random.default_rng(13)
    db=model.encode([model.mean+rng.normal(size=(3+i%7,2))*.0001 for i in range(320)])
    q=model.encode([model.mean+rng.normal(size=(7,2))*.0001])
    expected=model.candidate_indices(q,db,20,candidate_search='full')
    actual=model.candidate_indices(q,db,20)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    tree=model.candidate_indices(q,db,20,candidate_search='tree')
    torch.testing.assert_close(tree,expected,rtol=0,atol=0)
    assert model.config.dtw_pool==256


def test_execution_config_validation():
    with pytest.raises(ValueError,match='dtw_execution'):
        RetrievalConfig(dtw_execution='fast-ish')
