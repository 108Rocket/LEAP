# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: DTW execution optimizations preserve outputs; no timing result is produced.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Output-preserving prototype checks. No model training."""
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pytest
import torch

from leap import LEAP
from leap import RetrievalConfig
from leap import DualFieldRetriever
from leap.dtw import ordered_dtw
from leap.dtw.dtw_routing_tree import RoutingTree
from leap.dtw.dtw_ragged import topk
from leap.common.interval_index import lex_keys

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA retrieval test')


@pytest.mark.parametrize('size',[31,129,521])
@pytest.mark.parametrize('same',[False,True])
@torch.no_grad()
def test_routing_tree_exact(size,same):
    rng=np.random.default_rng(331)
    model=DualFieldRetriever(rng.normal(size=(64,2)),[0.,0.],[1.,1.],config=RetrievalConfig('dtw','ordered'))
    paths=[rng.normal(size=(2+i%40,2)).cumsum(0) for i in range(size)]
    paths[1]=paths[0].copy()
    paths[2]=np.zeros((7,2))
    db=model.encode(paths)
    q=replace(db,coordinates=db.coordinates[:25],offsets=db.offsets[:26],ordered_summary=db.ordered_summary[:25]) if same else model.encode(paths[:25])
    qids=torch.arange(25,device='cuda',dtype=torch.int64) if same else None
    tree=RoutingTree(db)
    for k in (1,20,min(128,size-int(same))):
        got=tree.candidates(q,k,qids)
        expected=ordered_dtw.candidates(q,db,k,qids,max_work=True)
        torch.testing.assert_close(got,expected,rtol=0,atol=0)
    db.ordered_summary.add_(1.)
    with pytest.raises(ValueError,match='changed'):
        tree.candidates(q,20,qids)


@pytest.mark.parametrize('k',[1,5,20,64])
@torch.no_grad()
def test_ragged_sort_ties_and_empty(k):
    rng=np.random.default_rng(44)
    counts=torch.tensor([0,1,127,128,129,501,5001],device='cuda',dtype=torch.int32)
    offsets=counts.long().cumsum(0)-counts
    total=int(counts.sum())
    values=torch.as_tensor(rng.integers(0,6,total),device='cuda',dtype=torch.float32)
    dids=torch.arange(total,device='cuda',dtype=torch.int64)+1000
    initial=torch.arange(7*64,device='cuda',dtype=torch.int64).reshape(7,64)
    initial_d=torch.as_tensor(rng.integers(0,6,(7,64)),device='cuda',dtype=torch.float32)
    got=topk(initial,initial_d,dids,values,offsets,counts,k)
    for i in range(7):
        lo,hi=int(offsets[i]),int(offsets[i]+counts[i])
        reference=torch.cat([lex_keys(initial_d[i],initial[i]),lex_keys(values[lo:hi],dids[lo:hi])]).topk(k,largest=False).values
        torch.testing.assert_close(got[i],reference,rtol=0,atol=0)


@torch.no_grad()
def test_cached_ragged_search_and_split():
    rng=np.random.default_rng(73)
    model=LEAP.from_asset(config=RetrievalConfig('dtw','signal',dtw_routing='learned'))
    raw=[rng.normal(size=(2+i%27,2)).cumsum(0)*.0001+model.mean for i in range(155)]
    db=model.encode(raw)
    tree=model.build_search_index(db)
    for same in (True,False):
        q=db if same else model.encode(raw[:15])
        d=None if same else db
        expected=model.search_range(q,d,search_index=tree,query_batch=16)
        got=model.search_range(q,d,search_index=tree,query_batch=2000,reuse_initial=True,ragged_dtw=True)
        for name in ('indices','distances','counts','radius'):
            torch.testing.assert_close(got[name],expected[name],rtol=0,atol=0)
        from leap.dtw.dtw_range import search
        def bounded(*a,**kw):
            if len(a[0].coordinates)>8:
                raise MemoryError('Force prototype fallback')
            return search(*a,**kw)
        original=model._dtw_candidates
        with patch.object(model,'_dtw_candidates',wraps=original) as candidates:
            with patch('leap.dtw.dtw_range.search',side_effect=bounded):
                result=model.search_range(q,d,search_index=tree,query_batch=2000,reuse_initial=True,ragged_dtw=True)
        assert candidates.call_count==1
        torch.testing.assert_close(result['indices'],expected['indices'],rtol=0,atol=0)
