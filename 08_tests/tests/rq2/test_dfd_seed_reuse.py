# Test group: RQ2.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Initial-candidate distance reuse and complete refinement under retries.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Exact initial scoring and cache lifetime under bounded recursive retries."""
from unittest.mock import patch
import numpy as np
import pytest
import torch
from leap import LEAP
from leap import RetrievalConfig
from leap.dfd import dfd_range
from leap.dfd.dfd import dfd_pairs, dfd_pairs_bucketed

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='L20 CUDA validation')


def example():
    rng=np.random.default_rng(51011)
    model=LEAP(rng.normal(size=(64,2)),[0.,0.],[1.,1.],config=RetrievalConfig('dfd'))
    lengths=[2,3,4,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129]
    paths=[rng.normal(size=(n,2)).cumsum(0) for n in lengths]
    paths.extend([paths[i%len(lengths)].copy() for i in range(70)])
    db=model.encode(paths)
    q=model.encode(paths[:9]+[paths[11][::-1].copy()])
    return model,q,db,model.build_search_index(db)


def test_bucket_boundaries_bitwise():
    _,q,d,_=example()
    ids=torch.arange(len(d.coordinates),device='cuda')[None].expand(len(q.coordinates),-1)
    rows=torch.arange(len(q.coordinates),device='cuda')[:,None].expand_as(ids)
    rows,ids=rows.reshape(-1),ids.reshape(-1)
    assert torch.equal(dfd_pairs(q,d,rows,ids),dfd_pairs_bucketed(q,d,rows,ids))
    assert not dfd_pairs_bucketed(q,d,rows[:0],ids[:0]).numel()


@pytest.mark.parametrize('same',[False,True])
@pytest.mark.parametrize('k',[1,20,64])
def test_retries_reuse_every_seed_once(same,k):
    model,q,d,tree=example()
    q=d if same else q
    kwargs=dict(search_index=tree,top_k=k,initial_k=max(32,k))
    expected=model.search_range(q,None if same else d,**kwargs)
    calls=[]
    enumerate_original=dfd_range.enumerate_candidates
    score_original=dfd_range.initial_distances
    def score(query,database,initial):
        values=score_original(query,database,initial)
        calls.append(initial.numel())
        return values
    def limited(kernel,args,n,*rest):
        if n>1:
            raise MemoryError('Force split after initial DP')
        return enumerate_original(kernel,args,n,*rest)
    with patch.object(dfd_range,'initial_distances',side_effect=score),patch.object(
            dfd_range,'enumerate_candidates',side_effect=limited):
        actual=model.search_range(q,None if same else d,**kwargs)
    assert calls==[len(q.coordinates)*max(32,k)]
    for key in ('indices','distances','counts','radius'):
        assert torch.equal(actual[key],expected[key]),key
    # A later public call must score its own inputs, not reuse stale request data.
    with patch.object(dfd_range,'initial_distances',side_effect=score):
        model.search_range(q,None if same else d,**kwargs)
    assert len(calls)==2


def test_single_request_keeps_original_scorer():
    model,q,d,tree=example()
    one=model.encode([np.array([[0.,0.],[1.,1.]])])
    with patch.object(dfd_range,'dfd_pairs',wraps=dfd_pairs) as raw,patch.object(
            dfd_range,'dfd_pairs_bucketed',wraps=dfd_pairs_bucketed) as bucket:
        model.search_range(one,d,search_index=tree)
    assert raw.call_count==1 and bucket.call_count==0


def test_reject_mismatched_precomputed_scores():
    model,q,d,tree=example()
    ids=torch.full((len(q.coordinates),),-1,device='cuda',dtype=torch.int64)
    initial=model.candidate_indices(q,d,search_index=tree)
    with pytest.raises(ValueError,match='Initial distances'):
        dfd_range.search(q,d,tree,ids,initial,raw_initial=torch.zeros(1,device='cuda'))
