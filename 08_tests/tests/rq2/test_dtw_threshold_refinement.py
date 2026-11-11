# Test group: RQ2.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Threshold-aware DTW scheduling preserves exact refinement.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""L20 runtime checks for exact DTW scheduling changes; no training."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from leap import LEAP
from leap import RetrievalConfig
from leap.dtw.dtw import dtw_pairs
from leap.dtw.dtw_bounded import pairs
from leap.dtw.dtw_range import search

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA retrieval test')


def packed(paths):
    return SimpleNamespace(points=torch.tensor(np.concatenate(paths),device='cuda',dtype=torch.float32),
        offsets=torch.tensor(np.r_[0,np.cumsum([len(p) for p in paths])],device='cuda',dtype=torch.int64))


@pytest.mark.parametrize('bucket',[False,True])
@pytest.mark.parametrize('check',[1,8])
@torch.no_grad()
def test_bounded_frontiers(bucket,check):
    rng=np.random.default_rng(851)
    paths=[rng.normal(size=(n,2)).cumsum(0) for n in (1,2,3,4,7,8,9,17,31,32,33,64,65)]
    paths += [np.zeros((n,2)) for n in (1,2,3,8,17)]
    # A zero-cost diagonal skips the expensive odd antidiagonal.
    paths += [np.array([[0.,0.],[100.,0.]])]
    data=packed(paths)
    ids=torch.arange(len(paths),device='cuda')
    qi,di=torch.cartesian_prod(ids,ids).unbind(1)
    expected=dtw_pairs(data,data,qi,di)
    torch.testing.assert_close(pairs(data,data,qi,di,bucket=bucket),expected,rtol=0,atol=0)
    for limits in (expected,torch.nextafter(expected,torch.zeros_like(expected)),expected*.6,
                   torch.full_like(expected,torch.inf),torch.zeros_like(expected)):
        got,steps=pairs(data,data,qi,di,limits,bucket=bucket,check=check,record=True)
        within=expected<=limits
        torch.testing.assert_close(got[within],expected[within],rtol=0,atol=0)
        assert torch.all((got==expected)|torch.isinf(got))
        assert torch.all(got[~within]>limits[~within])
        maximum=data.offsets[qi+1]-data.offsets[qi]+data.offsets[di+1]-data.offsets[di]-1
        assert torch.all((steps>0)&(steps<=maximum))
    assert pairs(data,data,qi[:0],di[:0],expected[:0],bucket=bucket).numel()==0


@pytest.mark.parametrize('capacity',[0,1,2048])
@torch.no_grad()
def test_complete_and_overflow(capacity):
    rng=np.random.default_rng(64)
    model=LEAP.from_asset(config=RetrievalConfig('dtw','signal',dtw_routing='learned',dtw_execution='reference'))
    paths=[rng.normal(size=(2+i%35,2)).cumsum(0)*.0001+model.mean for i in range(150)]
    paths[2]=paths[1].copy()
    db=model.encode(paths)
    tree=model.build_search_index(db)
    for same in (False,True):
        q=db if same else model.encode(paths[:11])
        d=None if same else db
        args=dict(search_index=tree,query_batch=2000,reuse_initial=True,ragged_dtw=True)
        expected=model.search_range(q,d,**args)
        overflows=[]
        def optimized(*a,**kw):
            result=search(*a,**kw,bounded_dtw=True,bucket_dtw=True,one_pass_capacity=capacity)
            overflows.append(result['overflow_rows'])
            return result
        with patch('leap.dtw.dtw_range.search',side_effect=optimized):
            got=model.search_range(q,d,**args)
        for key in ('indices','distances','counts','radius'):
            torch.testing.assert_close(got[key],expected[key],rtol=0,atol=0)
        if capacity==1:
            assert sum(overflows)>0
