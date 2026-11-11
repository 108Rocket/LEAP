# Test group: RQ3.
# Paper RQ: RQ3: Contributions of C1--C3.
# Checks: Weighted DTW summary recurrence, ordered ranking and bounded cascade.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Weighted summary equation and the bounded public cascade, not teacher fitting."""
from dataclasses import replace
from unittest.mock import patch
import json
import numpy as np
import pytest
import torch
import triton
from leap import DualFieldRetriever
from leap import RetrievalConfig


def reference(a,b,n,m):
    dp=np.full((9,9),np.inf);dp[0,0]=0.
    for i in range(8):
        for j in range(8):
            c=np.linalg.norm(a[i]-b[j])
            dp[i+1,j+1]=min(dp[i,j+1]+c*n/8,dp[i+1,j]+c*m/8,dp[i,j]+c*max(n,m)/8)
    return dp[-1,-1]


def test_config_pool(tmp_path):
    assert RetrievalConfig('dtw').dtw_pool==128
    for pool in (0,-1,True,1.5,'128',None):
        with pytest.raises(ValueError):RetrievalConfig('dtw',dtw_pool=pool)
    p=tmp_path/'weighted.json'
    p.write_text(json.dumps({'distance':'dtw','dtw_seed':'weighted','dtw_pool':64}))
    assert RetrievalConfig.from_json(p)==RetrievalConfig('dtw','weighted',64)


@pytest.fixture
def model():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    return DualFieldRetriever(np.random.default_rng(32).normal(size=(64,2)),[0,0],[1,1],config=RetrievalConfig('dtw'))


def test_recurrence_partial_tiles_short_paths(model):
    from leap.dtw.weighted_dtw import alignment_kernel
    rng=np.random.default_rng(51)
    lengths=np.resize([2,3,7,8,9,20,46,56,107,162,200,257,512],19)
    paths=[rng.normal(size=(n,2)).cumsum(0) for n in lengths]
    paths[1]=paths[0].copy()
    index=model.encode(paths)
    ids=torch.arange(19,device='cuda')[None].expand(19,-1).contiguous()
    scores=torch.empty((19,19),device='cuda')
    alignment_kernel[(19,2)](index.ordered_summary,index.ordered_summary,index.offsets,index.offsets,
                            ids,scores,19,0,num_warps=4,enable_fp_fusion=False)
    means=index.ordered_summary.cpu().numpy().reshape(-1,8,2).astype(np.float64)
    n=np.diff(index.offsets.cpu().numpy())
    expected=np.array([[reference(a,b,n[i],n[j]) for j,b in enumerate(means)] for i,a in enumerate(means)])
    np.testing.assert_allclose(scores.cpu().numpy(),expected,rtol=5e-6,atol=4e-4)
    np.testing.assert_allclose(scores.cpu().numpy(),scores.T.cpu().numpy(),rtol=5e-6,atol=4e-4)
    assert scores[0,1]==0


def test_pool_sizes_self_ties_and_large_k(model):
    index=model.encode([np.zeros((2+i%9,2)) for i in range(151)])
    for pool,k in ((128,7),(128,140),(3,7),(999,150)):
        model.config=RetrievalConfig('dtw',dtw_pool=pool)
        got=model.candidate_indices(index,candidate_k=k).cpu().numpy()
        for row in range(151):
            np.testing.assert_array_equal(got[row],np.delete(np.arange(151),row)[:k])
    query=model.encode([np.zeros((2,2))])
    np.testing.assert_array_equal(model.candidate_indices(query,index,7).cpu().numpy()[0],np.arange(7))
    tiny=model.encode([np.zeros((2,2)),np.ones((3,2))])
    np.testing.assert_array_equal(model.candidate_indices(tiny, candidate_k=50).cpu().numpy(),[[1],[0]])


def test_weighted_stale_summary_and_mode_switch(model):
    paths=[np.array([[0.,0.],[1.,float(i)]]) for i in range(10)]
    index=model.encode(paths)
    with pytest.raises(ValueError):model.candidate_indices(replace(index,ordered_summary=None))
    with pytest.raises(ValueError):model.candidate_indices(replace(index,ordered_summary=index.ordered_summary.clone()))
    model.config=RetrievalConfig('dtw','ordered')
    model.candidate_indices(index)
    model.config=RetrievalConfig('dtw','pse')
    legacy=model.encode(paths)
    model.config=RetrievalConfig('dtw','weighted')
    with pytest.raises(ValueError):model.candidate_indices(legacy)
    index.ordered_summary.add_(1)
    with pytest.raises(ValueError):model.candidate_indices(index)


def test_rectangular_bounded_scan_and_alignment(model):
    from leap.dtw.ordered_dtw import candidates
    from leap.common.interval_index import lex_keys
    rng=np.random.default_rng(82)
    d=model.encode([rng.normal(size=(2+i%19,2)) for i in range(4101)])
    q=model.encode([rng.normal(size=(2+i%11,2)) for i in range(2003)])
    dm=d.ordered_summary.cpu().numpy().reshape(-1,8,2).astype(np.float64)
    dn=np.diff(d.offsets.cpu().numpy())
    for query,same in ((q,False),(d,True)):
        qids=torch.arange(len(query.coordinates),device='cuda') if same else None
        initial=candidates(query,d,128,qids,max_work=True)
        allocations=[]
        original=torch.empty
        def bounded(shape,*args,**kwargs):
            if isinstance(shape,tuple) and len(shape)==2:
                allocations.append(shape)
                assert shape[0]<=2000 and shape[1]<=4096
            return original(shape,*args,**kwargs)
        with patch('torch.empty',side_effect=bounded):
            actual=model.candidate_indices(query,d,candidate_k=7)
        assert allocations and actual.shape==(len(query.coordinates),7)
        for row in (0,1999,2000,len(query.coordinates)-1):
            c=query.ordered_summary[row].reshape(8,2)
            n=int(query.offsets[row+1]-query.offsets[row])
            scale=torch.maximum(d.offsets[1:]-d.offsets[:-1],torch.tensor(n,device='cuda'))
            fast=(d.ordered_summary.reshape(-1,8,2)-c).square().sum(-1).sqrt().sum(1)/8*scale
            if same:fast[row]=torch.inf
            ids=torch.arange(len(d.coordinates),device='cuda')
            expected_pool=lex_keys(fast,ids).topk(128,largest=False).values&0xffffffff
            assert torch.equal(initial[row],expected_pool)
            pool=expected_pool.cpu().numpy()
            cm=c.cpu().numpy().astype(np.float64)
            distances=np.array([reference(cm,dm[j],n,dn[j]) for j in pool])
            expected=pool[np.lexsort((pool,distances))[:7]]
            np.testing.assert_array_equal(actual[row].cpu().numpy(),expected)


def test_range_defaults_explicit_override_and_wrapper(model):
    from leap.dtw.dtw_range import search
    rng=np.random.default_rng(73)
    index=model.encode([rng.normal(size=(3,2)) for _ in range(71)])
    tree=model.build_search_index(index)
    widths=[]
    def capture(*args,**kwargs):
        widths.append(args[6].shape[1])
        return search(*args,**kwargs)
    with patch('leap.dtw.dtw_range.search',side_effect=capture):
        default=model.search_range(index,search_index=tree)
        explicit=model.search_range(index,search_index=tree,seed_k=50)
        wrapper=model.search(index,search_index=tree,candidate_search='range')
        larger=model.search_range(index,search_index=tree,top_k=32)
        assert widths==[20,50,20,32]
    assert torch.equal(default['indices'],explicit['indices'])
    assert torch.equal(default['indices'],wrapper['indices'])
    assert torch.equal(larger['indices'][:,:20],default['indices'])
    for mode in ('ordered','pse'):
        model.config=RetrievalConfig('dtw',mode)
        widths.clear()
        with patch('leap.dtw.dtw_range.search',side_effect=capture):
            model.search_range(index,search_index=tree)
        assert widths==[50]
    with pytest.raises(ValueError):model.search_range(index,search_index=tree,seed_k=19)
