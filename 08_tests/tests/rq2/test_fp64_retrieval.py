# Test group: RQ2.
# Paper RQ: RQ2: Role of Full Search.
# Checks: Higher-precision filtering and final ordering for the remaining-miss control.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""FP64 geometry must remain FP64 through filtering and final ordering."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from leap import LEAP
from leap import RetrievalConfig
from leap.common.precision import raw_pairs, stable_top, search, subset

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA retrieval tests')


def model_for(metric,large=False):
    return LEAP.from_asset(config=RetrievalConfig(distance=metric,precision='fp64',
        full_large_k=large,dtw_seed='signal'))


def trajectories(model,n):
    rng=np.random.default_rng(106)
    paths=[model.mean+model.std*rng.normal(size=(2+i%13,2)).cumsum(0) for i in range(n)]
    paths[1]=paths[0].copy()
    paths[2]=np.repeat(model.mean[None],5,axis=0)
    return paths


@pytest.mark.parametrize('metric',['hausdorff','dfd','dtw'])
@pytest.mark.parametrize('large',[False,True])
@torch.no_grad()
def test_complete_matches_unfiltered(metric,large):
    model=model_for(metric,large)
    paths=trajectories(model,75 if large else 67)
    d=model.encode(paths)
    q=model.encode([paths[i] for i in (0,2,17)])
    tree=model.build_search_index(d)
    n=len(d.coordinates);k=65 if large else 20
    rows=torch.arange(3,device=model.device)[:,None].expand(3,n).reshape(-1)
    ids=torch.arange(n,device=model.device)[None].expand(3,n).reshape(-1)
    vals=raw_pairs(metric,q,d,rows,ids).reshape(3,n)
    assert vals.dtype==torch.float64
    excluded=torch.tensor([0,2,17],device=model.device)
    for self_exclude in (False,True):
        scores=vals.clone()
        if self_exclude:
            scores[torch.arange(3,device=model.device),excluded]=torch.inf
        got=search(model,q,d,search_index=tree,top_k=k,
                   self_ids=excluded if self_exclude else None)
        assert got['execution']['serial_fallback'] is False
        for row in range(3):
            expected=stable_top(scores[row],ids[:n],k)
            assert torch.equal(got['indices'][row],expected[0])
            torch.testing.assert_close(got['distances'][row],expected[1],rtol=0,atol=0)
    assert d.points.dtype==d.coordinates.dtype==torch.float64
    expected=(np.concatenate(paths)-model.mean)/model.std
    np.testing.assert_array_equal(d.points.cpu().numpy(),expected)
    assert d.ranking_index.points.dtype==torch.float32
    with pytest.raises(ValueError,match='float64 input'):
        model.encode([paths[0].astype(np.float32)])
    with pytest.raises(ValueError,match='precision'):
        model._check_index(d.ranking_index)
    with pytest.raises(ValueError,match='stale'):
        model._check_index(replace(d,points=d.points.clone()))


def test_score_bits_not_discarded():
    values=torch.tensor([1+1e-10,1.,1.],device='cuda',dtype=torch.float64)
    ids=torch.tensor([0,2,1],device='cuda')
    assert stable_top(values,ids,3)[0].tolist()==[1,2,0]


@pytest.mark.parametrize('metric',['hausdorff','dfd','dtw'])
@torch.no_grad()
def test_raw_scores_independent_double_oracle(metric):
    from traj_dist import distance
    model=model_for(metric)
    paths=trajectories(model,20)
    index=model.encode(paths)
    rows=torch.arange(20,device=model.device)
    ids=rows.roll(7)
    result=raw_pairs(metric,index,index,rows,ids).cpu().numpy()
    normalized=[(p-model.mean)/model.std for p in paths]
    if metric=='hausdorff':
        def directed(a,b):
            v=b[1:]-b[:-1]
            r=a[:,None]-b[None,:-1]
            t=(r*v).sum(-1)/np.maximum((v*v).sum(-1),1e-30)
            return np.sqrt(((r-np.clip(t,0,1)[...,None]*v)**2).sum(-1)).min(1).max()
        fn=lambda a,b:max(directed(a,b),directed(b,a))
    else:
        fn=getattr(distance,'discret_frechet' if metric=='dfd' else 'dtw')
    expected=[fn(normalized[i],normalized[j]) for i,j in zip(rows.tolist(),ids.tolist())]
    np.testing.assert_allclose(result,expected,rtol=5e-14,atol=5e-14)


@pytest.mark.parametrize('metric',['hausdorff','dfd','dtw'])
def test_single_request_and_viewer(metric):
    from leap.common.query_search import search_row
    model=model_for(metric)
    paths=trajectories(model,71)
    database=model.encode(paths)
    tree=model.build_search_index(database)
    q=model.encode([paths[17]])
    expected=model.search_range(q,database,search_index=tree)['indices'].cpu().numpy()
    model.config=replace(model.config,query_profile='single_query')
    session=model.query_session(database,search_index=tree)
    np.testing.assert_array_equal(session.search(paths[17]),expected)
    for mode in ('initial','pool','complete'):
        out=search_row(model,database,tree,17,20,mode,50)
        assert len(out['ids'])==len(set(out['ids']))==20
        assert 17 not in out['ids']
    self_result=model.search_range(database,search_index=tree)['indices'][17].tolist()
    assert out['ids']==self_result


@pytest.mark.parametrize('metric',['hausdorff','dfd','dtw'])
def test_public_search_dispatch(metric):
    model=model_for(metric)
    paths=trajectories(model,75)
    d=model.encode(paths);q=model.encode(paths[:3]);tree=model.build_search_index(d)
    result=model.search(q,d,search_index=tree)
    assert result['indices'].shape==(3,20) and result['distances'].dtype==torch.float64
    with pytest.raises(ValueError,match='refinement'):
        model.search(q,d,search_index=tree,refinement='unknown')
    model.config=replace(model.config,full_large_k=True)
    result=model.search(q,d,search_index=tree,top_k=65)
    assert result['execution']['full_database'] and not result['execution']['serial_fallback']
    with pytest.raises(ValueError,match='global range'):
        model.search(q,d,search_index=tree,candidate_search='full')


def test_precision_config(tmp_path):
    config=tmp_path/'precision.json'
    config.write_text('{"precision":"fp64"}')
    assert RetrievalConfig.from_json(config).precision=='fp64'
    assert RetrievalConfig().precision=='fp32'
    with pytest.raises(ValueError,match='precision'):
        RetrievalConfig(precision='fp16')
    with pytest.raises(ValueError,match='legacy'):
        RetrievalConfig(precision='fp64',large_k=True)
    with pytest.raises(ValueError,match='optimized'):
        RetrievalConfig(distance='dtw',precision='fp64',dtw_execution='reference')


@pytest.mark.parametrize('seed',['pse','ordered','weighted'])
def test_non_neural_dtw_initializers(seed):
    model=LEAP.from_asset(config=RetrievalConfig(distance='dtw',dtw_seed=seed,precision='fp64'))
    paths=trajectories(model,70)
    d=model.encode(paths);q=model.encode(paths[:3]);tree=model.build_search_index(d)
    result=model.search_range(q,d,search_index=tree,top_k=20)
    rows=torch.arange(3,device='cuda')[:,None].expand(3,70).reshape(-1)
    ids=torch.arange(70,device='cuda')[None].expand(3,70).reshape(-1)
    values=raw_pairs('dtw',q,d,rows,ids).reshape(3,70)
    for i in range(3):
        expected=stable_top(values[i],ids[:70],20)
        assert torch.equal(expected[0],result['indices'][i])


def test_pointset_fp64():
    model=LEAP.from_asset(config=RetrievalConfig(precision='fp64',full_large_k=True,hausdorff_target='pointset'))
    paths=trajectories(model,75)
    d=model.encode(paths);q=model.encode(paths[:3]);tree=model.build_search_index(d)
    result=model.search_range(q,d,search_index=tree,top_k=65)
    rows=torch.arange(3,device='cuda')[:,None].expand(3,75).reshape(-1)
    ids=torch.arange(75,device='cuda')[None].expand(3,75).reshape(-1)
    values=raw_pairs('hausdorff',q,d,rows,ids,True).reshape(3,75)
    for i in range(3):
        assert torch.equal(stable_top(values[i],ids[:75],65)[0],result['indices'][i])
