# Test group: RQ2.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Hausdorff early refinement preserves survivors and strict threshold ties.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Fixed-radius Hausdorff RGR: exact survivors, strict ties, adapter isolation."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from leap.common.early_refinement import bounded_hausdorff_distances
from leap.common.raw_geometry_refinement import hausdorff_distances

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA retrieval tests')


def packed(paths):
    lengths=np.array([len(p) for p in paths])
    return SimpleNamespace(points=torch.tensor(np.concatenate(paths),device='cuda',dtype=torch.float32),
        offsets=torch.tensor(np.r_[0,lengths.cumsum()],device='cuda',dtype=torch.int64),
        coordinates=torch.empty((len(paths),128),device='cuda'))


@pytest.mark.parametrize('block',[8,16,32])
def test_exact_survivors_and_strict_boundary(block):
    rng=np.random.default_rng(412)
    paths=[rng.normal(size=(n,2)).cumsum(0) for n in (2,3,8,31,32,33,65,129)]
    paths.extend([np.zeros((3,2)),paths[-1].copy()])
    d=packed(paths)
    qids=torch.arange(len(paths),device='cuda').repeat_interleave(len(paths))
    dids=torch.arange(len(paths),device='cuda').repeat(len(paths))
    expected=hausdorff_distances(d,d,qids,dids)
    for factor in (0.,.5,1.,2.,float('inf')):
        radii=expected.reshape(len(paths),-1).amax(1)*factor
        actual=bounded_hausdorff_distances(d,d,qids,dids,radii,source_block=block)
        keep=expected<=radii[qids]
        assert torch.equal(actual[keep],expected[keep])
        assert torch.isinf(actual[~keep]).all()
    # Every individual score becomes its own query budget, including equality.
    for j in range(len(paths)):
        targets=torch.full((len(paths),),j,device='cuda',dtype=torch.int64)
        rows=torch.arange(len(paths),device='cuda')
        full=hausdorff_distances(d,d,rows,targets)
        assert torch.equal(bounded_hausdorff_distances(d,d,rows,targets,full,source_block=block),full)


def test_full_opposite_scan_and_reverse_direction():
    # A distant first segment must not cause rejection before later near segments.
    query=packed([np.array([[0.,0.],[101.,0.]]),np.array([[0.,0.],[1.,0.]])])
    target=packed([np.concatenate((np.tile([[100.,0.],[101.,0.]],(20,1)),[[0.,0.],[1.,0.]])),
                   np.array([[0.,0.],[1.,0.],[100.,0.]])])
    rows=torch.arange(2,device='cuda')
    full=hausdorff_distances(query,target,rows,rows)
    assert torch.equal(bounded_hausdorff_distances(query,target,rows,rows,full),full)
    result,diag=bounded_hausdorff_distances(query,target,rows,rows,torch.ones(2,device='cuda'),diagnostic=True)
    assert result[0]==0 and not diag['rejected'][0]
    assert torch.isinf(result[1]) and diag['rejected'][1]


def test_early_exit_reduces_geometric_work():
    q=packed([np.zeros((128,2))])
    d=packed([np.full((128,2),100.)])
    ids=torch.zeros(1,device='cuda',dtype=torch.int64)
    result,diag=bounded_hausdorff_distances(q,d,ids,ids,torch.ones(1,device='cuda'),diagnostic=True)
    assert torch.isinf(result).all()
    assert diag['work'].item()==32*127
    assert diag['work'].item()<2*128*127
    empty=ids[:0]
    assert bounded_hausdorff_distances(q,d,empty,empty,torch.ones(1,device='cuda')).shape==(0,)


def test_full_search_and_adapter_compatibility():
    from leap import LEAP
    from leap.hausdorff import range_search
    model=LEAP.from_asset()
    rng=np.random.default_rng(713)
    paths=[rng.normal(size=(2+i%67,2)).cumsum(0)*50+model.mean for i in range(129)]
    paths[1]=paths[0].copy()
    paths[2]=np.repeat(paths[0][:1],4,axis=0)
    d=model.encode(paths)
    tree=model.build_search_index(d)
    ids=torch.arange(len(paths),device='cuda')
    for k in (1,5,20):
        reference=range_search.search_chunk(d,d,tree,ids,top_k=k,bounded=False)
        actual=range_search.search_chunk(d,d,tree,ids,top_k=k)
        for key in ('indices','distances','counts','radius','seed_ids'):
            assert torch.equal(reference[key],actual[key]),key
    def alternate(q,d,qi,di,point=False):
        return hausdorff_distances(q,d,qi,di,point=True)
    with patch.object(range_search,'hausdorff_distances',alternate):
        with patch.object(range_search,'bounded_hausdorff_distances',side_effect=AssertionError('wrong target')):
            result=range_search.search_chunk(d,d,tree,ids)
            reference=range_search.search_chunk(d,d,tree,ids,bounded=False)
            assert torch.equal(result['indices'],reference['indices'])


def test_input_contract():
    d=packed([np.zeros((2,2))])
    ids=torch.zeros(1,device='cuda',dtype=torch.int64)
    radii=torch.ones(1,device='cuda')
    for kwargs in ({'source_block':4},):
        with pytest.raises(ValueError):bounded_hausdorff_distances(d,d,ids,ids,radii,**kwargs)
    with pytest.raises(ValueError):bounded_hausdorff_distances(d,d,ids.int(),ids,radii)
    with pytest.raises(ValueError):bounded_hausdorff_distances(d,d,ids,ids,radii.double())
