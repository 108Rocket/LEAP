# Test group: RQ2.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Parallel DTW subtree traversal preserves candidate enumeration.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Exact traversal, compaction, precision and boundary gates."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import triton

from leap.dtw import dtw_parallel as dp
from leap.common.runtime import options as opt


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='L20 CUDA required')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('radial',[False,True])
@pytest.mark.parametrize('groups,n',[(2,1),(256,1),(1024,1),(256,3)])
@torch.no_grad()
def test_exact_leaves_and_visits(dtype,radial,groups,n):
    torch.manual_seed(402)
    device='cuda'
    lengths=[17,64,129][:n]
    offsets=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device=device)
    points=torch.rand((sum(lengths),2),device=device,dtype=dtype)*4
    coordinates=torch.rand((n,128),device=device,dtype=dtype)
    q=SimpleNamespace(points=points,offsets=offsets,coordinates=coordinates)
    node_fields=torch.rand((2*groups,128),device=device,dtype=dtype)
    lo=torch.rand((2*groups,2),device=device,dtype=dtype)*8-2
    boxes=torch.cat((lo,lo+.5),1)
    tree=SimpleNamespace(groups=groups,tree=node_fields)
    extra={'treebox':boxes,'block':256}
    qgeom={'box':torch.zeros((n,4),device=device,dtype=dtype),
           'block':triton.next_power_of_2(max(lengths))}
    radial_fields=torch.rand((n,16,6),device=device,dtype=dtype)
    low=torch.rand((2*groups,16),device=device,dtype=dtype)*4
    radial_tree=torch.stack((low,low+.5),-1) if radial else None
    pf=(radial_fields,radial_fields) if radial else None

    def execute(enabled,radius):
        with patch.object(opt,'dtw_subtree_parallel',enabled):
            leaves,counts,visits,parts=dp.enumerate_leaves(
                q,tree,extra,qgeom,radius,torch.zeros_like(radius),pf,radial_tree)
        total=int(counts.sum());maximum=int(counts.max())
        ids=torch.empty(total,device=device,dtype=torch.int32)
        rows=torch.empty_like(ids)
        off=counts.long().cumsum(0)-counts
        if total:
            dp.compact_leaves[(n*parts,triton.cdiv(maximum,128))](
                leaves,counts,off,ids,rows,groups//parts,128,parts)
        return ids,rows,visits,counts.reshape(n,parts).sum(1),parts

    for value in (-1.,0.,5.,20.,100.,1e8):
        radius=torch.full((n,),value,device=device,dtype=dtype)
        old=execute(False,radius);new=execute(True,radius)
        assert new[-1]==(128 if n==1 and groups>=256 else 1)
        for a,b in zip(old[:-1],new[:-1]):
            assert torch.equal(a,b)
        if value<0:
            assert not len(new[0])
        if value==1e8:
            assert len(new[0])==n*groups
            assert (new[2][:,0]==2*groups-1).all()


@torch.no_grad()
def test_workspace_limits_remain_bounded():
    # Allocation bounds are checked before either traversal path is selected.
    import inspect
    text=inspect.getsource(dp.enumerate_parallel)
    assert text.index('n*g>MAX_LEAF_CELLS')<text.index('enumerate_leaves(')
    assert 'tasks*b>MAX_MEMBER_CELLS' in text
