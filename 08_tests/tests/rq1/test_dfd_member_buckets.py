# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: DFD length bucketing preserves original distances.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Default DFD buckets preserve exact distances across length boundaries."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from leap.dfd.dfd import dfd_pairs
from leap.dfd.dfd_refinement import cached_pairs

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

def pack(lengths):
    rng=np.random.default_rng(9130)
    points=np.concatenate([rng.normal(0,.05,(n,2)).cumsum(0).astype(np.float32) for n in lengths])
    return SimpleNamespace(points=torch.tensor(points,device='cuda'),
        offsets=torch.tensor([0]+np.cumsum(lengths).tolist(),device='cuda'),
        dfd_summary={'block':2**(max(lengths)-1).bit_length()})

@pytest.mark.parametrize('qlength',[17,129,257])
@torch.no_grad()
def test_default_bucket_distances(qlength):
    q=pack([qlength]);d=pack([2,63,64,65,127,128,129,255,256,257])
    ids=torch.arange(10,device='cuda');rows=torch.zeros_like(ids)
    initial=ids[:2].reshape(1,2)
    raw=dfd_pairs(q,d,rows[:2],ids[:2]).reshape(1,2)
    radii=torch.full((1,),torch.inf,device='cuda')
    values,flags=cached_pairs(q,d,rows,ids,initial,raw,radii)
    assert torch.equal(values,dfd_pairs(q,d,rows,ids))
    empty=cached_pairs(q,d,rows[:0],ids[:0],initial,raw,radii)
    assert empty[0].shape==empty[1].shape==(0,)
