# Test group: RQ3.
# Paper RQ: RQ2: Role of Full Search; RQ3: Contributions of C1--C3.
# Checks: DTW filtering shortcuts preserve bound decisions.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Decision parity for default DTW filter execution shortcuts."""
import pytest
import torch

from leap.dtw import dtw_parallel
from leap.dtw.dtw_probe_intervals import pairs, keep_pairs, bound_torch


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA filter tests')


@pytest.mark.parametrize('size',[0,1,4096])
@torch.no_grad()
def test_ordered_shortcircuit_threshold_boundaries(size):
    generator=torch.Generator(device='cuda').manual_seed(230930)
    n=max(size,1)
    a0=torch.rand((n,4,16,1),device='cuda',generator=generator)*20
    b0=torch.rand((n,4,16,1),device='cuda',generator=generator)*20
    a=torch.cat((a0,a0+torch.rand_like(a0)),dim=-1)
    b=torch.cat((b0,b0+torch.rand_like(b0)),dim=-1)
    lengths=torch.arange(n,device='cuda')%11+1
    offsets=torch.cat((torch.zeros(1,device='cuda',dtype=torch.int64),lengths.cumsum(0)))
    ids=torch.arange(size,device='cuda')
    ref=pairs(a,b,offsets,offsets,ids,ids)
    expected=bound_torch(a[:size],b[:size],lengths[:size],lengths[:size])
    torch.testing.assert_close(ref,expected,rtol=2e-6,atol=2e-5)
    for limits in [ref,torch.nextafter(ref,torch.full_like(ref,-torch.inf)),
                   torch.nextafter(ref,torch.full_like(ref,torch.inf)),
                   torch.zeros_like(ref),torch.full_like(ref,-1),torch.full_like(ref,torch.inf)]:
        assert torch.equal(keep_pairs(a,b,offsets,offsets,ids,ids,limits),ref<=limits)
    if size:
        with pytest.raises(ValueError):
            keep_pairs(a,b,offsets,offsets,ids,ids,ref.double())


@pytest.mark.parametrize('seed_count',[20,80])
@torch.no_grad()
def test_long_member_tiles_keep_all_decisions(seed_count):
    generator=torch.Generator(device='cuda').manual_seed(340930)

    def pack(lengths):
        curves=[torch.randn((n,2),device='cuda',generator=generator)*.1+i*.2 for i,n in enumerate(lengths)]
        points=torch.cat(curves)
        offsets=torch.tensor([0]+list(torch.tensor(lengths).cumsum(0).tolist()),device='cuda')
        boxes=torch.stack([torch.cat((p.amin(0),p.amax(0))) for p in curves])
        return points,offsets,boxes

    points,offsets,boxes=pack([1,2,7,31,128,129,511,1025]*16+[3836])
    qp,qo,qbox=pack([1,17,524])
    members=torch.arange(192,device='cuda').reshape(3,64)
    members[members>=129]=-1
    lids=torch.arange(3,device='cuda',dtype=torch.int32).repeat(3)
    rows=torch.arange(3,device='cuda',dtype=torch.int32).repeat_interleave(3)
    seeds=torch.arange(seed_count,device='cuda').expand(3,-1).contiguous()
    self_ids=torch.tensor([-1,1,128],device='cuda')
    slack=torch.full((3,),2e-5,device='cuda')
    for r in [0.,.1,10.,1000.,5000.,1e8]:
        radius=torch.full((3,),r,device='cuda')
        outputs=[]
        for tile in [64,4]:
            keep=torch.empty((9,64),device='cuda',dtype=torch.int32)
            examined=torch.empty_like(keep)
            dtw_parallel.check_leaf_tiles[(9,64//tile)](
                points,offsets,qp,qo,members,boxes,qbox,radius,slack,self_ids,seeds,lids,rows,
                keep,examined,64,4096,seed_count,tile,64,seed_count>64,seed_count.bit_length(),
                num_warps=8,enable_fp_fusion=False)
            outputs.append((keep,examined))
        assert torch.equal(outputs[0][0],outputs[1][0])
        assert torch.equal(outputs[0][1],outputs[1][1])
