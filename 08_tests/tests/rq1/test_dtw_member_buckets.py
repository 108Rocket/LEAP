# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: DTW leaf bucketing preserves results across length boundaries.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Boundary and layout parity for DTW leaf-width bucketing on L20."""
import pytest
import torch
from leap.dtw import dtw_parallel as dp

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def pack(lengths,constant=False):
    generator=torch.Generator(device='cuda').manual_seed(93421)
    curves=[torch.full((n,2),float(i+1),device='cuda') if constant else
            torch.randn((n,2),device='cuda',generator=generator)*.1+i*.2
            for i,n in enumerate(lengths)]
    points=torch.cat(curves)
    offsets=torch.tensor([0]+list(torch.tensor(lengths).cumsum(0).tolist()),device='cuda')
    boxes=torch.stack([torch.cat((p.amin(0),p.amax(0))) for p in curves])
    return points,offsets,boxes


def evaluate(p,po,q,qo,m,box,qbox,r,slack,self_ids,seeds,lids,rows,block):
    s=seeds.shape[1]
    outputs=[]
    for bucket in [False,True]:
        keep=torch.full((len(lids),64),-99,device='cuda',dtype=torch.int32)
        examined=torch.full_like(keep,-99)
        args=(p,po,q,qo,m,box,qbox,r,slack,self_ids,seeds,lids,rows,keep,examined)
        if bucket:
            dp.check_leaf_buckets(*args,64,block,s,4)
        elif len(lids):
            dp.check_leaf_tiles[(len(lids),16)](*args,64,block,s,4,64,s>64,s.bit_length(),
                                               num_warps=8,enable_fp_fusion=False)
        outputs.append((keep,examined))
    assert torch.equal(outputs[0][0],outputs[1][0]),'Pruning decisions changed'
    assert torch.equal(outputs[0][1],outputs[1][1]),'Member validity changed'
    return outputs


@pytest.mark.parametrize('seed_count',[1,20,80])
@torch.no_grad()
def test_mixed_lengths_and_masks(seed_count):
    lengths=[1,2,7,31,63,64,65,127]*8+[128,129,255,256,257,511,512,513]*8+[1025,2049,3836]
    p,po,box=pack(lengths)
    q,qo,qbox=pack([1,65,1025])
    m=torch.arange(192,device='cuda').reshape(3,64)
    m[m>=len(lengths)]=-1
    lids=torch.tensor([2,0,1,1,2,0,0,2,1],device='cuda',dtype=torch.int32)
    rows=torch.arange(3,device='cuda',dtype=torch.int32).repeat_interleave(3)
    seeds=torch.arange(seed_count,device='cuda').expand(3,-1).contiguous()
    self_ids=torch.tensor([-1,65,130],device='cuda')
    slack=torch.full((3,),2e-5,device='cuda')
    for value in [0.,.1,10.,1000.,5000.,1e8]:
        evaluate(p,po,q,qo,m,box,qbox,torch.full((3,),value,device='cuda'),slack,
                 self_ids,seeds,lids,rows,4096)


@torch.no_grad()
def test_query_width_dominates_and_padding_is_ignored():
    p,po,box=pack([2]*64)
    q,qo,qbox=pack([1025])
    m=torch.arange(128,device='cuda').reshape(2,64)
    m[1]=-1
    lids=torch.tensor([0,1],device='cuda',dtype=torch.int32)
    rows=torch.zeros(2,device='cuda',dtype=torch.int32)
    assert dp.member_task_widths(po,qo,m,lids,rows,64).tolist()==[1025,1025]
    evaluate(p,po,q,qo,m,box,qbox,torch.tensor([300.],device='cuda'),torch.zeros(1,device='cuda'),
             torch.tensor([-1],device='cuda'),torch.tensor([[-2]],device='cuda'),lids,rows,2048)


@torch.no_grad()
def test_exact_and_adjacent_thresholds():
    lengths=[1,2,31,32,33,63,64,65,127,128,129,255,256,257,511,512]*4
    po=torch.tensor([0]+list(torch.tensor(lengths).cumsum(0).tolist()),device='cuda')
    p=torch.zeros((sum(lengths),2),device='cuda');p[:,0]=1
    box=torch.tensor([[1.,0.,1.,0.]],device='cuda').repeat(64,1)
    q=torch.zeros((17,2),device='cuda');qo=torch.tensor([0,17],device='cuda')
    qbox=torch.zeros((1,4),device='cuda')
    m=torch.arange(64,device='cuda').reshape(1,64)
    lids=rows=torch.zeros(1,device='cuda',dtype=torch.int32)
    for value in [17.,64.,128.,512.]:
        r=torch.tensor([value],device='cuda')
        for limit in [r,torch.nextafter(r,torch.full_like(r,-torch.inf)),
                      torch.nextafter(r,torch.full_like(r,torch.inf))]:
            outputs=evaluate(p,po,q,qo,m,box,qbox,limit,torch.zeros_like(r),
                             torch.tensor([-1],device='cuda'),torch.tensor([[-2]],device='cuda'),
                             lids,rows,4096)
            expected=(torch.tensor(lengths,device='cuda').clamp_min(17)<=limit).int()
            assert torch.equal(outputs[1][0][0],expected)


@torch.no_grad()
def test_no_leaf_tasks():
    p,po,box=pack([2]);q,qo,qbox=pack([1])
    m=torch.full((1,64),-1,device='cuda',dtype=torch.int64)
    empty=torch.empty(0,device='cuda',dtype=torch.int32)
    evaluate(p,po,q,qo,m,box,qbox,torch.ones(1,device='cuda'),torch.zeros(1,device='cuda'),
             torch.tensor([-1],device='cuda'),torch.tensor([[-2]],device='cuda'),empty,empty,64)


@torch.no_grad()
def test_small_global_width():
    p,po,box=pack([7,16,31]);q,qo,qbox=pack([3])
    m=torch.full((1,64),-1,device='cuda',dtype=torch.int64);m[0,:3]=torch.arange(3,device='cuda')
    rows=torch.zeros(1,device='cuda',dtype=torch.int32)
    evaluate(p,po,q,qo,m,box,qbox,torch.ones(1,device='cuda'),torch.zeros(1,device='cuda'),
             torch.tensor([-1],device='cuda'),torch.tensor([[-2]],device='cuda'),rows,rows,32)
