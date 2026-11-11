# Test group: RQ2. Exact refinement and opt-in dynamic Top-k thresholds.
from dataclasses import replace
from unittest.mock import patch
import json
import numpy as np
import pytest
import torch

from leap import RetrievalConfig, load_checkpoint


def test_default_and_json_roundtrip(tmp_path):
    for distance in ('hausdorff','dfd','dtw'):
        assert RetrievalConfig(distance=distance).dynamic_threshold is False
        path=tmp_path/(distance+'.json')
        path.write_text(json.dumps(dict(distance=distance,dynamic_threshold=True)))
        assert RetrievalConfig.from_json(path).dynamic_threshold is True
    for invalid in ('true',1,None,[]):
        with pytest.raises(ValueError,match='dynamic_threshold'):
            RetrievalConfig(dynamic_threshold=invalid)


def fixture(metric):
    model=load_checkpoint('sdd.porto.'+metric)
    rng=np.random.default_rng(2000)
    lengths=(2,3,4,31,32,33,63,64,65,127,128,129,255,257)
    paths=[rng.normal(size=(lengths[i%len(lengths)],2)).cumsum(0)*.1*model.std+model.mean for i in range(81)]
    paths[-1]=paths[0].copy()
    db=model.encode(paths)
    q=model.encode(paths[:5])
    return model,q,db,model.build_search_index(db)


def same_result(a,b):
    assert torch.equal(a['indices'],b['indices'])
    assert torch.equal(a['distances'].view(torch.int32),b['distances'].view(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='L20 CUDA validation')
@pytest.mark.parametrize('metric',['hausdorff','dfd','dtw'])
@pytest.mark.parametrize('same',[False,True])
@torch.no_grad()
def test_enabled_parity_and_self_exclusion(metric,same):
    model,q,db,tree=fixture(metric)
    q=db if same else q
    original=model.config
    for k in (1,20,64):
        # Non-Top20 requests deliberately retain the established execution path.
        options=dict(search_index=tree,top_k=k,query_batch=17,
                     initial_k=max(k,dict(hausdorff=50,dfd=32,dtw=20)[metric]))
        model.config=original
        expected=model.search_range(q,None if same else db,**options)
        model.config=replace(original,dynamic_threshold=True)
        actual=model.search_range(q,None if same else db,**options)
        same_result(actual,expected)
        if same:
            assert not bool((actual['indices']==torch.arange(len(q.coordinates),device='cuda')[:,None]).any())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='L20 CUDA validation')
@pytest.mark.parametrize('metric',['dfd','dtw'])
@torch.no_grad()
def test_dynamic_retry_reuses_seed_scores(metric):
    from leap.dfd import dfd_range, dynamic_search as df
    from leap.dtw import dtw_range, dynamic_search as dt
    model,q,db,tree=fixture(metric)
    model.config=replace(model.config,dynamic_threshold=True)
    expected=model.search_range(q,db,search_index=tree)
    ranges=dfd_range if metric=='dfd' else dtw_range
    module=df if metric=='dfd' else dt
    original=module.search
    def limited(query,*args,**kwargs):
        if len(query.coordinates)>2:
            raise MemoryError('Force split after initial distances have been cached')
        return original(query,*args,**kwargs)
    with patch.object(ranges,'initial_distances',wraps=ranges.initial_distances) as seeds:
        with patch.object(module,'search',side_effect=limited):
            actual=model.search_range(q,db,search_index=tree)
    assert seeds.call_count==1
    same_result(actual,expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='L20 CUDA validation')
@torch.no_grad()
def test_single_pass_length_layout_matches_native():
    from leap.dtw import dtw_bounded, bucketed_refinement
    _,q,db,_=fixture('dtw')
    ids=torch.arange(len(db.coordinates),device='cuda')[None].expand(len(q.coordinates),-1).reshape(-1)
    rows=torch.arange(len(q.coordinates),device='cuda')[:,None].expand(-1,len(db.coordinates)).reshape(-1)
    raw=dtw_bounded.pairs(q,db,rows,ids,bucket=True)
    for budget in (None,raw,raw*.5,raw*1.5):
        expected=dtw_bounded.pairs(q,db,rows,ids,budget,bucket=True)
        actual=bucketed_refinement.pairs(q,db,rows,ids,budget,bucket=True)
        assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
    assert not bucketed_refinement.pairs(q,db,rows[:0],ids[:0],bucket=True).numel()
    with pytest.raises(ValueError,match='maximum'):
        bucketed_refinement.pairs(q,db,rows,ids,bucket=True,maximum=32)
