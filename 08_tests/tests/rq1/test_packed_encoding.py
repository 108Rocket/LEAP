# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance.
# Checks: Packed input encoding, batching and search-output equivalence.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

from dataclasses import replace
from pathlib import Path
import warnings

import numpy as np
import pytest
import torch

from leap import LEAP
from leap import RetrievalConfig
from leap import PackedTrajectories
from leap.common.data import checked_pack, checked_packed
from leap.common.raw import normalize_points_serial


def test_packed_structure_and_lazy_views():
    points = np.arange(16, dtype=np.float64).reshape(-1, 2)
    offsets = np.array([0, 2, 5, 8])
    paths = PackedTrajectories(points, offsets)
    assert len(paths) == 3 and np.shares_memory(paths[1], points)
    np.testing.assert_array_equal(paths[-1], points[5:])
    assert len(paths[1:]) == 2
    with pytest.raises(IndexError):
        paths[3]
    for ends in ([0], [0,1,8], [1,8], [0,9,8], [0,5,3,8], [0,-1,8], [0,7]):
        with pytest.raises(ValueError):checked_packed(points, np.array(ends))
    for ends in (np.array([0.,8.]), np.array([False,True])):
        with pytest.raises(ValueError):checked_packed(points, ends)


def legacy_encode(model, paths):
    from leap.common.dual_field import fields_packed
    from leap.hausdorff.pointset_index import fields
    points, offsets = checked_pack(paths)
    points = normalize_points_serial(points, model.mean, model.std)
    p, o = torch.from_numpy(points).to(model.device), torch.from_numpy(offsets).to(model.device)
    function = fields if model.config.hausdorff_fields == 'near_far' else fields_packed
    chunk = min(1000, model.batch_size)
    z = torch.cat([function(p, o[i:i+chunk+1], model.anchors)
                   for i in range(0, len(offsets)-1, chunk)])
    return model._finish_encoding(z, p, o)


def equal(a, b):
    def same(x,y):
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for key in x:same(x[key],y[key])
        elif isinstance(x,torch.Tensor):assert torch.equal(x,y)
        else:assert x==y
    assert a.asset_id == b.asset_id
    for key in ('coordinates','points','offsets','ordered_summary','dfd_summary'):
        x,y=getattr(a,key),getattr(b,key)
        if x is None:assert y is None
        else:same(x,y)
    if a.dtw_probe_summary is not None:
        for key in a.dtw_probe_summary:assert torch.equal(a.dtw_probe_summary[key],b.dtw_probe_summary[key])


@pytest.mark.parametrize('metric', ['polyline','pointset','dfd','dtw'])
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_encode_and_search_exact(metric, dtype):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    config = RetrievalConfig() if metric=='polyline' else (
        RetrievalConfig(hausdorff_target='pointset',hausdorff_fields='near_far') if metric=='pointset'
        else RetrievalConfig(distance=metric,dtw_seed='signal'))
    model=LEAP.from_asset(config=replace(config,parameter_mode='auto'))
    rng=np.random.default_rng(905)
    paths=[(rng.normal(size=(2+i%16,2))*model.std+model.mean).astype(dtype) for i in range(140)]
    points,offsets=checked_pack(paths)
    points=points.astype(dtype)
    reference=legacy_encode(model,paths)
    for got in (model.encode(paths),model.encode(PackedTrajectories(points,offsets)),model.encode_packed(points,offsets)):
        equal(got,reference)
    encoder=model.packed_encoder()
    got=encoder.encode(points,offsets)
    stored=got.points.clone()
    encoder.encode(points[:offsets[1]],np.array([0,offsets[1]]))
    assert torch.equal(stored,got.points)
    readonly=offsets.copy();readonly.flags.writeable=False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        equal(encoder.encode(points,readonly),reference)
    assert not caught
    tree=model.build_search_index(reference)
    new_tree=model.build_search_index(got)
    query=model.encode(paths[:2])
    for k in (1,20,100):
        a=model.search_range(query,reference,search_index=tree,top_k=k)
        b=model.search_range(query,got,search_index=new_tree,top_k=k)
        for name in ('indices','distances'):assert torch.equal(a[name],b[name])


def test_finite_mutation_and_fp64_guards():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    model=LEAP.from_asset()
    points=np.tile(model.mean,(8,1));offsets=np.array([0,4,8])
    paths=PackedTrajectories(points,offsets)
    model.encode(paths)
    for value in (np.nan,np.inf,1e300):
        points[-1,0]=value
        with pytest.raises(ValueError,match='overflow|finite'):model.encode(paths)
    points[:]=model.mean
    offsets[1]=1
    with pytest.raises(ValueError):model.encode(paths)
    offsets[1]=4
    high=LEAP.from_asset(config=RetrievalConfig(precision='fp64'))
    equal(high.encode(list(paths)),high.encode_packed(points,offsets))
    offsets[-1]+=1
    with pytest.raises(ValueError,match='span'):
        high.encode(paths)
    offsets[-1]-=1
    with pytest.raises(ValueError,match='float64'):
        high.encode_packed(points.astype('float32'),offsets)


def test_npz_loader_preserves_packed_input(tmp_path):
    from leap.reproduction.replay import geometry
    p=tmp_path/'geometry.npz'
    points=np.arange(12,dtype='float64').reshape(6,2)
    np.savez(p,points=points,offsets=[0,2,6],geometry_mean=[0,0],geometry_std=[1,1])
    packed,_,_=geometry({'path':str(p)},packed=True)
    assert isinstance(packed,PackedTrajectories)
    expected,_,_=geometry({'path':str(p)})
    for a,b in zip(packed,expected):np.testing.assert_array_equal(a,b)


def test_explicit_encoding_batch_size_is_respected(monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from leap.common.encoding import PackedEncoder
    default=LEAP.from_asset()
    small=LEAP.from_asset(batch_size=5)
    assert default.batch_size==65536
    assert LEAP.from_asset(batch_size=1000).packed_encoder().trajectory_chunk==1000
    assert LEAP.from_asset(config=RetrievalConfig(precision='fp64')).batch_size==1000
    sizes=[]
    original=PackedEncoder._fields
    def counted(self,points,offsets,output,start,stop):
        sizes.append(stop-start)
        return original(self,points,offsets,output,start,stop)
    monkeypatch.setattr(PackedEncoder,'_fields',counted)
    paths=[np.tile(default.mean,(2,1)) for _ in range(16)]
    a=default.encode(paths)
    assert sizes==[16]
    sizes.clear()
    b=small.encode(paths)
    assert sizes==[5,5,5,1]
    equal(a,b)
