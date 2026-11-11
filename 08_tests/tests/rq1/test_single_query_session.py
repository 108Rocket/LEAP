# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Single-request sessions preserve results, state isolation and validation.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from leap import LEAP
from leap import RetrievalConfig
from leap.common.raw import normalize_points_serial, normalize_points_into


def test_normalize_into_exact():
    rng=np.random.default_rng(51)
    mean=np.array([116.41234567,39.9456789])
    std=np.array([0.00321,0.041])
    raw=mean+rng.normal(size=(43,2))*std
    backing=np.full((64,2),-71.,dtype=np.float32)
    normalize_points_into(raw,mean,std,backing[:43])
    np.testing.assert_array_equal(backing[:43].view(np.uint32),
                                  normalize_points_serial(raw,mean,std).view(np.uint32))
    assert (backing[43:]==-71.).all()


@pytest.fixture(params=['hausdorff','dfd','dtw'])
def example(request):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    config=RetrievalConfig.from_json(Path(__file__).parents[3]/'02_configs'/request.param/f'single_query_{request.param}.json')
    model=LEAP.from_asset(config=config)
    rng=np.random.default_rng(9)
    paths=[model.mean+rng.normal(size=(int(n),2))*model.std
           for n in rng.integers(3,32,size=128)]
    database=model.encode(paths)
    tree=model.build_search_index(database)
    return model,database,tree,paths


def native(model,db,tree,raw,mode):
    query=model.encode([raw])
    if mode=='ranking':
        return model.candidate_indices(query,db,candidate_k=20,search_index=tree).cpu().numpy()
    return model.search_range(query,db,top_k=20,search_index=tree)['indices'].cpu().numpy()


def test_session_matches_repeated_owned_results(example):
    model,db,tree,paths=example
    session=model.query_session(db,search_index=tree,initial_capacity=2)
    saved=[]
    for mode in ('ranking','full'):
        for raw in (paths[0],paths[1],paths[3],paths[0]):
            expected=native(model,db,tree,raw,mode)
            got=session.search(raw,mode=mode)
            np.testing.assert_array_equal(got,expected)
            reference=model.encode([raw])
            assert torch.equal(session.fields,reference.coordinates)
            assert torch.equal(session.points[:len(raw)],reference.points)
            assert session.capacity>=len(raw)
            saved.append((got,expected.copy()))
    for got,expected in saved:
        np.testing.assert_array_equal(got,expected)
    ptr=session.device_packet.data_ptr()
    session.search(paths[0],mode='ranking')
    assert session.device_packet.data_ptr()==ptr


def test_invalid_input_preserves_validation_and_recovers(example):
    model,db,tree,paths=example
    session=model.query_session(db,search_index=tree)
    for raw in (np.zeros((1,2)),np.zeros((3,3)),np.array([[np.nan,0.],[0.,0.]])):
        with pytest.raises(ValueError):
            session.search(raw)
    overflowing=model.mean+np.full((3,2),1e20)*model.std
    with pytest.raises(ValueError,match='overflowed'):
        model.encode([overflowing])
    for mode in ('ranking','full'):
        with pytest.raises(ValueError,match='overflowed'):
            session.search(overflowing,mode=mode)
        np.testing.assert_array_equal(session.search(paths[0],mode=mode),
                                      native(model,db,tree,paths[0],mode))


def test_session_guards_changed_database(example):
    model,db,tree,paths=example
    session=model.query_session(db,search_index=tree)
    db.coordinates.add_(0.)
    with pytest.raises(ValueError,match='modified database'):
        session.search(paths[0])


def test_serializes_concurrent_callers(example):
    model,db,tree,paths=example
    session=model.query_session(db,search_index=tree)
    expected=[native(model,db,tree,p,'ranking') for p in paths[:4]]
    # Warm specialization outside worker threads; the same buffers serve all.
    session.search(paths[0],mode='ranking')
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs=list(pool.map(lambda p:session.search(p,mode='ranking'),paths[:4]))
    for a,b in zip(outputs,expected):
        np.testing.assert_array_equal(a,b)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fused_tree_slack_ties_and_overflow():
    from leap.common.interval_index import IntervalSearchIndex, tree_kernel
    from leap.common.model import TrajectoryIndex
    device='cuda'
    rng=np.random.default_rng(4)
    for scale in (1e-8,1.,1e8):
        a=rng.uniform(.1,1.5,size=(256,64)).astype(np.float32)*scale
        b=a*rng.uniform(0.,1.,size=a.shape).astype(np.float32)
        z=torch.tensor(np.concatenate([a,b],axis=1),device=device)
        z[1]=z[0]
        db=TrajectoryIndex(z,torch.zeros((512,2),device=device),
                           torch.arange(0,514,2,device=device), 'test')
        tree=IntervalSearchIndex(db)
        qids=torch.tensor([-1],device=device)
        for k in (1,20,64):
            for q in (z[:1],z[8:9],(z[8:9]+z[9:10])*.5):
                ref,visits=tree.candidates(q,qids,k)
                out=torch.empty_like(ref)
                got_visits=torch.empty_like(visits)
                tree_kernel[(1,)](q,q,tree.d,tree.members,tree.tree,qids,tree.database_scale,
                    out,got_visits,tree.groups,tree.block,k,64,True,
                    num_warps=4,enable_fp_fusion=False)
                assert torch.equal(ref,out)
                assert torch.equal(visits,got_visits)
        q=torch.full((1,128),float('inf'),device=device)
        tree_kernel[(1,)](q,q,tree.d,tree.members,tree.tree,qids,tree.database_scale,
            out,got_visits,tree.groups,tree.block,64,64,True,
            num_warps=4,enable_fp_fusion=False)
        assert (out==-1).all() and (got_visits==0).all()
