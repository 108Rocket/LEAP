# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Single-request DTW compaction and storage reuse preserve results.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""L20 gates for stable compaction, reuse, empty sets and public auto plans."""
from unittest.mock import patch

import numpy as np
import pytest
import torch
import triton

from leap import LEAP
from leap import RetrievalConfig
from leap.common.runtime import options as opt
from leap.dtw import dtw_parallel as dp
from leap.dtw import dtw_single_layout as layout
from leap.common.runtime.plan import SingleQueryPlan
from leap.common.runtime.context import activate


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='L20 CUDA required')


def test_adopted_defaults():
    assert opt.dtw_subtree_parallel is True
    assert opt.dtw_single_layout is True


@pytest.mark.parametrize('tasks', [0, 1, 127, 128, 129, 4095, 8193, 32768])
@pytest.mark.parametrize('all_keep', [False, True])
@torch.no_grad()
def test_stable_pair_scan(tasks, all_keep):
    torch.manual_seed(815)
    keep = torch.randint(0, 2, (tasks, 64), device='cuda', dtype=torch.int32)
    if all_keep:
        keep.fill_(1)
    examined = torch.ones_like(keep)
    counts = torch.empty(tasks, device='cuda', dtype=torch.int32)
    checked = torch.empty_like(counts)
    offsets = torch.empty_like(counts)
    total = torch.full((1,), -7, device='cuda', dtype=torch.int32)
    visits = torch.tensor([[928, -7]], device='cuda', dtype=torch.int32)
    zero = torch.full((1,), -7, device='cuda', dtype=torch.int64)
    if tasks:
        layout.member_counts[(tasks,)](keep, examined, counts, checked, 64)
    layout.pair_scan[(1,)](counts, checked, offsets, total, visits, zero,
                          tasks, triton.next_power_of_2(max(1, tasks)), num_warps=8)
    expected = keep.sum(1)
    assert torch.equal(counts, expected)
    assert torch.equal(offsets, expected.cumsum(0) - expected)
    assert int(total[0]) == int(expected.sum())
    assert visits.tolist() == [[928, tasks * 64]]
    assert int(zero[0]) == 0


@pytest.mark.parametrize('parts', [1, 128])
@torch.no_grad()
def test_leaf_scan(parts):
    counts = torch.arange(parts, device='cuda', dtype=torch.int32) % 257
    for value in (counts, counts * 0, counts * 0 + 256):
        offsets = torch.empty_like(value)
        total = torch.empty(1, device='cuda', dtype=torch.int32)
        layout.leaf_scan[(1,)](value, offsets, total, parts)
        assert torch.equal(offsets, value.cumsum(0) - value)
        assert int(total[0]) == int(value.sum())


@pytest.fixture(scope='module')
def system():
    torch.manual_seed(829)
    cfg = RetrievalConfig('dtw', 'signal', dtw_routing='learned', parameter_mode='auto',
                          query_profile='single_query')
    model = LEAP.from_asset(config=cfg)
    rng = np.random.default_rng(829)
    paths = [model.mean + model.std * (rng.normal(size=(2+i%4, 2)).cumsum(0)
             + rng.normal(size=2)*8) for i in range(8193)]
    paths[1] = paths[0].copy()
    db = model.encode(paths)
    tree = model.build_search_index(db)
    assert tree.groups == 256
    q = model.encode([paths[3]])
    return model, paths, db, tree, q


@pytest.mark.parametrize('k', [1, 20, 1000])
@torch.no_grad()
def test_auto_full_exact_and_owned_outputs(system, k):
    model, paths, db, tree, q = system
    runtime = SingleQueryPlan(model, db, tree)
    with patch.object(opt, 'dtw_single_layout', False), patch.object(opt, 'dtw_subtree_parallel', False):
        expected = model.search_range(q, db, search_index=tree, top_k=k)
    original = layout.enumerate_members
    with patch.object(opt, 'dtw_single_layout', True), patch.object(opt, 'dtw_subtree_parallel', True):
        # Explicit runtime checks the fused path even when auto dispatch prefers
        # the generic session for large k. The public session is checked below.
        with activate(runtime, len(paths[3])), patch.object(layout, 'enumerate_members', wraps=original) as spy:
            actual = model.search_range(q, db, search_index=tree, top_k=k)
            assert spy.call_count > 0
        saved = {name: value.clone() for name, value in actual.items() if isinstance(value, torch.Tensor)}
        with activate(runtime, len(paths[3])):
            model.search_range(q, db, search_index=tree, top_k=1)
        for name in ('indices', 'distances', 'counts', 'radius'):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
            torch.testing.assert_close(actual[name], saved[name], rtol=0, atol=0)
        session = model.query_session(db, search_index=tree, top_k=k)
        np.testing.assert_array_equal(session.search(paths[3]), expected['indices'].cpu().numpy())


@pytest.mark.parametrize('parallel', [False, True])
@torch.no_grad()
def test_empty_allpass_reuse_and_workspace_bound(system, parallel):
    from leap.dtw.dtw_range import sidecars
    model, paths, db, tree, q = system
    qids = torch.full((1,), -1, device='cuda', dtype=torch.int64)
    initial = torch.arange(20, device='cuda').reshape(1, 20)
    runtime = SingleQueryPlan(model, db, tree)
    qgeom = sidecars(q)
    for value in (-1e8, 1e8, 5., -1e8):
        radius = torch.full((1,), value, device='cuda')
        args = (q, db, tree, tree.dtw_geometry, qgeom, qids, initial,
                radius, torch.zeros_like(radius))
        with patch.object(opt, 'dtw_subtree_parallel', parallel), activate(runtime, len(paths[3])):
            with patch.object(opt, 'dtw_single_layout', False):
                expected = dp.enumerate_parallel(*args)
            with patch.object(opt, 'dtw_single_layout', True):
                actual = dp.enumerate_parallel(*args)
            for x, y in zip(actual, expected):
                if isinstance(x, torch.Tensor):
                    assert torch.equal(x, y)
                else:
                    assert x == y
            if value == 1e8:
                assert actual[5] == len(paths) - 20
                with patch.object(opt, 'dtw_single_layout', True), patch.object(dp, 'MAX_MEMBER_CELLS', 1):
                    with pytest.raises(MemoryError, match='Bounded parallel'):
                        dp.enumerate_parallel(*args)


@torch.no_grad()
def test_batch_keeps_multiquery_path(system):
    import copy
    from dataclasses import replace
    model, paths, db, tree, _ = system
    batch_model = copy.copy(model)
    batch_model.config = replace(model.config, parameter_mode='manual',
                                 query_profile='throughput', query_batch=512)
    q = batch_model.encode(paths[:3])
    with patch.object(opt, 'dtw_subtree_parallel', False), patch.object(opt, 'dtw_single_layout', False):
        expected = batch_model.search_range(q, db, search_index=tree)
    with patch.object(layout, 'enumerate_members', side_effect=AssertionError('Single layout on batch')):
        actual = batch_model.search_range(q, db, search_index=tree)
    for name in ('indices', 'distances', 'counts', 'radius'):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
