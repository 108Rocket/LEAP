# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Batched DTW frontier execution preserves original-distance results.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Migration gates for the accepted Batch-only DTW block frontier."""
from copy import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from leap import RetrievalConfig
from leap import LEAP
from leap.dtw import dtw_bounded
from leap.dtw.dtw import dtw_pairs
from leap.common.runtime import options


class KernelSpy:
    def __init__(self, kernel):
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches += 1
            return self.kernel[grid](*args, **kwargs)
        return launch


@pytest.mark.parametrize('rows,dtype,bucket,check,has_limit,has_plan,enabled,expected', [
    (2, torch.float32, True, 8, True, False, True, True),
    (1, torch.float32, True, 8, True, False, True, False),
    (2, torch.float64, True, 8, True, False, True, False),
    (2, torch.float32, False, 8, True, False, True, False),
    (2, torch.float32, True, 1, True, False, True, False),
    (2, torch.float32, True, 8, False, False, True, False),
    (2, torch.float32, True, 8, True, True, True, False),
    (2, torch.float32, True, 8, True, False, False, False),
])
def test_batch_dispatch_scope(rows, dtype, bucket, check, has_limit, has_plan, enabled, expected):
    q = SimpleNamespace(points=torch.empty(0, dtype=dtype), offsets=torch.arange(rows+1))
    with patch.object(options, 'dtw_batch_frontier', enabled):
        assert dtw_bounded._use_batch_frontier(q, object() if has_limit else None,
            bucket=bucket, check=check, plan=object() if has_plan else None) == expected
    with patch.object(torch.version, 'hip', 'diagnostic-hip'):
        assert not dtw_bounded._use_batch_frontier(q, object(), bucket=True, check=8, plan=None)


@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA retrieval test')
def test_threshold_values_and_written_banks():
    rng = np.random.default_rng(13300)
    lengths = [1, 2, 31, 63, 64, 65, 127, 173, 524]
    paths = [rng.normal(size=(n, 2)).cumsum(0).astype(np.float32)*.1 for n in lengths]
    paths += [np.zeros((64, 2), np.float32), np.zeros((65, 2), np.float32)]
    data = SimpleNamespace(points=torch.tensor(np.concatenate(paths), device='cuda'),
        offsets=torch.tensor(np.r_[0, np.cumsum([len(p) for p in paths])], device='cuda'))
    ids = torch.arange(len(paths), device='cuda')
    qr, dr = torch.cartesian_prod(ids, ids).unbind(1)
    reference = dtw_pairs(data, data, qr, dr)
    budgets = [reference, torch.nextafter(reference, torch.full_like(reference, -torch.inf)),
        torch.nextafter(reference, torch.full_like(reference, torch.inf)), reference*.6,
        torch.zeros_like(reference), torch.full_like(reference, torch.inf)]
    for limits in budgets:
        with patch.object(options, 'dtw_batch_frontier', True):
            observed, steps = dtw_bounded.pairs(data, data, qr, dr, limits, bucket=True, record=True)
        with patch.object(options, 'dtw_batch_frontier', False):
            old = dtw_bounded.pairs(data, data, qr, dr, limits, bucket=True)
        valid = reference <= limits
        assert torch.equal(observed <= limits, valid)
        assert torch.equal(old <= limits, valid)
        torch.testing.assert_close(observed[valid], reference[valid], rtol=0, atol=0)
        total = data.offsets[qr+1]-data.offsets[qr]+data.offsets[dr+1]-data.offsets[dr]-1
        assert bool(((steps > 0) & (steps <= total)).all())
    assert dtw_bounded.pairs(data, data, qr[:0], dr[:0], reference[:0], bucket=True).numel() == 0


@pytest.mark.parametrize('top_k', [1, 20, 64])
@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA retrieval test')
def test_public_full_and_native_single(top_k):
    from leap.dtw import dtw_frontier
    from leap.common.single_kernels import pair_bucket
    assert pair_bucket.fn.__globals__['bounded_kernel'] is dtw_bounded.bounded_kernel
    model = LEAP.from_asset(config=RetrievalConfig('dtw', 'signal', dtw_routing='learned'))
    rng = np.random.default_rng(6402)
    paths = [model.mean + rng.normal(size=(65+i%33, 2)).cumsum(0)*.0001 for i in range(150)]
    paths[2] = paths[1].copy()
    d, q = model.encode(paths), model.encode(paths[:2])
    tree = model.build_search_index(d)
    with patch.object(options, 'dtw_batch_frontier', False):
        expected = model.search_range(q, d, search_index=tree, top_k=top_k)
    kernel = KernelSpy(dtw_frontier.batch_frontier_kernel)
    with patch.object(options, 'dtw_batch_frontier', True), patch.object(
            dtw_frontier, 'batch_frontier_kernel', kernel):
        observed = model.search_range(q, d, search_index=tree, top_k=top_k)
        if top_k >= 20:
            assert kernel.launches > 0
    for key in ('indices', 'distances', 'counts', 'radius'):
        torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0)
    single = copy(model)
    single.config = replace(model.config, query_profile='single_query', query_batch=None)
    session = single.query_session(d, search_index=tree, top_k=top_k, initial_capacity=128)
    kernel = KernelSpy(dtw_frontier.batch_frontier_kernel)
    with patch.object(options, 'dtw_batch_frontier', True), patch.object(
            dtw_frontier, 'batch_frontier_kernel', kernel):
        session.warmup(paths[:2], modes=('full',))
        for i, p in enumerate(paths[:2]):
            np.testing.assert_array_equal(session.search(p, mode='full')[0], expected['indices'][i].cpu().numpy())
        assert kernel.launches == 0
