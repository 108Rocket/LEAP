# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance.
# Checks: Single-request scheduling, length buckets and buffer reuse.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Host scheduling contracts, independent of the unchanged distance kernels."""

from types import SimpleNamespace

import pytest
import torch

from leap.common.runtime import plan as module


@pytest.mark.parametrize('block', [1, 32, 64, 128, 256, 512, 4096])
def test_length_buckets_match_existing_dispatch(block):
    widths = [2**i for i in range(6, max(64, block).bit_length())]
    expected = tuple(zip([0] + widths[:-1], widths))
    assert module._length_buckets(block) == expected
    assert module.SingleQueryPlan.widths(block) == widths
    assert module._length_buckets(block) is module._length_buckets(block)


@pytest.mark.parametrize('block', [64, 256, 4096])
@pytest.mark.parametrize('count', [1, 2730, 100000])
def test_workspace_budget_is_unchanged(block, count):
    expected = max(1, min(count, (32 * 1024 * 1024) // (3 * block)))
    assert module._workspace_chunk(count, block) == expected


def bare_plan():
    plan = module.SingleQueryPlan.__new__(module.SingleQueryPlan)
    plan.device = torch.device('cuda')
    plan.database_block = 256
    plan.query_length = 17
    plan.buffers = {}
    plan.stats = dict(buffer_growths=0)
    return plan


class RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


@pytest.mark.parametrize('kind', [0, 1, 2, 3])
@pytest.mark.parametrize('count', [0, 1, 5])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA buffer test')
def test_pair_views_reused_with_exact_chunk_and_limit_semantics(monkeypatch, kind, count):
    plan = bare_plan()
    kernel = RecordingKernel()
    monkeypatch.setattr(module, 'pair_bucket', kernel)
    monkeypatch.setattr(module, 'DP_WORK_ELEMENTS', 2 * 3 * 256)
    index = SimpleNamespace(
        points=torch.zeros((2, 2), device='cuda'), offsets=torch.tensor([0, 2], device='cuda')
    )
    qids = torch.zeros(count, device='cuda', dtype=torch.int64)
    dids = torch.arange(count, device='cuda')
    limits = torch.arange(1 if kind == 3 else count, device='cuda', dtype=torch.float32)
    initial = torch.zeros((1, 20), device='cuda')
    raw = torch.ones((1, 20), device='cuda')
    result = plan.pair_values(index, index, qids, dids, kind, limits, initial, raw)
    out = result[0] if kind == 3 else result
    assert out.shape == (count,)
    if kind == 3:
        assert result[1].shape == (count,)
    assert len(kernel.calls) == ((count + 1) // 2) * 3
    for offset in range(0, len(kernel.calls), 3):
        chunk_calls = kernel.calls[offset : offset + 3]
        first_args = chunk_calls[0][1]
        start = (offset // 3) * 2
        end = min(count, start + 2)
        for (lower, width), (grid, args, kwargs) in zip(module._length_buckets(256), chunk_calls):
            assert grid == (end - start,)
            for position in (4, 5, 6, 7, 9):
                assert args[position] is first_args[position]
            torch.testing.assert_close(args[5], dids[start:end], rtol=0, atol=0)
            assert args[10] is initial and args[11] is raw
            assert args[12:17] == (width, lower, kind, 20, 8)
            assert kwargs == dict(num_warps=4, enable_fp_fusion=False)
            if kind == 3:
                assert args[9] is limits
            else:
                torch.testing.assert_close(args[9], limits[start:end], rtol=0, atol=0)


@pytest.mark.parametrize('kind', [0, 1, 2, 3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA buffer test')
def test_optional_pair_buffers_keep_native_abi(monkeypatch, kind):
    plan = bare_plan()
    kernel = RecordingKernel()
    monkeypatch.setattr(module, 'pair_bucket', kernel)
    index = SimpleNamespace(
        points=torch.zeros((2, 2), device='cuda'), offsets=torch.tensor([0, 2], device='cuda')
    )
    ids = torch.zeros(1, device='cuda', dtype=torch.int64)
    result = plan.pair_values(index, index, ids, ids, kind)
    out = result[0] if kind == 3 else result
    for _, args, _ in kernel.calls:
        assert args[9] is (None if kind == 3 else out)
        assert args[10] is out and args[11] is out
        assert args[15] == 0


@pytest.mark.parametrize('count', [0, 1, 5])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA buffer test')
def test_signal_views_reused_per_chunk(monkeypatch, count):
    plan = bare_plan()
    kernel = RecordingKernel()
    monkeypatch.setattr(module, 'signal_bucket', kernel)
    monkeypatch.setattr(module, 'DP_WORK_ELEMENTS', 2 * 3 * 256)
    index = SimpleNamespace(offsets=torch.tensor([0, 2], device='cuda'))
    signal = torch.zeros((2, 16), device='cuda')
    ids = torch.zeros(count, device='cuda', dtype=torch.int64)
    out = plan.signal_values(index, index, ids, ids, (signal, signal))
    assert out.shape == (count,)
    assert len(kernel.calls) == ((count + 1) // 2) * 3
    for offset in range(0, len(kernel.calls), 3):
        first_args = kernel.calls[offset][1]
        for _, args, _ in kernel.calls[offset : offset + 3]:
            for position in (4, 5, 6):
                assert args[position] is first_args[position]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA buffer test')
def test_storage_grows_and_separates_dtype():
    plan = bare_plan()
    first = plan.storage('test', (3,))
    first.fill_(7)
    smaller = plan.storage('test', (2,))
    assert smaller.data_ptr() == first.data_ptr()
    torch.testing.assert_close(smaller, torch.full_like(smaller, 7))
    integer = plan.storage('test', (2,), torch.int32)
    assert integer.dtype == torch.int32 and integer.data_ptr() != first.data_ptr()
    grown = plan.storage('test', (9,))
    assert grown.data_ptr() != first.data_ptr()
    assert plan.buffers[('test', torch.float32)].numel() == 16
    assert plan.stats['buffer_growths'] == 3
