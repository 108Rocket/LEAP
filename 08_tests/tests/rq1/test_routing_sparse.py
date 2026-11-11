# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ3: Contributions of C1--C3.
# Checks: Sparse DTW summary routing preserves native arithmetic and outputs.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Native-order routing, narrow dispatch, mutations and owned storage."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from leap.dtw import dtw_routing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(autouse=True)
def exact_math():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def make_index(width, count=1037):
    lengths = torch.arange(count, device='cuda') % min(width - 1, 65) + 2
    lengths[min(500, count - 1)] = width
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    points = torch.randn((int(offsets[-1]), 2), device='cuda')
    points[:20] = 0
    return SimpleNamespace(points=points, offsets=offsets)


@pytest.mark.parametrize('width,count', [(132, 1037), (256, 1024), (1024, 1037),
                                      (3836, 1037), (4096, 1037), (4097, 1037),
                                      (129, 1037), (64, 1037), (132, 17)])
def test_shapes(width, count):
    model = dtw_routing.OrderedMass().cuda().eval()
    with torch.no_grad():
        model.net[-1].weight.normal_()
        model.net[-1].bias.fill_(4.)
    index = make_index(width, count)
    assert torch.equal(dtw_routing.encode_index(model, index),
                       dtw_routing._encode_reference(model, index))


@pytest.mark.parametrize('seed', [2000, 2001, 2002])
def test_input_weight_mutations(seed):
    torch.manual_seed(seed)
    model = dtw_routing.OrderedMass().cuda().eval()
    index = make_index(3836)
    with torch.no_grad():
        model.net[-1].weight.normal_()
    first = dtw_routing.encode_index(model, index)
    saved = first.clone()
    assert torch.equal(first, dtw_routing._encode_reference(model, index))
    with torch.no_grad():
        model.net[-1].weight.add_(.2)
        index.points[40:50].add_(.1)
    second = dtw_routing.encode_index(model, index)
    assert torch.equal(second, dtw_routing._encode_reference(model, index))
    assert torch.equal(first, saved)
    assert first.data_ptr() != second.data_ptr()


def test_backend_and_tf32_fallback():
    from leap.dtw import _routing_sparse
    from leap.dtw import _routing_workspace
    model = dtw_routing.OrderedMass().cuda().eval()
    index = make_index(132)
    with patch.object(_routing_sparse, 'supported_backend', return_value=False), \
            patch.object(_routing_workspace, 'encode', wraps=_routing_workspace.encode) as spy:
        got = dtw_routing.encode_index(model, index)
        assert spy.call_count == 1
        assert torch.equal(got, dtw_routing._encode_reference(model, index))
    torch.backends.cuda.matmul.allow_tf32 = True
    with patch.object(_routing_sparse, 'encode', side_effect=AssertionError('unexpected fast path')):
        dtw_routing.encode_index(model, index)


def test_dispatch_enters_input_device():
    from leap.dtw import _routing_sparse
    index = make_index(132)
    model = dtw_routing.OrderedMass().cuda().eval()
    seen = []
    class ObservedDevice(torch.cuda.device):
        def __init__(self, value):
            seen.append(value)
            super().__init__(value)
    with patch.object(torch.cuda, 'device', ObservedDevice), \
            patch.object(_routing_sparse, 'encode', return_value='selected'):
        assert dtw_routing.encode_index(model, index) == 'selected'
        assert seen == [index.points.device]


@pytest.mark.parametrize('width', [132, 256, 512, 1024, 2048, 3836, 4096])
def test_native_reduction_trees(width):
    from leap.dtw._routing_transport import sum_native, scan_native
    for local in sorted({64, min(width, 128), min(width, 256), width}):
        for wide in (False, True):
            x = torch.rand((512, width), device='cuda')
            if wide:
                x *= 10 ** torch.randint(-4, 5, x.shape, device='cuda')
            x[:, local:] = 0
            compact = x[:, :local].contiguous()
            total = torch.empty((512, 1), device='cuda')
            scan = torch.empty_like(compact)
            chunk = 2 << max(4, min(9, (width - 1).bit_length() // 2))
            sum_native[(512,)](compact, total, local, width, num_warps=1, enable_fp_fusion=False)
            scan_native[(512,)](compact, scan, local, chunk, num_warps=4, enable_fp_fusion=False)
            assert torch.equal(total, x.sum(1, keepdim=True))
            assert torch.equal(scan, x.cumsum(1)[:, :local])
