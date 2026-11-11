# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ3: Contributions of C1--C3.
# Checks: DTW query summaries and routing preserve values and fallback gradients.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""DTW query dispatch preserves values, gradients, fallbacks and ownership."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from leap.dtw import dtw_routing
from leap.dtw import dtw_signal
from leap.common import _query_encoding

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(autouse=True)
def exact_math():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def make_index(count, width, dtype=torch.float32):
    lengths = torch.arange(count, device='cuda') % min(width - 1, 53) + 2
    lengths[-1] = width
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    points = torch.randn((int(offsets[-1]), 2), device='cuda', dtype=dtype)
    points[:24] = 0
    return SimpleNamespace(points=points, offsets=offsets)


@pytest.mark.parametrize('count,width', [(1, 132), (63, 65), (64, 2), (64, 65),
                                      (65, 129), (128, 64), (500, 3836), (512, 132),
                                      (512, 4097), (513, 132)])
@pytest.mark.parametrize('seed', [2000, 2001])
def test_routing_exact(count, width, seed):
    torch.manual_seed(seed)
    model = dtw_routing.OrderedMass().cuda().eval()
    with torch.no_grad():
        model.net[-1].weight.normal_()
        model.net[-1].bias.fill_(3.)
    index = make_index(count, width)
    got = dtw_routing.encode_index(model, index)
    assert torch.equal(got, dtw_routing._encode_reference(model, index))


@pytest.mark.parametrize('count', [1, 63, 64, 500, 512, 513])
@pytest.mark.parametrize('probes', [1, 4, 16, 64])
def test_signals_exact(count, probes):
    index = make_index(count, 132)
    theta = torch.randn((probes, 2), device='cuda')
    with torch.no_grad():
        got = dtw_signal.encode(index, theta)
        expected = (index.points[:, None] - theta[None]).square().sum(-1).sqrt()
    assert torch.equal(got, expected)


def test_mutations_and_storage():
    index = make_index(64, 128)
    model = dtw_routing.OrderedMass().cuda().eval()
    theta = torch.randn((16, 2), device='cuda')
    with torch.no_grad():
        first = dtw_routing.encode_index(model, index)
        signals = dtw_signal.encode(index, theta)
        saved, saved_signals = first.clone(), signals.clone()
        model.net[-1].weight.normal_()
        index.points.add_(.1)
        theta.add_(.2)
        second = dtw_routing.encode_index(model, index)
        second_signals = dtw_signal.encode(index, theta)
        assert torch.equal(second, dtw_routing._encode_reference(model, index))
        assert torch.equal(second_signals, (index.points[:, None] - theta[None]).square().sum(-1).sqrt())
        assert torch.equal(first, saved) and torch.equal(signals, saved_signals)
        assert first.data_ptr() != second.data_ptr()
        assert signals.data_ptr() != second_signals.data_ptr()


def test_training_gradient_fallback():
    index = make_index(64, 65)
    index.points.requires_grad_()
    theta = torch.randn((4, 2), device='cuda', requires_grad=True)
    with patch.object(_query_encoding, 'signals', side_effect=AssertionError('inference only')):
        got = dtw_signal.encode(index, theta)
    expected = (index.points[:, None] - theta[None]).square().sum(-1).sqrt()
    grads = torch.autograd.grad(got.sum(), (index.points, theta))
    reference = torch.autograd.grad(expected.sum(), (index.points, theta))
    assert all(torch.equal(a, b) for a, b in zip(grads, reference))


@pytest.mark.parametrize('fallback', ['fp64', 'tf32', 'custom_batch', 'backend', 'noncontiguous'])
def test_routing_fallback(fallback):
    model = dtw_routing.OrderedMass().cuda().eval()
    index = make_index(64, 128)
    batch = 512
    if fallback == 'fp64':
        model.double(); index.points = index.points.double()
    elif fallback == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
    elif fallback == 'custom_batch':
        batch = 32
    elif fallback == 'noncontiguous':
        index.points = index.points.t().contiguous().t()
    with patch.object(_query_encoding, 'routing', side_effect=AssertionError('unexpected fast path')):
        if fallback == 'backend':
            with patch.object(_query_encoding, 'supported', return_value=False):
                got = dtw_routing.encode_index(model, index, batch)
        else:
            got = dtw_routing.encode_index(model, index, batch)
    assert torch.equal(got, dtw_routing._encode_reference(model, index, batch))
