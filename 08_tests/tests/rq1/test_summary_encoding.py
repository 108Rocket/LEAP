# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ3: Contributions of C1--C3.
# Checks: DFD/DTW optimized encoders preserve reference geometric summaries.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Default large-input encoders preserve the reference summaries exactly."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from leap.dfd import dfd_geometry
from leap.dtw import dtw_probe_intervals
from leap.dtw import dtw_radial
from leap.dtw import dtw_routing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def geometry(rows, seed=71, dtype=torch.float32, device='cuda'):
    generator = torch.Generator(device=device).manual_seed(seed)
    lengths = torch.tensor([2, 3, 7, 16, 17, 31, 32, 33, 63, 64, 65], device=device)
    lengths = lengths.repeat((rows + len(lengths) - 1) // len(lengths))[:rows]
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    points = torch.randn((int(offsets[-1]), 2), generator=generator,
                         device=device, dtype=dtype)
    points[:int(offsets[3])] = 0
    return SimpleNamespace(points=points, offsets=offsets,
                           coordinates=torch.empty((rows, 2), device=device))


@pytest.mark.parametrize('rows', [4095, 4096, 4107])
def test_summary_default_matches_reference(rows):
    index = geometry(rows)
    theta = torch.tensor([[0., 0.], [1., -1.], [-2., 3.]], device='cuda')
    actual = dfd_geometry.prepare(index)
    expected = dfd_geometry._prepare_reference(index)
    assert actual['block'] == expected['block']
    for key in expected['ids']:
        assert torch.equal(actual['ids'][key], expected['ids'][key])
    for module in (dtw_radial, dtw_probe_intervals):
        assert torch.equal(module.encode(index, theta), module._encode_reference(index, theta))


@pytest.mark.parametrize('seed', [2000, 2001, 2002])
def test_routing_fusion_nonzero_weights(seed):
    torch.manual_seed(seed)
    model = dtw_routing.OrderedMass().cuda().eval()
    with torch.no_grad():
        model.net[-1].weight.normal_()
        model.net[-1].bias.normal_()
    index = geometry(529, seed)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        assert torch.equal(dtw_routing.encode_index(model, index),
                           dtw_routing._encode_reference(model, index))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_small_and_fp64_fallback():
    from leap.dtw import _routing_workspace
    from leap.dtw import _summary_encoding
    for rows, dtype in ((7, torch.float32), (4096, torch.float64)):
        index = geometry(rows, dtype=dtype)
        theta = index.points[:3]
        with patch.object(_summary_encoding, 'dfd_summary', side_effect=AssertionError('fast path')):
            dfd_geometry.prepare(index)
        for module, name in ((dtw_radial, 'radial'), (dtw_probe_intervals, 'ordered')):
            with patch.object(_summary_encoding, name, side_effect=AssertionError('fast path')):
                module.encode(index, theta)
    model = dtw_routing.OrderedMass().cuda().eval()
    with patch.object(_routing_workspace, 'encode', side_effect=AssertionError('fast path')):
        dtw_routing.encode_index(model, geometry(7))
        dtw_routing.encode_index(model, geometry(529), batch=128)
    cpu_model = dtw_routing.OrderedMass().eval()
    with patch.object(_routing_workspace, 'encode', side_effect=AssertionError('fast path')):
        dtw_routing.encode_index(cpu_model, geometry(513, device='cpu'))


@pytest.mark.parametrize('width', [2, 7, 31, 32, 63, 64, 65, 127, 128, 129, 512, 513])
def test_routing_first_layer_shape_boundaries(width):
    from leap.dtw import _routing_encoding as fused
    import triton
    torch.manual_seed(350 + width)
    model = dtw_routing.OrderedMass().cuda().eval()
    features = torch.randn((512, width, 6), device='cuda')
    lengths = torch.full((512,), width, dtype=torch.int64, device='cuda')
    hidden = torch.empty((512, width, 16), device='cuda')
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if width >= 64:
            fused.first_layer[(512, triton.cdiv(width, 64))](features, lengths,
                model.net[0].weight, model.net[0].bias, torch.tanh(model.net[0].bias),
                hidden, width, 64, num_warps=4, enable_fp_fusion=False)
            assert torch.equal(hidden, model.net[:2](features))
        else:
            index = SimpleNamespace(points=torch.randn((513*width,2),device='cuda'),
                                    offsets=torch.arange(514,device='cuda')*width)
            assert torch.equal(dtw_routing.encode_index(model,index),
                               dtw_routing._encode_reference(model,index))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
