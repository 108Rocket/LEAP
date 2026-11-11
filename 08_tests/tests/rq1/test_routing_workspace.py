# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance; RQ3: Contributions of C1--C3.
# Checks: Compact DTW routing workspace preserves summaries and ownership.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Compact routing preserves native arithmetic, dispatch and output ownership."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from leap.dtw import dtw_routing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("seed", [2000, 2001, 2002])
def test_mixed_widths_tail_and_mutations(seed):
    torch.manual_seed(seed)
    model = dtw_routing.OrderedMass().cuda().eval()
    with torch.no_grad():
        model.net[-1].weight.normal_()
        model.net[-1].bias.normal_()
    lengths = torch.arange(1029, device="cuda") % 61 + 2
    lengths[512:1024] += 200
    lengths[1024] = 4097
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    points = torch.randn((int(offsets[-1]), 2), device="cuda")
    points[:30] = 0
    index = SimpleNamespace(points=points, offsets=offsets)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        first = dtw_routing.encode_index(model, index)
        assert torch.equal(first, dtw_routing._encode_reference(model, index))
        saved = first.clone()
        with torch.no_grad():
            model.net[-1].bias.add_(0.1)
            points[40:50].add_(0.2)
        second = dtw_routing.encode_index(model, index)
        assert torch.equal(second, dtw_routing._encode_reference(model, index))
        assert first.data_ptr() != second.data_ptr()
        assert torch.equal(first, saved)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_tf32_and_custom_model_do_not_enter_workspace():
    from leap.dtw import _routing_workspace

    class CustomMass(dtw_routing.OrderedMass):
        pass

    index = SimpleNamespace(points=torch.randn((1026, 2), device="cuda"),
                            offsets=torch.arange(514, device="cuda") * 2)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        with patch.object(_routing_workspace, "encode",
                          side_effect=AssertionError("unexpected workspace")):
            torch.backends.cuda.matmul.allow_tf32 = True
            dtw_routing.encode_index(dtw_routing.OrderedMass().cuda().eval(), index)
            torch.backends.cuda.matmul.allow_tf32 = False
            dtw_routing.encode_index(CustomMass().cuda().eval(), index)
            double_index = SimpleNamespace(points=index.points.double(), offsets=index.offsets)
            dtw_routing.encode_index(dtw_routing.OrderedMass().cuda().double().eval(), double_index)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
