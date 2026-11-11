# Test group: RQ2.
# Paper RQ: RQ2: Role of Full Search.
# Checks: Coordinate normalization preserves precision before floating-point casting.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Regression tests for normalization before lossy absolute-coordinate casting."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from leap.common.data import checked_pack, NORMALIZATION_VERSION


def test_pack_preserves_sub_float32_spacing_and_inputs():
    mean = np.array([-959467.8236869909, 5035825.64370849])
    path = mean + np.array([[.001, .003], [.002, .004], [.012, .009]])
    before = path.copy()
    path.flags.writeable = False
    points, offsets = checked_pack(p for p in [path])
    np.testing.assert_array_equal(points, before)
    np.testing.assert_array_equal(path, before)
    assert offsets.tolist() == [0, 3]
    assert not np.array_equal(points.astype(np.float32).astype(np.float64), before)
    points[0] += 1
    np.testing.assert_array_equal(path, before)


@pytest.mark.parametrize('paths', [[], [np.zeros((1, 2))], [np.zeros((2, 3))],
                                  [np.full((2, 2), np.nan)], [np.full((2, 2), np.inf)]])
def test_pack_rejects_invalid_geometry(paths):
    with pytest.raises(ValueError):
        checked_pack(paths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA retrieval test')
def test_encoder_preserves_precision_and_rejects_old_index():
    from leap import LEAP
    model = LEAP.from_asset()
    mean, std = model.mean, model.std
    paths = [mean + np.array([[.001, .003], [.002, .004], [.012, .009]]) + i*.01
             for i in range(4)]
    index = model.encode(paths)
    expected = ((np.concatenate(paths) - mean) / std).astype(np.float32)
    np.testing.assert_array_equal(index.points.cpu().numpy(), expected)
    assert index.points.dtype == torch.float32
    suffix = ':' + NORMALIZATION_VERSION
    assert index.asset_id.endswith(suffix)
    stale = replace(index, asset_id=index.asset_id[:-len(suffix)])
    with pytest.raises(ValueError, match='re-encode and rebuild'):
        model._check_index(stale)
