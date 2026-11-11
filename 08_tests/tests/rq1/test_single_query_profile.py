# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance.
# Checks: Single-request profiles preserve normalization and search configuration.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from leap import LEAP
from leap import RetrievalConfig
from leap.common.raw import normalize_points, normalize_points_serial


@pytest.mark.parametrize('metric', ['hausdorff', 'dfd', 'dtw'])
def test_config_preserves_algorithm(metric):
    root = Path(__file__).parents[3] / '02_configs' / metric
    base = RetrievalConfig.from_json(root / (metric + '.json'))
    single = RetrievalConfig.from_json(root / ('single_query_' + metric + '.json'))
    assert single == replace(base, query_profile='single_query')
    assert RetrievalConfig().query_profile == 'throughput'


@pytest.mark.parametrize('invalid', [None, True, 1, 'fast'])
def test_invalid_profile(invalid):
    with pytest.raises(ValueError, match='query_profile'):
        RetrievalConfig(query_profile=invalid)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('length', [0, 1, 17, 157, 4097])
def test_normalization_bitwise(dtype, length):
    rng = np.random.default_rng(52)
    mean = np.array([116.409123456, 39.90123456], dtype=np.float64)
    std = np.array([0.00731, 0.039123], dtype=np.float64)
    raw = (mean + rng.normal(size=(length, 4))[:, :2] * std).astype(dtype)
    for points in (raw, raw[::-1]):
        a = normalize_points(points, mean, std)
        b = normalize_points_serial(points, mean, std)
        assert a.dtype == b.dtype == np.float32
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_dispatch_and_batch_fallback():
    rng = np.random.default_rng(3)
    model = LEAP(rng.normal(size=(64, 2)), [0., 0.], [1., 1.],
                   config=RetrievalConfig(query_profile='single_query'))
    rows = [rng.normal(size=(17, 2)), rng.normal(size=(23, 2))]
    from leap.common.encoding import _normalize_serial, _normalize_parallel
    with patch('leap.common.encoding._normalize_serial', wraps=_normalize_serial) as serial:
        with patch('leap.common.encoding._normalize_parallel', wraps=_normalize_parallel) as parallel:
            one = model.encode(rows[:1])
            batch = model.encode(rows)
            assert serial.call_count == 2
            assert parallel.call_count == 0
    model.config = replace(model.config, query_profile='throughput')
    old = model.encode(rows[:1])
    assert torch.equal(old.points, one.points)
    assert torch.equal(old.coordinates, one.coordinates)
    assert torch.equal(batch.coordinates[:1], one.coordinates)
