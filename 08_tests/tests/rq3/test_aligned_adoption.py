# Test group: RQ3.
# Paper RQ: RQ3: Contributions of C1--C3.
# Checks: DTW learning derivatives, matched assets and alignment controls.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Matched assets, implicit selection derivatives and buffer lifetime guards."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from leap import RetrievalConfig
from leap import LEAP
from leap.dtw.dtw_aligned_loss import first_selection, conditional_selection, tail_value, aligned_radius_loss
from leap.dtw.dtw_routing import load_model
from leap.training.finetune_dtw_routing import validate_pool
from leap.common.runtime.plan import SingleQueryPlan

ROOT = Path(__file__).parents[3]


def test_selected_bundle():
    signal = json.loads((ROOT / '01_LEAP/leap/assets/sdd_porto_dtw_signal16.json').read_text())
    router = ROOT / '01_LEAP/leap/assets' / signal['default_routing_asset']
    _, weights = load_model(signal['geometry_mean'], signal['geometry_std'], 'cpu', router)
    assert weights == 'e195dfd4e2afe3eb8e0590e1e793fd2fc37b3bda9ddb970cdc164a07633dfe0f'
    with pytest.raises(ValueError, match='normalization'):
        load_model([0., 0.], [1., 1.], 'cpu', router)


def test_implicit_derivatives():
    generator = torch.Generator().manual_seed(1729)
    scores = (torch.rand((2, 276), generator=generator, dtype=torch.float64) * 2).requires_grad_()
    valid = torch.ones_like(scores, dtype=torch.bool)
    valid[:, -20:] = False
    signal = torch.rand(scores.shape, generator=generator, dtype=torch.float64) * 2
    distances = torch.rand(scores.shape, generator=generator, dtype=torch.float64) + 1
    p = first_selection(scores, valid)
    a = conditional_selection(p, signal)
    torch.testing.assert_close(p.sum(1), torch.full((2,), 128., dtype=torch.float64))
    torch.testing.assert_close(a.sum(1), torch.full((2,), 20., dtype=torch.float64))
    loss = tail_value(a, distances).mean()
    gradient, = torch.autograd.grad(loss, scores)
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient[:, -20:], torch.zeros_like(gradient[:, -20:]))
    for row, col in ((0, 24), (1, 65), (1, 91)):
        delta = 1e-5
        low, high = scores.detach().clone(), scores.detach().clone()
        low[row, col] -= delta
        high[row, col] += delta
        def value(s):
            return tail_value(conditional_selection(first_selection(s, valid), signal), distances).mean()
        numerical = (value(high) - value(low)) / (2 * delta)
        torch.testing.assert_close(gradient[row, col], numerical, atol=2e-7, rtol=2e-4)
    with pytest.raises(ValueError, match='frozen'):
        conditional_selection(p, signal.requires_grad_())


def test_aligned_loss_and_pool_guards():
    scores = torch.linspace(.1, 2., 276)[None].requires_grad_()
    signal = torch.linspace(2., .1, 276)[None]
    distances = torch.linspace(1., 3., 276)[None]
    valid = torch.ones((1, 276), dtype=torch.bool)
    loss = aligned_radius_loss(scores, signal, distances, valid, torch.tensor([2.]))
    assert torch.isfinite(loss)
    with pytest.raises(ValueError):
        first_selection(scores, valid, k=276)
    n = 300
    pool = np.array([(np.arange(1, 277) + row) % n for row in range(n)])
    mask = np.ones_like(pool, dtype=bool)
    validate_pool(pool, mask, pool[:, :20], n)
    bad = pool.copy()
    bad[:, -1] = bad[:, -2]
    with pytest.raises(ValueError, match='distinct'):
        validate_pool(bad, mask, pool[:, :20], n)


def test_growth_invalidates_graphs_only_when_storage_moves():
    plan = SimpleNamespace(device='cpu', buffers={}, graphs={}, stats={'buffer_growths': 0})
    first = SingleQueryPlan.storage(plan, 'x', (8,))
    plan.graphs['captured'] = object()
    same = SingleQueryPlan.storage(plan, 'x', (4,))
    assert first.data_ptr() == same.data_ptr() and len(plan.graphs) == 1
    bigger = SingleQueryPlan.storage(plan, 'x', (32,))
    assert bigger.data_ptr() != first.data_ptr() and not plan.graphs


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA integration')
def test_matched_asset_auto_binding_and_reject_paths(tmp_path):
    config = RetrievalConfig.from_json(ROOT / '02_configs/dtw/sdd_porto_dtw_aligned.json')
    asset = ROOT / '01_LEAP/leap/assets/sdd_porto_dtw_signal16.json'
    model = LEAP.from_asset(asset, config=config)
    assert ':dtw-routing:e195dfd4' in model.asset_id
    data = json.loads(asset.read_text())
    data['default_routing_asset'] = '../router.json'
    invalid = tmp_path / 'invalid.json'
    invalid.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='routing asset'):
        LEAP.from_asset(invalid, config=config)
