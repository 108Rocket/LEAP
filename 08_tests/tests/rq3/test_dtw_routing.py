# Test group: RQ3.
# Paper RQ: RQ1: Overall Performance; RQ3: Contributions of C1--C3.
# Checks: Learned DTW candidate routing and its interface to exact refinement.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Routing invariants, checksum, API isolation and unchanged exact refinement."""
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from leap import RetrievalConfig
from leap import LEAP
from leap.dtw.dtw_routing import OrderedMass, prepare, encode_index, load_model, DEFAULT_ASSET


def test_routing_config():
    assert RetrievalConfig.from_json(Path(__file__).parents[3] / '02_configs/dtw/dtw.json').dtw_routing == 'learned'
    assert RetrievalConfig('dtw', 'signal').dtw_routing == 'fixed'
    for kw in [dict(distance='dfd'), dict(distance='dtw', dtw_seed='weighted')]:
        with pytest.raises(ValueError):
            RetrievalConfig(**kw, dtw_routing='learned')


def test_asset_checksum():
    asset = json.loads(DEFAULT_ASSET.read_text())
    assert asset['epoch'] == 100
    net, _ = load_model(asset['geometry_mean'], asset['geometry_std'], 'cpu')
    assert sum(x.numel() for x in net.parameters()) == 129
    bad = copy.deepcopy(asset)
    bad['state_dict']['net.2.bias'][0] += 1
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'bad.json'
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='checksum'):
            load_model(asset['geometry_mean'], asset['geometry_std'], 'cpu', path)


def test_mass_and_rigid_motion():
    asset = json.loads(DEFAULT_ASSET.read_text())
    net, _ = load_model(asset['geometry_mean'], asset['geometry_std'], 'cpu')
    points = torch.tensor([[0., 0.], [1., 2.], [2., 1.], [2., 1.], [4., 0.], [5., 0.]])
    index = SimpleNamespace(points=points, offsets=torch.tensor([0, 4, 6]))
    data = prepare(index)
    summary, mass, overlap = net(*data, details=True)
    torch.testing.assert_close(overlap.sum(1), torch.full((2, 8), .125))
    assert bool((overlap >= 0).all() & torch.isfinite(summary).all())
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    moved = SimpleNamespace(points=points @ rotation + 3., offsets=index.offsets)
    torch.testing.assert_close(encode_index(net, moved).reshape(2, 8, 2),
                               summary.reshape(2, 8, 2) @ rotation + 3.)
    torch.testing.assert_close(encode_index(net, index, batch=1), summary)
    zeros = SimpleNamespace(points=torch.zeros(4, 2), offsets=torch.tensor([0, 2, 4]))
    assert bool(torch.isfinite(encode_index(net, zeros)).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA integration')
def test_learned_public_api():
    config = RetrievalConfig.from_json(Path(__file__).parents[3] / '02_configs/dtw/dtw.json')
    model = LEAP.from_asset(config=config)
    rng = np.random.default_rng(400)
    paths = [rng.normal(size=(3 + i % 20, 2)).cumsum(0) * model.std + model.mean for i in range(150)]
    db = model.encode(paths)
    tree = model.build_search_index(db)
    with patch('leap.dtw.dtw.dtw_pairs', side_effect=AssertionError('raw DTW in coarse')):
        ids = model.candidate_indices(db, candidate_k=20)
    assert not bool((ids == torch.arange(150, device='cuda')[:, None]).any())
    from leap.dtw.dtw import dtw_pairs
    from leap.common.interval_index import lex_keys
    q = model.encode(paths[:7])
    rows = torch.arange(7, device='cuda')[:, None].expand(7, 150)
    cols = torch.arange(150, device='cuda')[None].expand_as(rows)
    truth = lex_keys(dtw_pairs(q, db, rows.flatten(), cols.flatten()).reshape_as(rows), cols)
    for k in (1, 5, 20):
        actual = model.search_range(q, db, search_index=tree, query_batch=3, top_k=k)['indices']
        assert torch.equal(actual, truth.topk(k, largest=False).values & 0xffffffff)
    fixed = LEAP.from_asset(config=RetrievalConfig('dtw', 'signal'))
    with pytest.raises(ValueError, match='different model'):
        fixed.candidate_indices(db, candidate_k=20)
    next(model.dtw_router.parameters()).add_(.01)
    with pytest.raises(ValueError, match='routing weights'):
        model.candidate_indices(db, candidate_k=20)
