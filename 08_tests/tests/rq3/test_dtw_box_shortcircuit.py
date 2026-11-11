# Test group: RQ3.
# Paper RQ: RQ2: Role of Full Search; RQ3: Contributions of C1--C3.
# Checks: Two-direction DTW box filtering preserves decisions and reduces checks.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Exact mask and work gates for DTW's two-direction box filter."""
from pathlib import Path

import numpy as np
import pytest
import torch
import triton

from leap import RetrievalConfig
from leap import LEAP
from leap.dtw.dtw_parallel import check_leaf_tiles
from leap.common.runtime import options

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='L20 CUDA required')


def run_kernel(lengths, queries, radius, seed_k, tile):
    rng = np.random.default_rng(1819)
    paths = [rng.normal(size=(int(n), 2)).astype(np.float32) + i*.1
             for i, n in enumerate(lengths)]
    points = np.concatenate(paths)
    offsets = np.r_[0, np.cumsum(lengths)]
    qp = np.concatenate(queries).astype(np.float32)
    qo = np.r_[0, np.cumsum([len(x) for x in queries])]
    boxes = np.array([np.r_[x.min(0), x.max(0)] for x in paths], np.float32)
    qbox = np.array([np.r_[x.min(0), x.max(0)] for x in queries], np.float32)
    p, o = torch.tensor(points, device='cuda'), torch.tensor(offsets, device='cuda')
    q, off = torch.tensor(qp, device='cuda'), torch.tensor(qo, device='cuda')
    box, qb = torch.tensor(boxes, device='cuda'), torch.tensor(qbox, device='cuda')
    members = torch.arange(128, device='cuda', dtype=torch.int64).reshape(2, 64)
    members[-1, -7:] = -1
    lids = torch.tensor([0, 1, 0, 1, 0, 1], device='cuda', dtype=torch.int32)
    rows = torch.tensor([0, 0, 1, 1, 2, 2], device='cuda', dtype=torch.int32)
    seeds = torch.arange(seed_k, device='cuda', dtype=torch.int64)[None].repeat(3, 1)*2
    selfs = torch.tensor([-1, 13, 77], device='cuda', dtype=torch.int64)
    radii = torch.full((3,), radius, device='cuda', dtype=torch.float32)
    slack = torch.zeros_like(radii)
    n = triton.next_power_of_2(int(max(max(lengths), max(map(len, queries)))))
    outputs = []
    for enabled in [False, True]:
        keep = torch.full((6, 64), -7, device='cuda', dtype=torch.int32)
        examined = torch.full_like(keep, -7)
        counts = torch.zeros((6, 64, 4), device='cuda', dtype=torch.int64)
        check_leaf_tiles[(6, triton.cdiv(64, tile))](
            p, o, q, off, members, box, qb, radii, slack, selfs, seeds,
            lids, rows, keep, examined, 64, n, seed_k, tile,
            max(64, triton.next_power_of_2(seed_k)), seed_k > 64,
            seed_k.bit_length(), enabled, counts, True,
            num_warps=8, enable_fp_fusion=False)
        outputs.append((keep, examined, counts))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])
    assert int(outputs[1][2][..., :2].sum()) <= int(outputs[0][2][..., :2].sum())
    assert int(outputs[1][2][..., 3].sum()) <= int(outputs[0][2][..., 3].sum())


@pytest.mark.parametrize('width,tile', [(64, 64), (256, 16), (4096, 4)])
@pytest.mark.parametrize('seed_k', [0, 20, 128])
@pytest.mark.parametrize('radius', [-1., .75, 1e8])
def test_masks_work_and_sorted_membership(width, tile, seed_k, radius):
    rng = np.random.default_rng(2819)
    lengths = rng.integers(2, min(width, 80), size=256)
    lengths[31] = width - 1
    queries = [rng.normal(size=(n, 2)).astype(np.float32) for n in [2, 17, 31]]
    run_kernel(lengths, queries, radius, seed_k, tile)


@pytest.mark.parametrize('radius', [np.nextafter(np.float32(.125), np.float32(0)),
                                  np.float32(.125),
                                  np.nextafter(np.float32(.125), np.float32(1))])
def test_exact_threshold_neighbors(radius):
    p = torch.tensor([[0., 0.], [1., 0.], [.125, 0.], [1.125, 0.]], device='cuda')
    offsets = torch.tensor([0, 2, 4], device='cuda')
    qp = p[:2]
    qo = torch.tensor([0, 2], device='cuda')
    mem = torch.full((1, 64), -1, device='cuda', dtype=torch.int64)
    mem[0, 0] = 1
    box = torch.tensor([[0., 0., 1., 0.], [.125, 0., 1.125, 0.]], device='cuda')
    r = torch.tensor([radius], device='cuda')
    slack = torch.zeros_like(r)
    selfs = torch.tensor([-1], device='cuda', dtype=torch.int64)
    seed = torch.tensor([[0]], device='cuda', dtype=torch.int64)
    lids = torch.tensor([0], device='cuda', dtype=torch.int32)
    masks = []
    for enabled in [False, True]:
        keep = torch.zeros((1, 64), device='cuda', dtype=torch.int32)
        examined = keep.clone()
        check_leaf_tiles[(1,)](
            p, offsets, qp, qo, mem, box, box[:1], r, slack, selfs, seed, lids, lids,
            keep, examined, 64, 64, 1, 64, 64, False, 0, enabled,
            num_warps=8, enable_fp_fusion=False)
        masks.append(keep)
    assert torch.equal(*masks)
    assert int(masks[0][0, 0]) == int(radius >= np.float32(.125))


def test_adopted_mode_defaults():
    assert options.dtw_box_shortcircuit is True
    assert options.dtw_single_box_shortcircuit is False


@pytest.mark.parametrize('enabled', [False, True])
@torch.no_grad()
def test_single_strategy_is_frozen_and_independent(monkeypatch, enabled):
    config = RetrievalConfig.from_json(
        Path(__file__).parents[3] / '02_configs' / 'dtw' / 'single_query_dtw.json')
    model = LEAP.from_asset(config=config)
    rng = np.random.default_rng(1913)
    paths = [model.mean + rng.normal(size=(n, 2))*model.std
             for n in ([3, 17, 33, 65]*16)]
    database = model.encode(paths)
    tree = model.build_search_index(database)
    monkeypatch.setattr(options, 'dtw_single_box_shortcircuit', enabled)
    session = model.query_session(database, search_index=tree)
    assert session.plan.dtw_box_shortcircuit is enabled
    monkeypatch.setattr(options, 'dtw_box_shortcircuit', not enabled)
    monkeypatch.setattr(options, 'dtw_single_box_shortcircuit', not enabled)
    assert session.plan.dtw_box_shortcircuit is enabled
    query = model.encode([paths[2]])
    expected = model.search_range(query, database, search_index=tree)['indices']
    np.testing.assert_array_equal(session.search(paths[2], mode='full'),
                                  expected.cpu().numpy())
