# Test group: RQ2. Role of Full Search: reuse exact seed distances in refinement.
from dataclasses import replace
import numpy as np
import pytest
import torch

from leap import LEAP, RetrievalConfig
from leap.hausdorff.cached_refinement import cached_pairs
from leap.common.raw_geometry_refinement import hausdorff_distances

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA validation')


@pytest.mark.parametrize('point', [False, True])
@pytest.mark.parametrize('width', [1, 20, 50, 65, 129])
@torch.no_grad()
def test_cache_matches_raw_and_skips_geometry(point, width):
    rng = np.random.default_rng(2000)
    model = LEAP(rng.normal(size=(64, 2)), [0., 0.], [1., 1.],
                 config=RetrievalConfig('hausdorff'))
    lengths = [2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257]
    paths = [rng.normal(size=(lengths[i % len(lengths)], 2)) for i in range(140)]
    paths[1] = paths[0].copy()
    q = model.encode(paths[:3])
    db = model.encode(paths)
    seeds = torch.arange(width, device='cuda')[None].expand(3, -1).contiguous()
    rows = torch.arange(3, device='cuda')[:, None].expand_as(seeds).reshape(-1)
    raw = hausdorff_distances(q, db, rows, seeds.flatten(), point=point).reshape_as(seeds)
    # Poison geometry after computing the seed distances: a hit must not score it again.
    badq = replace(q, points=torch.full_like(q.points, torch.nan))
    baddb = replace(db, points=torch.full_like(db.points, torch.nan))
    got = cached_pairs(badq, baddb, rows, seeds.flatten(), seeds, raw, point=point, bounded=False)
    assert torch.equal(got, raw.flatten())
    ids = torch.arange(140, device='cuda')[None].expand(3, -1).contiguous()
    rows = torch.arange(3, device='cuda')[:, None].expand_as(ids).reshape(-1)
    expected = hausdorff_distances(q, db, rows, ids.flatten(), point=point)
    got = cached_pairs(q, db, rows, ids.flatten(), seeds, raw, point=point, bounded=False)
    assert torch.equal(got, expected)
    if not point:
        radius = torch.full((3,), torch.inf, device='cuda')
        assert torch.equal(cached_pairs(q, db, rows, ids.flatten(), seeds, raw, radius), expected)
    assert not cached_pairs(q, db, rows[:0], ids.flatten()[:0], seeds, raw,
                            point=point, bounded=False).numel()
    with pytest.raises(ValueError, match='Cached distances'):
        cached_pairs(q, db, rows, ids.flatten(), seeds, raw[:, :0], point=point)
