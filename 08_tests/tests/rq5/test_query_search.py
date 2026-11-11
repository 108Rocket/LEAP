# Test group: RQ5.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search; RQ5: Initialization Budget and Result Count.
# Checks: Full-database versus initial-pool search with varying result count k.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""GPU contracts for the single-query API's variable result count."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA retrieval tests')


@pytest.fixture(params=['hausdorff', 'dfd', 'dtw'], scope='module')
def database(request):
    from leap import LEAP
    from leap import RetrievalConfig
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    model = LEAP.from_asset(config=RetrievalConfig.from_json(root / '02_configs' / request.param / (request.param + '.json')))
    rng = np.random.default_rng(519)
    paths = [model.mean + rng.normal(size=(2 + i % 6, 2)).cumsum(0) * 20 for i in range(1102)]
    paths[1] = paths[0].copy()
    db = model.encode(paths)
    return model, db, model.build_search_index(db)


@pytest.mark.parametrize('n', [1, 20, 100, 1000])
def test_global_equals_full_original_geometry(database, n):
    from leap.common.query_search import query_row, raw_scores, search_row
    from leap.common.interval_index import lex_keys
    model, db, tree = database
    row = 7
    q = query_row(db, row, model.config.distance)
    ids = torch.arange(len(db.coordinates), device='cuda')[None]
    scores = raw_scores(q, db, ids, model.config.distance)
    scores[0, row] = torch.inf
    expected = (lex_keys(scores, ids).topk(n, largest=False).values & 0xffffffff)[0].tolist()
    actual = search_row(model, db, tree, row, n, 'complete')
    assert actual['ids'] == expected
    assert actual['scope'] == 'database'


def test_pool_stays_inside_initial_candidates(database):
    from leap.common.query_search import search_row
    model, db, tree = database
    initial = search_row(model, db, tree, 1, 150, 'initial')
    pool = search_row(model, db, tree, 1, 100, 'pool', 150)
    assert set(pool['ids']).issubset(initial['ids'])
    assert len(set(pool['ids'])) == 100 and 1 not in pool['ids']
    assert pool['survivor_count'] == 150


def test_invalid_n_pool_and_row(database):
    from leap.common.query_search import search_row
    model, db, tree = database
    for row, n, mode, pool in [(0, 1001, 'complete', 50), (-1, 20, 'initial', 50),
                                (0, 100, 'pool', 50), (0, True, 'initial', 50)]:
        with pytest.raises(ValueError):
            search_row(model, db, tree, row, n, mode, pool)
