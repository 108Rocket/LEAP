# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance.
# Checks: Specialized single-request execution preserves outputs and session state.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from leap import LEAP
from leap import RetrievalConfig
from leap.common.runtime.context import activate, current


def test_execution_modes_validation():
    assert RetrievalConfig().query_profile == 'throughput'
    with pytest.raises(ValueError, match='single_query_engine'):
        RetrievalConfig(single_query_engine='unknown')


@pytest.mark.parametrize('metric', ['dfd', 'dtw'])
@torch.no_grad()
def test_long_shape_sidecars_and_results(metric):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    cfg = RetrievalConfig.from_json(Path(__file__).parents[3]/'02_configs'/metric/f'single_query_{metric}.json')
    model = LEAP.from_asset(config=cfg)
    rng = np.random.default_rng(78)
    sizes = ([2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257]*6)[:64]
    paths = [model.mean + rng.normal(size=(n, 2)).cumsum(0)*model.std*.05 for n in sizes]
    database = model.encode(paths)
    tree = model.build_search_index(database)
    fast = model.query_session(database, search_index=tree, initial_capacity=2)
    fast.warmup([paths[11], paths[1], paths[6], paths[8]], modes=('ranking',))
    for path in [paths[1], paths[6], paths[8], paths[11], paths[1]]:
        reference = model.encode([path])
        for mode in ('ranking', 'full'):
            expected = (model.candidate_indices(reference, database, 20, search_index=tree)
                        if mode == 'ranking' else model.search_range(reference, database, search_index=tree)['indices'])
            got = fast.search(path, mode=mode)
            np.testing.assert_array_equal(got, expected.cpu().numpy())
            assert current() is None
        with torch.cuda.stream(fast.stream), activate(fast.plan, len(path)):
            query = model._finish_encoding(fast.fields, fast.points[:len(path)], fast.offsets)
            fast.stream.synchronize()
            if metric == 'dfd':
                for key in (8, 16, 32):
                    assert torch.equal(query.dfd_summary['ids'][key], reference.dfd_summary['ids'][key])
            else:
                assert torch.equal(query.ordered_summary, reference.ordered_summary)
                for key in ('signal', 'radial', 'ordered'):
                    assert torch.equal(query.dtw_probe_summary[key], reference.dtw_probe_summary[key])
        assert current() is None
    # Same-length requests with different contents must not return graph-cached answers.
    changed = paths[1] + np.array([.25, -.15])*model.std
    reference = model.encode([changed])
    expected = model.candidate_indices(reference, database, 20, search_index=tree).cpu().numpy()
    np.testing.assert_array_equal(fast.search(changed, mode='ranking'), expected)
    with pytest.raises(ValueError):
        fast.search(np.zeros((1, 2)))
    assert current() is None
    assert fast.plan.stats['max_geometry_reductions'] == 1


@torch.no_grad()
def test_two_sessions_keep_execution_state_private():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    cfg = RetrievalConfig(distance='dfd', query_profile='single_query')
    model = LEAP.from_asset(config=cfg)
    rng = np.random.default_rng(71)
    data = [model.mean + rng.normal(size=(15, 2))*model.std for _ in range(64)]
    db = model.encode(data); tree = model.build_search_index(db)
    a = model.query_session(db, search_index=tree)
    b = model.query_session(db, search_index=tree)
    first = a.search(data[0], mode='ranking')
    b.search(data[1], mode='full')
    np.testing.assert_array_equal(first, a.search(data[0], mode='ranking'))
    assert a.plan is not b.plan and current() is None


@torch.no_grad()
def test_unseen_length_does_not_capture_on_request():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    cfg=RetrievalConfig(distance='dtw',dtw_seed='signal',dtw_routing='learned',query_profile='single_query')
    model=LEAP.from_asset(config=cfg)
    rng=np.random.default_rng(51)
    data=[model.mean+rng.normal(size=(15,2))*model.std for _ in range(128)]
    db=model.encode(data);tree=model.build_search_index(db)
    session=model.query_session(db,search_index=tree)
    session.search(data[0],mode='ranking')
    assert session.plan.stats['graph_builds']==0 and not session.plan.capture_enabled
    session.warmup([data[0]],modes=('ranking',))
    assert session.plan.stats['graph_builds']==1 and not session.plan.capture_enabled
    new=model.mean+rng.normal(size=(17,2))*model.std
    got=session.search(new,mode='ranking')
    q=model.encode([new])
    expected=model.candidate_indices(q,db,20,search_index=tree).cpu().numpy()
    np.testing.assert_array_equal(got,expected)
    assert session.plan.stats['graph_builds']==1 and session.plan.stats['eager_requests']>=2
    assert current() is None
