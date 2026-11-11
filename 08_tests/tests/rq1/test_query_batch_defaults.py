# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance.
# Checks: Published batch defaults, overrides and memory limits.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

from pathlib import Path

import pytest

from leap.common.runtime.batching import resolve_query_batch
from leap.common.config import RetrievalConfig


@pytest.mark.parametrize('metric,expected', [('hausdorff',512),('dfd',512),('dtw',2000)])
def test_default_and_published_batch_profile(metric, expected):
    assert resolve_query_batch(RetrievalConfig(distance=metric)) == expected
    path = Path(__file__).resolve().parents[3] / '02_configs' / metric / f'batch_query_{metric}.json'
    assert resolve_query_batch(RetrievalConfig.from_json(path)) == expected


def test_override_precedence_and_reference_path():
    config = RetrievalConfig(query_batch=256)
    assert resolve_query_batch(config) == 256
    assert resolve_query_batch(config,32) == 32
    assert resolve_query_batch(config,default=128) == 256
    reference = RetrievalConfig(distance='dtw',dtw_seed='signal',dtw_execution='reference')
    assert resolve_query_batch(reference) == 64


@pytest.mark.parametrize('bad', [True,False,0,-1,1.5,'32'])
def test_invalid_batch_not_silently_truncated(bad):
    with pytest.raises(ValueError,match='positive integer'):
        RetrievalConfig(query_batch=bad)
    with pytest.raises(ValueError,match='positive integer'):
        resolve_query_batch(RetrievalConfig(),bad)


@pytest.mark.parametrize('metric',['hausdorff','dfd','dtw'])
def test_dense_memory_cap(metric):
    config = RetrievalConfig(distance=metric,query_batch=512)
    # K controls output count, not the size of a dense query/database score matrix.
    assert resolve_query_batch(config,database_size=1597579,max_score_pairs=26000000) == 16
    assert resolve_query_batch(config,database_size=100000,max_score_pairs=26000000) == 260


def test_dense_defaults_and_guards():
    config = RetrievalConfig()
    assert resolve_query_batch(config,default=128,database_size=100000,max_score_pairs=26000000) == 128
    with pytest.raises(ValueError,match='capacity'):
        resolve_query_batch(config,database_size=26000001,max_score_pairs=26000000)
    with pytest.raises(ValueError,match='needs'):
        resolve_query_batch(config,database_size=100000)
