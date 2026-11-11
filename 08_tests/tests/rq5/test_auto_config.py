# Test group: RQ5.
# Paper RQ: RQ1: Overall Performance; RQ5: Initialization Budget and Result Count.
# Checks: Candidate-budget constraints and automatic configuration; no budget sweep.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Automatic plans are arithmetic, explicit and independent of retrieval results."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from leap import RetrievalConfig
from leap import TopKProfile
from leap import LEAP
from leap.common.runtime.configuration import MAX_SCORE_PAIRS


def config(metric='hausdorff', **kw):
    return RetrievalConfig(distance=metric, parameter_mode='auto',
                           dtw_seed='signal', **kw)


@pytest.mark.parametrize('metric', ['hausdorff', 'dfd', 'dtw'])
@pytest.mark.parametrize('k', [1, 20, 64, 65, 100, 1000])
@pytest.mark.parametrize('n', [100_000, 1_600_000])
def test_resolve_limits_and_no_mutation(metric, k, n):
    cfg = config(metric)
    before = asdict(cfg)
    plan = cfg.resolve(n, k)
    c = plan.config
    assert asdict(cfg) == before
    assert k <= c.initial_k <= n
    assert c.full_large_k == (k > 64)
    assert c.parameter_mode == 'manual' and not c.top_k_profiles
    assert c.precision == cfg.precision
    if k > 64:
        assert plan.path == 'tree_full'
        assert c.query_batch == (2000 if metric == 'dtw' else 512)
        expected = dict(hausdorff=None, dfd=2*k, dtw=(32*k+4)//5)[metric]
        assert c.initial_pool == expected
    assert not plan.as_dict()['recall_guarantee']
    assert plan.as_dict()['benchmarks_run'] == 0


def test_native_endpoint_budgets():
    for metric, seed, pool in [('hausdorff',50,None), ('dfd',32,64), ('dtw',20,128)]:
        cfg = config(metric)
        small = cfg.resolve(1_597_579, 20).config
        assert (small.initial_k, small.initial_pool) == (seed, pool)
        assert small.query_batch == (2000 if metric == 'dtw' else 512)
        assert cfg.resolve(1_597_579, 1000).config.query_batch == (2000 if metric == 'dtw' else 512)


def test_overrides_precedence_and_immutable_roundtrip(tmp_path):
    profiles = {'20': {'initial_k': 40, 'initial_pool': 80, 'query_batch': 7},
                '1000': {'initial_k': 1100, 'initial_pool': 7000}}
    cfg = config('dtw', initial_k=30, query_batch=10, top_k_profiles=profiles)
    profiles['20']['initial_k'] = 3
    c = cfg.resolve(2000,20).config
    assert (c.initial_k,c.initial_pool,c.query_batch) == (40,80,7)
    c = cfg.resolve(2000,20,initial_k=60,query_batch=9).config
    assert (c.initial_k,c.query_batch) == (60,9)
    assert cfg.resolve(1_600_000,1000).config.full_large_k
    path = tmp_path/'config.json'
    path.write_text(json.dumps(asdict(cfg)), encoding='utf-8')
    assert RetrievalConfig.from_json(path) == cfg
    assert cfg.top_k_profiles[0] == TopKProfile(20,40,80,7)


def test_small_database_and_self():
    plan = config('dfd').resolve(21,20,exclude_self=True)
    assert (plan.available,plan.config.initial_k,plan.config.initial_pool) == (20,20,20)
    with pytest.raises(ValueError,match='available'):
        config().resolve(20,20,exclude_self=True)
    with pytest.raises(ValueError,match='available'):
        config(top_k_profiles={'20':{'initial_k':50}}).resolve(40,20)


def test_dense_overrides_are_capped():
    cfg = config('dtw', initial_search='full', top_k_profiles={'20':{'initial_pool':256,'query_batch':9999}})
    plan = cfg.resolve(1_600_000,20)
    assert plan.config.full_large_k
    assert plan.config.query_batch == 16
    assert plan.config.initial_pool == 256
    with pytest.raises(ValueError,match='bounded coarse'):
        cfg.resolve(MAX_SCORE_PAIRS+1,20)


def test_single_profile_and_near_far():
    cfg = config('hausdorff',hausdorff_target='pointset',hausdorff_fields='near_far',
                 query_profile='single_query')
    for k in (1,20,1000):
        plan = cfg.resolve(10000,k)
        assert plan.config.query_batch == 1
        assert plan.path.startswith('near_far')
    assert config().resolve(10000,20,query_count=3).config.query_batch == 3


@pytest.mark.parametrize('value',[True,0,-1,1.5,'20'])
def test_invalid_dimensions(value):
    with pytest.raises(ValueError):
        config().resolve(2000,value)
    with pytest.raises(ValueError):
        config().resolve(value,20)
    with pytest.raises(ValueError):
        config().resolve(2000,20,initial_k=value)


@pytest.mark.parametrize('kwargs',[
    {'parameter_mode':'fast'}, {'parameter_mode':'auto','large_k':True},
    {'parameter_mode':'auto','distance':'dtw'},
    {'initial_k':True}, {'top_k_profiles':{'20':{}}},
    {'parameter_mode':'auto','top_k_profiles':{'20':{'target_recall':.999}}},
    {'parameter_mode':'auto','top_k_profiles':{'20':{'initial_k':19}}},
    {'parameter_mode':'auto','top_k_profiles':{'20':{'initial_pool':64}}},
    {'parameter_mode':'auto','top_k_profiles':{True:{}}},
    {'parameter_mode':'auto','top_k_profiles':{'20':{},'020':{}}},
])
def test_config_rejects_unsupported_requests(kwargs):
    with pytest.raises(ValueError):
        RetrievalConfig(**kwargs)


def test_pool_cannot_be_smaller_than_seed():
    with pytest.raises(ValueError,match='cover'):
        config('dfd',top_k_profiles={'20':{'initial_pool':10}}).resolve(1000,20)


def test_auto_search_routes_full_without_changing_manual_api():
    calls=[]
    model=SimpleNamespace(config=config(),
        search_range=lambda *a,**kw: calls.append((a,kw)) or 'full')
    assert LEAP.search(model,'q','d',top_k=1000,search_index='tree') == 'full'
    assert calls[0][1]['top_k']==1000
    with pytest.raises(ValueError,match='Full search'):
        LEAP.search(model,'q','d',candidate_search='tree')


def test_json_examples_load():
    root=Path(__file__).parents[3]/'02_configs'
    paths = sorted(root.rglob('auto_*.json'))
    assert len(paths) == 4
    for path in paths:
        cfg=RetrievalConfig.from_json(path)
        for k in (1,20,1000):
            assert cfg.resolve(1_600_000,k).profile_matched


def test_manual_default_unchanged():
    assert RetrievalConfig().parameter_mode == 'manual'
    with pytest.raises(ValueError,match='parameter_mode'):
        RetrievalConfig().resolve(2000,20)


def test_explicit_manual_dtw_pool_reaches_candidate_kernel():
    from unittest.mock import patch
    cfg=RetrievalConfig(distance='dtw',dtw_seed='signal',initial_pool=256,initial_search='full')
    model=SimpleNamespace(config=cfg)
    db=SimpleNamespace(coordinates=range(1000))
    with patch('leap.dtw.dtw_probes.candidates',return_value='ids') as kernel:
        assert LEAP._dtw_candidates(model,'q',db,20,None)=='ids'
        assert kernel.call_args.kwargs['pool']==256
        assert kernel.call_args.kwargs['routing_tree'] is None
