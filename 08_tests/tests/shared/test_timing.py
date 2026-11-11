# Test group: SHARED.
# Paper RQ: Shared support for RQ1--RQ5.
# Checks: Timing boundaries, warmup exclusion and arithmetic seed aggregation.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from leap.reproduction import config
from leap.cli import main
from leap.reproduction.replay import save
from leap.reproduction.timing import alternating, prepare_plan, selected_ids, summarize, timed
from leap.reproduction.timing_summary import aggregate


def base_plan(tmp_path):
    value = json.loads(config.TEMPLATE.read_text())
    value['datasets']['porto']['self_query'] = False
    value['datasets']['porto']['queries'] = 'queries.npz'
    return config.plan(tmp_path / 'input.json', value, 'porto-dtw')


def test_dry_run_no_gpu_import_or_execution(tmp_path, monkeypatch):
    import leap.reproduction.timing as timing
    monkeypatch.setattr(timing, 'run', lambda *a: pytest.fail('unexpected execution'))
    path = config.ROOT / '05_examples/reproduce_seed2000.json'
    assert main(['timing', '--config', str(path), '--job', 'sdd.porto.dtw', '--dry-run']) == 0


@pytest.mark.parametrize('fraction', [0, -1, 1.1, float('nan'), float('inf')])
def test_memory_fraction(tmp_path, fraction):
    with pytest.raises(ValueError, match='memory_fraction'):
        prepare_plan(base_plan(tmp_path), memory_fraction=fraction)


@pytest.mark.parametrize('change', [dict(self_query=True), dict(initial_k=32), dict(candidate_search='full')])
def test_single_protocol_guards(tmp_path, change):
    plan = dict(base_plan(tmp_path), **change)
    with pytest.raises(ValueError):
        prepare_plan(plan)
    prepare_plan(plan, scope='batch')


@pytest.mark.parametrize('key', ['large_k', 'full_large_k'])
def test_no_silent_large_k_substitution(tmp_path, key):
    plan = base_plan(tmp_path)
    plan['retrieval'][key] = True
    with pytest.raises(ValueError, match='Top-1..64'):
        prepare_plan(plan)


def test_query_selection(tmp_path):
    assert selected_ids(3).tolist() == [0, 1, 2]
    assert len(selected_ids(100)) == 64
    path = tmp_path / 'ids.npy'
    np.save(path, [2, 0])
    assert selected_ids(3, path).tolist() == [2, 0]
    for ids in ([1, 1], [], [-1], [3], [[0]], [0.5]):
        np.save(path, ids)
        with pytest.raises(ValueError):
            selected_ids(3, path)
    with pytest.raises(ValueError):
        prepare_plan(base_plan(tmp_path), scope='batch', query_ids=path)


def test_mean_not_median_and_no_warmup():
    rows = [dict(round=r, stage='full', seconds=s) for r, s in [(-1, 1000), (0, 1), (1, 2), (2, 9)]]
    stats = summarize(rows)['full']
    assert stats['samples'] == 3 and stats['mean_seconds'] == 4
    assert stats['min_seconds'] == 1 and stats['max_seconds'] == 9
    assert alternating(0) == tuple(reversed(alternating(1)))


def test_synchronization_inside_stop_boundary(monkeypatch):
    import leap.reproduction.timing as timing
    calls = []
    ticks = iter([100, 600])
    def clock():
        calls.append('clock')
        return next(ticks)
    monkeypatch.setattr(timing.time, 'perf_counter_ns', clock)
    value, seconds = timed(lambda: calls.append('work') or 42, lambda: calls.append('sync'))
    assert (value, seconds) == (42, 5e-7)
    assert calls == ['sync', 'clock', 'work', 'sync', 'clock']


def write_result(directory, value, model):
    directory.mkdir()
    report = dict(status='complete', repeat_outputs_equal=True, database_sha256='db',
        queries_sha256='q', database_size=10, batch_queries=3, single_query_ids=[0],
        retrieval={}, source_sha256={}, gpu='L20', versions={}, model_asset_id=model,
        summary={'full': dict(samples=3, mean_seconds=value)},
        plan={key: 1 for key in ('timing_schema', 'scope', 'top_n', 'initial_k', 'query_batch',
            'candidate_search', 'warmup', 'repeats', 'threads', 'memory_fraction')})
    save(directory / 'result.json', report)
    (directory / 'samples.jsonl').write_text('{}\n')
    save(directory / 'ALL_DONE.json', dict(success=True,
        result_sha256=config.digest(directory / 'result.json'),
        samples_sha256=config.digest(directory / 'samples.jsonl')))


def test_seed_aggregation_and_reject_relabeling(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    write_result(a, 2, 'asset-a')
    write_result(b, 4, 'asset-b')
    result = aggregate([(2000, a), (2001, b)])
    assert result['summary']['full']['mean_seconds'] == 3
    assert result['independent_training_not_verified']
    with pytest.raises(ValueError, match='same timing'):
        aggregate([(2000, a), (2001, a)])
    with pytest.raises(ValueError, match='distinct'):
        aggregate([(2000, a), (2000, b)])
    (b / 'samples.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        aggregate([(2000, a), (2001, b)])


def test_reject_different_protocols(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    write_result(a, 1, 'a')
    write_result(b, 2, 'b')
    result = json.loads((b / 'result.json').read_text())
    result['single_query_ids'] = [1]
    save(b / 'result.json', result)
    receipt = json.loads((b / 'ALL_DONE.json').read_text())
    receipt['result_sha256'] = config.digest(b / 'result.json')
    save(b / 'ALL_DONE.json', receipt)
    with pytest.raises(ValueError, match='protocols'):
        aggregate([(0, a), (1, b)])


@pytest.mark.parametrize('metric,profile,asset', [
    ('hausdorff', 'batch_query_hausdorff', 'porto_dual64'),
    ('hausdorff', 'batch_query_pointset_hd', 'sdd_porto_hausdorff64'),
    ('dfd', 'batch_query_dfd', 'porto_dual64'),
    ('dtw', 'batch_query_dtw', 'porto_dtw_signal16')])
def test_public_timing_gpu(tmp_path, metric, profile, asset):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from leap.reproduction.timing import run
    from leap.common.dual_field import DEFAULT_DUAL_ASSET
    asset_path = (DEFAULT_DUAL_ASSET if asset == 'porto_dual64' else
                  config.ROOT / '01_LEAP/leap/assets' / (asset + '.json'))
    model = json.loads(asset_path.read_text())
    mean, std = np.array(model['geometry_mean']), np.array(model['geometry_std'])
    rng = np.random.default_rng(12)
    for name, size in [('database', 160), ('queries', 3)]:
        paths = [rng.normal(size=(3 + i % 5, 2)) * std + mean for i in range(size)]
        np.savez(tmp_path / (name + '.npz'), points=np.concatenate(paths),
                 offsets=np.r_[0, np.cumsum([len(p) for p in paths])],
                 geometry_mean=mean, geometry_std=std)
    cfg = dict(version=1, device='cuda:0', threads=4, output_root='runs',
        datasets={'external': dict(database='database.npz', queries='queries.npz', self_query=False)},
        jobs={'test': dict(dataset='external', metric=metric, asset=str(asset_path),
                          profile=f'repo:02_configs/{metric}/{profile}.json', top_n=5, warmup=1, repeats=2)})
    if metric == 'dtw':
        cfg['jobs']['test']['routing_asset'] = 'repo:01_LEAP/leap/assets/porto_dtw_routing_e100.json'
        from leap import LEAP
        from leap import RetrievalConfig
        with pytest.raises(FileNotFoundError):
            LEAP.from_asset(asset_path, config=RetrievalConfig.from_json(
                config.ROOT / f'02_configs/{metric}/{profile}.json'), routing_asset=tmp_path / 'absent.json')
    plan = prepare_plan(config.plan(tmp_path / 'config.json', cfg, 'test'))
    out = run(plan, tmp_path / 'result')
    result = json.loads((out / 'result.json').read_text())
    assert (out / 'ALL_DONE.json').exists() and not (out / 'FAILED.json').exists()
    assert result['summary']['encode_database']['samples'] == 2
    assert result['summary']['single_full']['samples'] == 6
    assert result['summary']['batch_ranking']['samples'] == 2
    assert result['repeat_outputs_equal']
    assert len((out / 'hardware.jsonl').read_text().splitlines()) == 3
    assert all(row['ordered_equal_queries'] == 3 for row in result['single_batch_comparison'].values())
    before = (out / 'result.json').read_bytes()
    with pytest.raises(FileExistsError):
        run(plan, out)
    assert (out / 'result.json').read_bytes() == before
    bad = copy.deepcopy(plan)
    bad['database']['sha256'] = 'bad'
    failed = tmp_path / 'failed'
    with pytest.raises(ValueError, match='SHA256'):
        run(bad, failed)
    assert (failed / 'FAILED.json').exists() and not (failed / 'ALL_DONE.json').exists()
