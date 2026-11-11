# Test group: SHARED.
# Paper RQ: Shared support for RQ1--RQ5.
# Checks: Output grouping, configuration snapshots and console/failure records.
# Role: infrastructure checks, not a model experiment.
import json
from pathlib import Path
import sys

import pytest

from leap.checkpoints import ROOT
from leap.cli import main
from leap.reproduction.logging import output_path, run_logged


def test_shipped_log_examples_are_porto_hausdorff_only():
    root = ROOT/'14_logs/examples'
    records = json.loads((root/'INDEX.json').read_text())
    expected = {f'{rq}/hausdorff' for rq in ('training', 'rq1', 'rq2', 'rq3', 'rq4', 'rq5')}
    assert {record['directory'] for record in records} == expected
    assert {p.parent.relative_to(root).as_posix() for p in root.rglob('SOURCE.json')} == expected
    for record in records:
        folder = root/record['directory']
        assert (folder/'SOURCE.json').is_file()
        assert (folder/'configuration.json').is_file()
    assert not any(p.name in ('dfd', 'dtw', 'code_tests') for p in root.rglob('*'))


def test_log_run_keeps_configuration_stdout_and_stderr(tmp_path):
    out = tmp_path/'run'
    def execute(artifacts):
        assert not artifacts.exists()
        artifacts.mkdir()
        print('epoch record')
        print('warning record', file=sys.stderr)
        (artifacts/'result.json').write_text('{}')
        return artifacts
    assert run_logged(out, {'seed': 2000}, execute) == out/'artifacts'
    assert json.loads((out/'configuration.json').read_text())['seed'] == 2000
    assert json.loads((out/'status.json').read_text())['status'] == 'complete'
    assert (out/'console.log').read_text() == 'epoch record\nwarning record\n'


def test_failed_run_keeps_traceback_and_restores_streams(tmp_path):
    before = sys.stdout, sys.stderr
    def fail(artifacts):
        raise RuntimeError('fixture failure')
    out = tmp_path/'failure'
    with pytest.raises(RuntimeError, match='fixture failure'):
        run_logged(out, {}, fail)
    assert (sys.stdout, sys.stderr) == before
    assert json.loads((out/'status.json').read_text())['status'] == 'failed'
    assert 'RuntimeError: fixture failure' in (out/'console.log').read_text()


def test_existing_output_is_never_reused(tmp_path):
    with pytest.raises(FileExistsError):
        run_logged(tmp_path, {}, lambda _: pytest.fail('must not execute'))


@pytest.mark.parametrize('category', ['training', 'evaluation', 'timing', 'tests'])
def test_log_categories(category, tmp_path):
    result = output_path(tmp_path, category, 'sdd.porto.dtw')
    assert result.parent == tmp_path/category
    assert result.name.startswith('sdd.porto.dtw-')
    assert not result.exists()


def test_evaluation_cli_captures_output_without_model_execution(tmp_path, monkeypatch):
    import leap.reproduction.replay as replay
    calls = []
    def fake(plan, output):
        calls.append(plan['job'])
        Path(output).mkdir()
        print('fixture evaluation')
        return output
    monkeypatch.setattr(replay, 'run', fake)
    out = tmp_path/'evaluation'
    assert main(['evaluate', '--job', 'sdd.porto.dtw', '--output', str(out)]) == 0
    assert calls == ['sdd.porto.dtw']
    cfg = json.loads((out/'configuration.json').read_text())
    assert cfg['configuration']['output_root'] == 'repo:14_logs'
    assert 'fixture evaluation' in (out/'console.log').read_text()


def test_cli_dry_run_has_no_log_side_effects(tmp_path, monkeypatch):
    import leap.reproduction.replay as replay
    monkeypatch.setattr(replay, 'run', lambda *_: pytest.fail('must not execute'))
    out = tmp_path/'dry'
    assert main(['evaluate', '--job', 'sdd.porto.dtw', '--output', str(out), '--dry-run']) == 0
    assert not out.exists()


def test_training_cli_logs_only_explicit_execution(tmp_path, monkeypatch):
    import leap.reproduction.training as training
    cfg = json.loads((ROOT/'05_examples/reproduce_seed2000.json').read_text())
    cfg['training'] = {'raw_inputs': str(tmp_path/'missing.npz'), 'seed': 2000}
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    calls = []
    def fake(plan):
        calls.append(plan)
        Path(plan['output']).mkdir()
        print('fixture training, no model')
    monkeypatch.setattr(training, 'run', fake)
    out = tmp_path/'training'
    args = ['train', '--config', str(path), '--recipe', 'dtw-probes', '--output', str(out)]
    assert main(args) == 0
    assert not calls and not out.exists()
    assert main(args+['--execute']) == 0
    assert len(calls) == 1 and Path(calls[0]['output']) == out/'artifacts'
    assert json.loads((out/'status.json').read_text())['status'] == 'complete'
