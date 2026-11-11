# Test group: SHARED.
# Paper RQ: Shared support for RQ1--RQ5.
# Checks: Checkpoint integrity, published profiles and portable release paths.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Published checkpoint/profile consistency and movable-checkout behavior."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from leap.checkpoints import ROOT, catalog, resolve, verify
from leap.common.config import RetrievalConfig


def test_published_checkpoint_schemas_and_checksums():
    entries = catalog()
    bundles = [verify(entry) for entry in entries]
    assert sum(len(bundle) for bundle in bundles) == 9
    for entry, bundle in zip(entries, bundles):
        asset = bundle['asset']
        assert len(asset['anchors']) == 64
        assert all(len(anchor) == 2 for anchor in asset['anchors'])
        assert len(asset['geometry_mean']) == len(asset['geometry_std']) == 2
        if entry['distance'] == 'dtw':
            assert asset['format'] == 'leap.dtw-signals.v1'
            assert bundle['router']['format'] == 'leap.dtw-routing.v1'


def test_bundled_leap_asset_loaders():
    from leap.common.model import load_probe_asset
    from leap.dtw.dtw_probes import load_asset
    from leap.dtw.dtw_routing import load_model
    assert load_probe_asset()['format'] == 'leap.probes.v1'
    for name in ('porto', 'sdd_porto'):
        signal = ROOT/'01_LEAP/leap/assets'/(
            'porto_dtw_signal16.json' if name == 'porto' else 'sdd_porto_dtw_signal16.json')
        router = signal.with_name(
            'porto_dtw_routing_e100.json' if name == 'porto' else 'sdd_porto_dtw_routing_aligned.json')
        asset = load_asset(signal)
        model, _ = load_model(asset['geometry_mean'], asset['geometry_std'], 'cpu', router)
        assert not model.training


@pytest.mark.parametrize('entry',catalog(),ids=lambda row:row['id'])
def test_each_inference_bundle_matches_its_profile(entry):
    bundle = verify(entry)
    resolved = resolve(entry['id'])
    cfg = RetrievalConfig.from_json(resolved['profile_path'])
    assert cfg.distance == entry['distance']
    if cfg.distance == 'hausdorff':
        assert cfg.hausdorff_target == entry['target']
    if cfg.distance == 'dtw':
        assert resolved['router_path'] is not None
        assert bundle['asset']['geometry_mean'] == bundle['router']['geometry_mean']
    for mode in ('single','auto'):
        assert resolve(entry['id'],mode=mode)['profile_path'].is_file()


def test_checkpoint_checksum_rejects_tampering():
    entry = copy.deepcopy(catalog()[0])
    entry['asset_sha256'] = '0'*64
    with pytest.raises(ValueError,match='checksum'):
        verify(entry)


def test_catalog_has_porto_workloads_and_geolife_quick_start():
    entries = catalog()
    assert len(entries) == 7
    assert {row['seed'] for row in entries} == {2000}
    assert {(row['benchmark'],row['city']) for row in entries} == {('100K','Porto'),('SDD','Porto'),('SDD','Geolife')}
    assert [row['distance'] for row in entries if row['city'] == 'Geolife'] == ['hausdorff']
    assert {row['distance'] for row in entries} == {'hausdorff','dfd','dtw'}
    assert sum('router' in row for row in entries) == 2
    published = {row[key] for row in entries for key in ('asset','router') if key in row}
    present = {p.relative_to(ROOT).as_posix() for p in (ROOT/'03_checkpoints/seed2000').rglob('*.json')}
    assert present == published


@pytest.mark.parametrize('job',['100k.porto.dtw','sdd.porto.dtw'])
def test_dry_run_from_another_directory(tmp_path, job):
    result = subprocess.run([sys.executable,'-m','leap','timing','--config',
        str(ROOT/'05_examples/reproduce_seed2000.json'),'--job',job,'--scope','batch','--dry-run'],
        cwd=tmp_path,text=True,capture_output=True,check=True)
    request = json.loads(result.stdout)
    folder = ROOT/'03_checkpoints/seed2000'/job.split('.')[0]/'porto/dtw'
    assert Path(request['asset']) == folder/'asset.json'
    assert Path(request['routing_asset']) == folder/'router.json'
    assert Path(request['profile']).is_file()


def test_public_configuration_comes_from_owning_module():
    from leap import RetrievalConfig as public
    from leap.common.config import RetrievalConfig as implementation
    assert public is implementation
    assert implementation.__module__ == 'leap.common.config'


def test_numbered_layout_keeps_python_package_names():
    import leap
    from leap.reproduction import config
    assert Path(leap.__file__).resolve().parent == ROOT/'01_LEAP/leap'
    assert config.ROOT == ROOT
    assert (ROOT/'07_external_models/external_models/install_sources.py').is_file()
    assert (ROOT/'08_tests/tests/rq5').is_dir()


def test_all_example_jobs_resolve_numbered_paths():
    from leap.reproduction import config
    path, cfg = config.read(ROOT/'05_examples/reproduce_seed2000.json')
    assert set(cfg['jobs']) == {row['id'] for row in catalog()}
    assert set(cfg['datasets']) == {'100k.porto','sdd.porto','sdd.geolife'}
    for name in cfg['jobs']:
        request = config.plan(path, cfg, name)
        for key in ('profile', 'asset', 'routing_asset'):
            if request.get(key):
                assert Path(request[key]).is_file()
                assert Path(request[key]).is_relative_to(ROOT)
        assert Path(request['database']['path']).is_relative_to(ROOT/'04_data')


def test_100k_preparation_sources_are_shipped():
    for name in ('09_tools/data_preparation/archive/prepare_100k.py',
                 '09_tools/pack_data.py',
                 '07_external_models/external_models/learned/TrajCL/upstream/utils/preprocessing_porto.py'):
        assert (ROOT/name).is_file()
