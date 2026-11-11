# Test group: SHARED.
# Paper RQ: Shared support for RQ1--RQ5.
# Checks: Input formats, data validation and reproducible dataset loading.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

import ast
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

import leap
from leap.common.data import checked_pack, load_distances, symmetric_distance, official_hitting_ratio
from leap.training.data import load_raw_splits
from leap.common.model import DEFAULT_ASSET, load_probe_asset


def test_standalone_version_and_imports():
    assert leap.__version__ == '1.0.0'
    assert leap.TrajectoryIndex.__module__ == 'leap.common.model'
    assert DEFAULT_ASSET.is_file()
    assert np.asarray(load_probe_asset()['probes']).size == 384
    forbidden = {'model', 'utils', 'configs', 'evaluation', 'trajlab', 'external_models'}
    for path in Path(leap.__file__).parent.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            assert not forbidden.intersection(name.split('.')[0] for name in names), path


@pytest.mark.parametrize('module', ['leap', 'leap.training.train_dtw_probes'])
def test_cli_help_outside_checkout(module, tmp_path):
    result = subprocess.run([sys.executable, '-m', module, '--help'],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--help' in result.stdout


def test_pack_and_normalization_contract(tmp_path):
    paths = [np.zeros((2, 2)), np.ones((3, 2))]
    points, offsets = checked_pack(paths)
    assert points.dtype == np.float64
    assert offsets.tolist() == [0, 2, 5]
    asset = load_probe_asset()
    bundle = {key: np.asarray(asset[key], np.float64) for key in ('geometry_mean', 'geometry_std')}
    for split in ('train', 'validation', 'test'):
        bundle[split+'_points'] = points
        bundle[split+'_offsets'] = offsets
    path = tmp_path/'raw.npz'
    np.savez(path, **bundle)
    splits = load_raw_splits(path, asset)
    assert all(len(split) == 2 for split in splits)
    np.testing.assert_array_equal(splits[2][1], paths[1])
    bundle['geometry_mean'] = bundle['geometry_mean'] + 1
    np.savez(path, **bundle)
    with pytest.raises(ValueError, match='normalization'):
        load_raw_splits(path, asset)


def test_teacher_arithmetic_and_overlap(tmp_path):
    triangular = np.triu(np.arange(36).reshape(6, 6), 1)
    path = tmp_path/'teacher.pkl'
    with path.open('wb') as stream:
        pickle.dump((triangular, triangular, triangular, 100), stream)
    train, validation, test, maximum = load_distances(path)
    expected = torch.tensor(triangular, dtype=torch.float32)
    expected = (expected + expected.T)/100
    assert torch.equal(train, expected)
    assert torch.equal(validation, test)
    assert maximum == 100
    assert official_hitting_ratio(test, test, 5, 5) == 1


@pytest.mark.parametrize('matrix,maximum', [(np.ones((2, 3)), 1), (np.zeros((2, 2)), 0),
    (np.zeros((2, 2)), np.inf), (np.full((2, 2), np.nan), 1)])
def test_invalid_teacher(matrix, maximum):
    with pytest.raises(ValueError):
        symmetric_distance(matrix, maximum)
