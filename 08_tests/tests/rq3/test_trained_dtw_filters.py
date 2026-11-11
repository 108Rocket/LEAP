# Test group: RQ3.
# Checks: trained DTW filter checkpoints and independent ranking parameters.
"""Separate filter stages retain valid assets and reject stale cached fields."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from leap import load_checkpoint
from leap.checkpoints import catalog, verify
from leap.dtw.dtw_probes import validate_asset

ROOT = Path(__file__).parents[3]


def bundle():
    entry = next(r for r in catalog(ROOT) if r['id'] == 'sdd.porto.dtw')
    return entry, verify(entry, ROOT)['asset']


def test_completed_filter_stage():
    entry, asset = bundle()
    stages = entry['training_receipt']['stages']
    stage = stages[-1]
    assert all(s['complete'] for s in stages)
    assert stage['updates'] == stage['sampling_rounds'] * stage['steps_per_round'] == 3700
    assert stage['seed'] == 2000 and not stage['test_used']
    assert stage['input_asset_sha256'] == stages[0]['asset_sha256']
    assert entry['training_receipt']['router_sha256'] == stages[0]['router_sha256']
    validate_asset(asset)
    assert not np.array_equal(asset['radial_probes'], asset['ordered_probes'])
    assert len(asset['anchors']) == 64 and len(asset['signal_probes']) == 16


@pytest.mark.parametrize('name', ['radial', 'ordered'])
def test_filter_checksum_rejects_changed_positions(name):
    _, asset = bundle()
    changed = copy.deepcopy(asset)
    changed[name+'_probes'][0][0] += 0.125
    with pytest.raises(ValueError, match='checksum mismatch'):
        validate_asset(changed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA integration')
def test_filter_change_invalidates_cached_fields():
    from leap import LEAP
    model = load_checkpoint('sdd.porto.dtw', root=ROOT)
    _, original = bundle()
    for name in ('radial', 'ordered'):
        original[name+'_probes'] = original['anchors'][:16]
        original[name+'_probes_sha256'] = hashlib.sha256(
            np.asarray(original[name+'_probes'], np.float32).tobytes()).hexdigest()
    entry = next(r for r in catalog(ROOT) if r['id'] == 'sdd.porto.dtw')
    old = LEAP(original['anchors'], original['geometry_mean'], original['geometry_std'],
               config=model.config, dtw_probes=original, routing_asset=ROOT/entry['router'])
    center = np.asarray(original['geometry_mean'])
    points = np.stack([center, center + 0.001]).astype(np.float32)
    index = old.encode_packed(points, np.array([0, 2], dtype=np.int64))
    with pytest.raises(ValueError, match='different model'):
        model._check_index(index)
