"""Encoded trajectory records and paper-training initialization assets."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_ASSET = Path(__file__).resolve().parents[1] / 'assets/porto_probes64_directed.json'
ASSET_FORMAT = 'leap.probes.v1'


def load_probe_asset(path=DEFAULT_ASSET):
    asset = json.loads(Path(path).read_text(encoding='utf-8'))
    if asset.get('format') != ASSET_FORMAT:
        raise ValueError('Unsupported geometric probe asset format')
    if asset.get('feature_mode', 'symmetric') not in ('symmetric', 'separated'):
        raise ValueError('Unknown feature mode')
    probes = np.asarray(asset['probes'], dtype=np.float32)
    if probes.ndim != 3 or probes.shape[1:] != (3, 2) or not len(probes):
        raise ValueError('Expected nonempty [probe_count, 3, 2] probe coordinates')
    if not np.isfinite(probes).all():
        raise ValueError('Probe coordinates must be finite')
    if hashlib.sha256(probes.tobytes()).hexdigest() != asset['probes_sha256']:
        raise ValueError('Probe coordinate checksum mismatch')
    return asset


@dataclass(frozen=True)
class TrajectoryIndex:
    coordinates: torch.Tensor
    points: torch.Tensor
    offsets: torch.Tensor
    asset_id: str
    ordered_summary: torch.Tensor | None = None
    ordered_signature: tuple | None = None
    dfd_summary: dict | None = None
    dfd_signature: tuple | None = None
    dtw_probe_summary: dict | None = None
    dtw_probe_signature: tuple | None = None
    ranking_index: object | None = None
    precision_signature: tuple | None = None
