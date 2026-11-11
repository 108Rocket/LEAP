"""Portable raw geometry and the recorded RQ4 teacher/evaluation convention."""
from pathlib import Path
import pickle
import operator

import numpy as np
import torch


NORMALIZATION_VERSION = 'norm64-v1'


def checked_packed(points, offsets):
    """Check packed structure without rounding original floating coordinates."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Points must have shape (N, 2)')
    if points.dtype not in (np.dtype('float32'), np.dtype('float64')):
        points = np.asarray(points, dtype=np.float64)
    offsets = np.asarray(offsets)
    if offsets.ndim != 1 or len(offsets) < 2 or offsets.dtype.kind not in 'iu':
        raise ValueError('Offsets must be a one-dimensional integer array')
    if offsets[0] != 0 or offsets[-1] != len(points):
        raise ValueError('Offsets must span exactly the input points')
    if np.any(offsets > len(points)) or np.any(offsets < 0):
        raise ValueError('Offsets outside input points')
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    if np.any(np.diff(offsets) < 2):
        raise ValueError('Each trajectory must contain at least two points')
    return points, offsets


class PackedTrajectories:
    """Original resident points and offsets, with lazy per-trajectory views.

    Arrays are borrowed, never normalized in place. Encoding revalidates mutable
    inputs on every call; this object is not a trusted/prevalidated cache.
    """

    def __init__(self, points, offsets):
        self.points, self.offsets = checked_packed(points, offsets)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        i = operator.index(key)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError('Trajectory index outside packed input')
        return self.points[self.offsets[i]:self.offsets[i+1]]


def checked_pack(raw_paths):
    """Preserve absolute-coordinate precision until normalization; inputs are untouched."""
    raw = [np.asarray(p, np.float64) for p in raw_paths]
    if not raw or any(p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 for p in raw):
        raise ValueError('Provide nonempty finite trajectories, each with at least two 2D points')
    points = np.concatenate(raw)
    if not np.isfinite(points).all():
        raise ValueError('Provide nonempty finite trajectories, each with at least two 2D points')
    lengths = np.fromiter((len(p) for p in raw), np.int64, len(raw))
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths)))
    return points, offsets


def symmetric_distance(matrix, maximum):
    value = torch.as_tensor(np.asarray(matrix), dtype=torch.float32)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError('Expected a square triangular teacher matrix')
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('Teacher maximum must be finite and positive')
    # Stored teachers contain one triangle; retain the original FP32 arithmetic.
    value = (value + value.T) / float(maximum)
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError('Teacher distances must be finite and nonnegative')
    return value


def load_distances(path):
    """Read a TRUSTED local teacher pickle, never an untrusted download."""
    with Path(path).open('rb') as stream:
        train, validation, test, maximum = pickle.load(stream)
    return (
        symmetric_distance(train, maximum),
        symmetric_distance(validation, maximum),
        symmetric_distance(test, maximum),
        float(maximum),
    )


def official_hitting_ratio(predicted, target, predicted_k, target_k):
    """Recorded RQ4 convention: include diagonal, then subtract one self match."""
    predicted_ids = torch.topk(predicted, predicted_k + 1, dim=1, largest=False).indices.cpu().numpy()
    target_ids = torch.topk(target, target_k + 1, dim=1, largest=False).indices.cpu().numpy()
    overlap = sum(
        np.intersect1d(predicted_ids[index], target_ids[index]).size
        for index in range(len(predicted_ids))
    )
    return float((overlap - len(predicted_ids)) / (target_k * len(predicted_ids)))
