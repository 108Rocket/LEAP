"""Pack supplied coordinates into ConvTraj's sequence and raster inputs."""
import numpy as np
import torch


def preprocess(paths, spec):
    lo = np.asarray(spec['minimum'], np.float32)
    span = np.maximum(np.asarray(spec['maximum'], np.float32) - lo, 1.)
    size, length = spec['canvas_size'], spec['max_sequence_length']
    sequence = np.zeros((len(paths), length, 2), np.float32)
    raster = np.zeros((len(paths), 1, size, size), np.float32)
    for i, path in enumerate(paths):
        if len(path) > length:
            raise ValueError('Trajectory exceeds the configured author input length')
        values = np.clip((np.asarray(path, np.float32) - lo) / span, 0., 1.)
        sequence[i, :len(path)] = values
        cells = np.clip(np.floor(values * size).astype(np.int64), 0, size - 1)
        raster[i, 0, cells[:, 0], cells[:, 1]] = 1.
    return torch.from_numpy(sequence), torch.from_numpy(raster)
