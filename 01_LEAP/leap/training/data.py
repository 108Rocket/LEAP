"""Packed training split validation shared by the paper learning stages."""
import numpy as np


def load_raw_splits(path, asset):
    with np.load(path, allow_pickle=False) as bundle:
        for key in ('geometry_mean','geometry_std'):
            if not np.array_equal(bundle[key],np.asarray(asset[key],dtype=np.float64)):
                raise ValueError('Raw bundle normalization differs from model asset: '+key)
        splits = []
        for split in ('train','validation','test'):
            points,offsets = bundle[split+'_points'],bundle[split+'_offsets']
            if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                    or offsets.ndim != 1 or len(offsets) < 2 or offsets.dtype.kind not in 'iu'
                    or offsets[0] != 0 or offsets[-1] != len(points) or np.any(np.diff(offsets) < 2)):
                raise ValueError('Invalid packed raw split: '+split)
            splits.append([points[a:b].copy() for a,b in zip(offsets[:-1],offsets[1:])])
    return splits
