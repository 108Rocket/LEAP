"""Shared RQ4 data contract and deterministic, self-excluded retrieval metrics."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def symmetric(matrix):
    a = np.asarray(matrix, np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError('Invalid teacher matrix')
    if np.allclose(a, a.T, rtol=0, atol=1e-12):
        out = a.copy()
    elif not np.count_nonzero(np.tril(a, -1)) or not np.count_nonzero(np.triu(a, 1)):
        out = a + a.T
    else:
        raise ValueError('Teacher is neither symmetric nor triangular')
    if np.any(out < 0) or np.any(np.diag(out) != 0):
        raise ValueError('Invalid teacher costs or diagonal')
    return out


def load_data(geometry, teacher, output):
    geometry, teacher = Path(geometry), Path(teacher)
    with np.load(geometry) as data:
        points = np.asarray(data['points'], np.float64)
        lengths = np.asarray(data['lengths'], np.int64)
    if len(lengths) != 10000 or lengths.sum() != len(points) or lengths.min() < 2:
        raise ValueError('Expected the frozen 7000/1000/2000 raw geometry')
    if lengths.max() > 200 or not np.isfinite(points).all():
        raise ValueError('Invalid coordinates or unsupported length; never silently truncate')
    offsets = np.r_[0, lengths.cumsum()]
    paths = [points[a:b].copy() for a, b in zip(offsets[:-1], offsets[1:])]
    if teacher.suffix == '.json':
        cache = json.loads(teacher.read_text())
        if cache['geometry_sha256'] != sha(geometry):
            raise ValueError('Teacher cache belongs to a different geometry file')
        matrices = [np.load(teacher.parent / p,mmap_mode='r') for p in cache['matrices']]
        maximum = float(cache['maximum'])
    else:
        with teacher.open('rb') as f:
            raw = pickle.load(f)
        matrices = [symmetric(m) for m in raw[:3]]
        maximum = float(raw[3])
    if [len(m) for m in matrices] != [7000, 1000, 2000]:
        raise ValueError('Teacher split mismatch')
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('Invalid official normalization')
    split_paths = [paths[:7000], paths[7000:8000], paths[8000:]]
    hashes = [[hashlib.sha256(p.tobytes()).hexdigest() for p in ps] for ps in split_paths]
    audit = {'geometry': str(geometry.resolve()), 'teacher': str(teacher.resolve()),
             'geometry_sha256': sha(geometry), 'teacher_sha256': sha(teacher),
             'split': [7000, 1000, 2000], 'length_range': [int(lengths.min()), int(lengths.max())],
             'original_points_retained': True, 'teacher_normalization': maximum,
             'normalization_protocol': 'Existing formal RQ4 maximum, not a new train-only claim',
             'split_geometry_hashes': [hashlib.sha256(''.join(h).encode()).hexdigest() for h in hashes],
             'cross_split_geometry_overlap': {'train_val': len(set(hashes[0]) & set(hashes[1])),
                                             'train_test': len(set(hashes[0]) & set(hashes[2]))}}
    save(Path(output) / 'data_audit.json', audit)
    return split_paths, matrices, maximum, audit


def metrics(predicted, target):
    p = np.asarray(predicted, np.float64).copy()
    t = np.asarray(target, np.float64).copy()
    if p.shape != t.shape or not np.isfinite(p).all():
        raise ValueError('Nonfinite prediction or mismatched IDs')
    np.fill_diagonal(p, np.inf)
    np.fill_diagonal(t, np.inf)
    pred = np.argsort(p, axis=1, kind='stable')
    truth = np.argsort(t, axis=1, kind='stable')
    result = {}
    for k in [1, 5, 10, 20, 50]:
        result[f'hr{k}'] = float(np.mean([
            len(set(a[:k]) & set(b[:k])) / k for a, b in zip(pred, truth)]))
    result['r5_at_20'] = float(np.mean([
        len(set(a[:20]) & set(b[:5])) / 5 for a, b in zip(pred, truth)]))
    result['mr'] = float(np.mean([np.flatnonzero(a == b[0])[0] + 1 for a, b in zip(pred, truth)]))
    result['self_excluded'] = True
    result['ties'] = 'distance_then_database_id'
    return result


def source_audit(root):
    return {str(p.relative_to(root)).replace(os.sep, '/'): sha(p)
            for p in Path(root).rglob('*.py')}


@torch.no_grad()
def evaluate(encode, paths, targets, p=2, batch=128):
    vectors = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(0, len(paths), batch):
        vectors.append(encode(paths[i:i+batch]).detach())
    z = torch.cat(vectors)
    torch.cuda.synchronize()
    encoded_seconds = time.perf_counter() - start
    start = time.perf_counter()
    d = torch.cdist(z, z, p=p)
    torch.cuda.synchronize()
    search_seconds = time.perf_counter() - start
    result = metrics(d.cpu().numpy(), targets)
    result.update(encoding_with_preprocessing_seconds=encoded_seconds,
                  dense_embedding_search_seconds=search_seconds,
                  timing_scope='Single diagnostic pass, not a repeated efficiency benchmark')
    return result
