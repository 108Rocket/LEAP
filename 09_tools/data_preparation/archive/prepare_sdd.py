"""Safely unpack numeric SDD objects and verify the published target distances."""
import gc
import json
import os
from pathlib import Path
import pickle
import time
from paths import load_paths

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

REPORT = PREP = None


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def progress(stage, **kw):
    value = dict(stage=stage, time=time.time(), **kw)
    save(REPORT / 'preparation_state.json', value)
    print(json.dumps(value), flush=True)


class NumericUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray'):
            if name == '_reconstruct':
                return np.core.multiarray._reconstruct
            if name == 'scalar':
                return np.core.multiarray.scalar
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        raise pickle.UnpicklingError('Forbidden pickle global: ' + module + '.' + name)


def numeric_load(path):
    with path.open('rb') as stream:
        return NumericUnpickler(stream).load()


def exact(a, b, metric):
    costs = np.linalg.norm(a[:, None] - b[None, :], axis=2)
    if metric == 'haus':
        return max(costs.min(0).max(), costs.min(1).max())
    if metric == 'segment_haus':
        def directed(x, y):
            edge = np.diff(y, axis=0)
            rel = x[:, None] - y[None, :-1]
            alpha = np.clip((rel * edge).sum(-1) / np.maximum((edge * edge).sum(-1), 1e-30), 0, 1)
            return np.linalg.norm(rel - alpha[..., None] * edge, axis=-1).min(1).max()
        return max(directed(a, b), directed(b, a))
    previous = np.full(len(b) + 1, np.inf)
    previous[0] = 0
    for row in costs:
        current = np.full(len(b) + 1, np.inf)
        for j, local in enumerate(row, 1):
            predecessor = min(previous[j], previous[j-1], current[j-1])
            current[j] = max(local, predecessor) if metric == 'dfd' else local + predecessor
        previous = current
    return previous[-1]


def stable_top(row, k):
    subset = np.argpartition(row, k-1)[:k]
    pool = np.flatnonzero(row <= row[subset].max())
    return pool[np.lexsort((pool, row[pool]))[:k]]


def pack(source, name):
    progress('loading_paths', split=name, source=str(source))
    paths = numeric_load(source)
    lengths = np.fromiter((len(x) for x in paths), dtype=np.int64, count=len(paths))
    assert (lengths >= 2).all(), (name, 'unsupported short trajectory')
    offsets = np.r_[np.int64(0), lengths.cumsum()]
    PREP.mkdir(parents=True, exist_ok=True)
    np.save(PREP / (name + '_offsets.npy'), offsets, allow_pickle=False)
    points = np.lib.format.open_memmap(PREP / (name + '_points.npy'), mode='w+',
                                      dtype=np.float64, shape=(int(offsets[-1]), 2))
    for i, path in enumerate(paths):
        array = np.asarray(path, dtype=np.float64)
        assert array.shape == (int(lengths[i]), 2) and np.isfinite(array).all(), (name, i, array.shape)
        points[offsets[i]:offsets[i+1]] = array
    points.flush()
    metadata = dict(count=len(paths), points=int(offsets[-1]), source=str(source),
                    length_min=int(lengths.min()), length_max=int(lengths.max()),
                    length_quantiles=np.quantile(lengths, [.5, .9, .99, .999]).tolist(),
                    coordinate_min=points.min(0).tolist(), coordinate_max=points.max(0).tolist())
    del paths, points
    gc.collect()
    return metadata


def main(argv=None):
    global REPORT, PREP
    paths = load_paths("prepare_sdd", argv)
    if paths is None:
        return
    REPORT = paths["output_root"] / "reports"
    PREP = paths["output_root"] / "prepared/Porto"
    archive_sha = json.loads(paths["archive_manifest"].read_text())["sha256"]
    if (not isinstance(archive_sha, str) or len(archive_sha) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in archive_sha)):
        raise ValueError("archive_manifest.sha256 must be the verified archive SHA256")
    trajectory = paths["source_dir"]
    inventory = {str(p.relative_to(trajectory)): p.stat().st_size
                 for p in trajectory.rglob('*') if p.is_file() and not p.name.endswith('.part')}
    save(REPORT / 'TRAJECTORY_FILES.json', inventory)
    porto = trajectory / 'Porto'
    splits = {}
    for name in ('train', 'query', 'base'):
        matches = [p for p in porto.iterdir() if p.is_file() and p.name.lower() in (name, name + '_list', name + '_seq_list')]
        assert len(matches) == 1, (name, [str(x) for x in porto.iterdir()])
        splits[name] = pack(matches[0], name)
    save(REPORT / 'SPLITS.json', splits)
    qp = np.load(PREP / 'query_points.npy', mmap_mode='r')
    qo = np.load(PREP / 'query_offsets.npy', mmap_mode='r')
    bp = np.load(PREP / 'base_points.npy', mmap_mode='r')
    bo = np.load(PREP / 'base_offsets.npy', mmap_mode='r')
    protocol = dict(dataset='Airchang/SDD', city='Porto', splits=splits, training=False,
                    original_vertices_preserved=True, row_mapping='published query_list and base_list order',
                    archive_sha256=archive_sha, metrics={})
    for metric in ('haus', 'dfd', 'dtw'):
        files = [p for p in porto.rglob('*test*distance*') if p.is_file() and not p.name.endswith('.part')
                 and (metric in p.name.lower() or metric in str(p.parent).lower())]
        assert len(files) == 1, (metric, [str(p) for p in files])
        source = files[0]
        progress('loading_teacher', metric=metric, bytes=source.stat().st_size)
        teacher = numeric_load(source)
        assert isinstance(teacher, np.ndarray) and teacher.ndim == 2 and teacher.dtype.kind == 'f'
        nq, nb = teacher.shape
        assert nb == splits['base']['count'] and nq <= splits['query']['count'], (teacher.shape, splits)
        # Prefix rows are accepted only after first/middle/last independent checks below.
        for row in teacher:
            assert np.isfinite(row).all() and (row >= 0).all()
        np.save(PREP / (metric + '_teacher.npy'), teacher, allow_pickle=False)
        truth = np.array([stable_top(row, 128) for row in teacher])
        np.save(PREP / (metric + '_top128.npy'), truth, allow_pickle=False)
        rng = np.random.default_rng(2000)
        pairs = [(0,0), (nq-1,nb-1), (nq//2,nb//2)]
        pairs += list(zip(rng.integers(nq, size=12).tolist(), rng.integers(nb, size=12).tolist()))
        pairs += [(q, int(truth[q, rank])) for q in (0, 1, nq//2, nq-1) for rank in (0, 19)]
        checks = []
        for q, d in pairs:
            a, b = qp[qo[q]:qo[q+1]], bp[bo[d]:bo[d+1]]
            calculated = float(exact(a, b, metric))
            record = dict(query=q, database=d, calculated=calculated, stored=float(teacher[q,d]),
                          absolute_error=abs(calculated - teacher[q,d]), lengths=[len(a),len(b)])
            if metric == 'haus':
                record['vertex_to_segment'] = float(exact(a, b, 'segment_haus'))
            checks.append(record)
        record = dict(source=str(source), shape=list(teacher.shape), dtype=str(teacher.dtype),
                      query_prefix_rows=nq, unused_queries=splits['query']['count']-nq,
                      definition={'haus':'symmetric point-set Hausdorff', 'dfd':'discrete Frechet',
                                  'dtw':'unconstrained Euclidean-sum DTW'}[metric],
                      checks=checks, matched=all(np.isclose(x['calculated'], x['stored'],rtol=1e-7,atol=1e-9) for x in checks))
        protocol['metrics'][metric] = record
        save(REPORT / 'PROTOCOL.json', protocol)
        assert record['matched'], (metric, 'Author truth definition or row alignment mismatch')
        progress('teacher_verified', metric=metric, shape=record['shape'], checks=len(checks))
        del teacher, truth
        gc.collect()
    progress('preparation_complete')
    save(REPORT / 'PREPARATION_DONE.json', dict(time=time.time(), metrics=list(protocol['metrics'])))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if REPORT is not None:
            save(REPORT / 'PREPARATION_FAILED.json', dict(time=time.time(), error=repr(exc)))
        raise
