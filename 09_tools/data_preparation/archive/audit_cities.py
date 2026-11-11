"""Read-only SDD inventory and city-training contract audit on the L20 host."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from paths import load_paths
import prepare_sdd as reader

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

OUT = SOURCE = PREP = PROTOCOL = None


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)


def digest(paths):
    return [hashlib.sha256(np.asarray(p, dtype='<f8').tobytes()).hexdigest() for p in paths]


def describe(paths):
    lengths = np.array([len(p) for p in paths])
    assert all(np.asarray(p).shape == (len(p), 2) and np.isfinite(p).all() for p in paths)
    return dict(count=len(paths), minimum=int(lengths.min()), maximum=int(lengths.max()),
                points=int(lengths.sum()), median=float(np.median(lengths)),
                unique=len(set(digest(paths))))


def run(city):
    folder = SOURCE / city
    train = reader.numeric_load(folder / 'train_list')
    query = reader.numeric_load(folder / 'query_list')
    th, qh = digest(train), digest(query)
    result = dict(city=city, train=describe(train), query=describe(query),
                  train_query_exact_overlap=len(set(th) & set(qh)), metrics={})
    if city == 'Porto':
        bp = np.load(PREP / 'base_points.npy', mmap_mode='r')
        bo = np.load(PREP / 'base_offsets.npy', mmap_mode='r')
        base_count = len(bo) - 1
        base = None
        original = json.loads(PROTOCOL.read_text())
        result['base'] = original['splits']['base']
    else:
        base = reader.numeric_load(folder / 'base_list')
        base_count = len(base)
        result['base'] = describe(base)
        bh = digest(base)
        result['train_base_exact_overlap'] = len(set(th) & set(bh))
        result['query_base_exact_overlap'] = len(set(qh) & set(bh))
    for metric, subfolder in [('haus', 'Haus'), ('dfd', 'DFD'), ('dtw', 'DTW')]:
        tm = reader.numeric_load(folder / subfolder / (metric + '_train_distance_matrix_result'))
        assert tm.shape == (len(train), len(train)), (city, metric, tm.shape)
        assert np.isfinite(tm).all() and (tm >= 0).all()
        symmetric = np.allclose(tm, tm.T, rtol=1e-10, atol=1e-12)
        triangular = not np.count_nonzero(np.tril(tm, -1)) or not np.count_nonzero(np.triu(tm, 1))
        assert symmetric or triangular
        full = tm if symmetric else tm + tm.T
        checks = []
        rng = np.random.default_rng(2000)
        for a, b in rng.integers(0, len(train), size=(8, 2)):
            got = reader.exact(np.asarray(train[a]), np.asarray(train[b]), metric)
            checks.append(dict(a=int(a), b=int(b), actual=got, stored=float(full[a, b])))
        assert all(np.isclose(x['actual'], x['stored'], rtol=1e-7, atol=1e-9) for x in checks), (city, metric, checks)
        if city == 'Porto':
            test_record = original['metrics'][metric]
            test_shape = test_record['shape']
            test_checks = test_record['checks']
            assert test_record['matched']
        else:
            test = reader.numeric_load(folder / subfolder / (metric + '_test_distance_matrix_result'))
            test_shape = list(test.shape)
            assert test.shape[1] == base_count and test.shape[0] <= len(query)
            assert np.isfinite(test).all() and (test >= 0).all()
            test_checks = []
            pairs = [(0, 0), (len(test)-1, base_count-1), (len(test)//2, base_count//2)]
            for q in (0, len(test)//2, len(test)-1):
                pairs.append((q, int(test[q].argmin())))
            for q, b in pairs:
                actual = reader.exact(np.asarray(query[q]), np.asarray(base[b]), metric)
                test_checks.append(dict(query=q, database=b, actual=actual, stored=float(test[q,b])))
            assert all(np.isclose(x['actual'], x['stored'], rtol=1e-7, atol=1e-9) for x in test_checks), (city, metric, test_checks)
            del test
        result['metrics'][metric] = dict(train_shape=list(tm.shape), train_storage='symmetric' if symmetric else 'triangular',
            test_shape=test_shape, train_checks=checks, test_checks=test_checks, truth_verified=True,
            definition={'haus': 'symmetric point-set Euclidean Hausdorff', 'dfd': 'discrete Frechet Euclidean',
                        'dtw': 'unconstrained Euclidean-sum DTW'}[metric])
        save(city + '_AUDIT.json', result)
        print(json.dumps(dict(city=city, metric=metric, train_shape=list(tm.shape), test_shape=test_shape, verified=True)), flush=True)
    return result


def main(argv=None):
    global OUT, SOURCE, PREP, PROTOCOL
    paths = load_paths("audit_cities", argv)
    if paths is None:
        return
    OUT, SOURCE = paths["output_root"], paths["source_dir"]
    PREP, PROTOCOL = paths["prepared_dir"], paths["porto_protocol"]
    try:
        all_cities = {c: run(c) for c in ('Chengdu', 'Geolife', 'Porto')}
        save('CITY_AUDIT_DONE.json', dict(time=time.time(), cities=all_cities,
            scope='data verification only; no training or accuracy run started'))
    except Exception as error:
        save('CITY_AUDIT_FAILED.json', dict(error=repr(error), time=time.time()))
        raise


if __name__ == '__main__':
    main()
