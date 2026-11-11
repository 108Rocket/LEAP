"""Freeze external queries before any new target distances are computed."""
import hashlib
import json
import pickle
from pathlib import Path
import time
import numpy as np
import pandas as pd
from paths import load_paths

def digest(p):
    return hashlib.sha256(np.asarray(p, np.float64).tobytes()).hexdigest()


def save_paths(path, paths, source_ids):
    lengths = np.array([len(p) for p in paths], np.int64)
    np.savez(path, points=np.concatenate(paths), lengths=lengths, source_ids=source_ids)


def main(argv=None):
    paths = load_paths("prepare_100k", argv)
    if paths is None:
        return
    DATA = paths["output_root"] / "data"
    OUT = paths["output_root"] / "reports"
    DATA.mkdir(parents=True, exist_ok=False)
    OUT.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20260911)
    manifest = {'selection_seed': 20260911, 'queries': 1000, 'database_sizes': [2000, 20000, 100000],
                'selection_uses_distances': False, 'test_tuning': False, 'cities': {}}
    for city in ['porto', 'germany']:
        start = time.time()
        if city == 'porto':
            path = paths["porto_source"]
            frame = pd.read_pickle(path)
            formal = [np.asarray(p, np.float64) for p in frame.iloc[200000:210000].merc_seq]
            old_ids = json.loads(paths["porto_ids"].read_text())
            db_ids = np.asarray(old_ids['compact_iloc'], np.int64)
            database = [np.asarray(p, np.float64) for p in frame.iloc[db_ids].merc_seq]
            excluded = set(old_ids['raw_float64_sha256']) | {digest(p) for p in formal}
            pool = rng.permutation(np.arange(210000, 300000))
            query_ids, queries = [], []
            for i in pool:
                p = np.asarray(frame.iloc[i].merc_seq, np.float64)
                h = digest(p)
                if h not in excluded:
                    query_ids.append(int(i)); queries.append(p); excluded.add(h)
                if len(queries) == 1000:
                    break
            asset_path = paths["porto_asset"]
        else:
            path = paths["germany_source"]
            frame = pd.read_pickle(path)
            old = paths["germany_formal"]
            with np.load(old) as f:
                points, lengths = f['points'], f['lengths']
            offsets = np.r_[0, lengths.cumsum()]
            formal = [points[a:b].astype(np.float64) for a, b in zip(offsets[:-1], offsets[1:])]
            excluded = {digest(p) for p in formal}
            pool = rng.permutation(len(frame))
            query_ids, queries, database, db_ids = [], [], [], []
            for i in pool:
                p = np.asarray(frame.iloc[i].merc_seq, np.float64)
                h = digest(p)
                if h in excluded:
                    continue
                excluded.add(h)
                if len(queries) < 1000:
                    queries.append(p); query_ids.append(int(i))
                else:
                    database.append(p); db_ids.append(int(i))
                if len(database) == 100000:
                    break
            asset_path = paths["germany_asset"]
        assert len(queries) == 1000 and len(database) == 100000
        qhash, dhash, fhash = map(lambda ps: [digest(p) for p in ps], [queries, database, formal])
        assert not set(qhash) & (set(dhash) | set(fhash))
        assert len(set(qhash)) == 1000
        asset = json.loads(asset_path.read_text())
        (OUT / f'{city}.asset.json').write_text(json.dumps(asset, indent=2))
        save_paths(DATA / f'{city}.queries.npz', queries, np.asarray(query_ids))
        save_paths(DATA / f'{city}.database.npz', database, np.asarray(db_ids))
        save_paths(DATA / f'{city}.formal.npz', formal, np.arange(10000))
        record = {'source': str(path), 'source_rows': len(frame), 'query_source_ids': query_ids,
                  'query_sha256': qhash, 'database_hashes_sha256': hashlib.sha256(''.join(dhash).encode()).hexdigest(),
                  'query_database_hash_overlap': 0, 'query_formal_hash_overlap': 0,
                  'asset_sha256': hashlib.sha256(asset_path.read_bytes()).hexdigest(),
                  'normalization': 'frozen existing per-city formal asset, not refitted',
                  'query_lengths': [min(map(len, queries)), max(map(len, queries)), float(np.mean(list(map(len, queries))))],
                  'database_lengths': [min(map(len, database)), max(map(len, database)), float(np.mean(list(map(len, database))))],
                  'scope': 'New external-query draw; source city is not unseen. Porto database was previously inspected.',
                  'seconds': time.time() - start}
        manifest['cities'][city] = record
        print(json.dumps({city: {k: v for k, v in record.items() if k not in ['query_source_ids', 'query_sha256']}}), flush=True)
        del frame, formal, queries, database
    (DATA / 'FROZEN.json').write_text(json.dumps(manifest, indent=2))
    (OUT / 'FROZEN.json').write_text(json.dumps(manifest, indent=2))
    print('DATA_FROZEN', flush=True)


if __name__ == '__main__':
    main()
