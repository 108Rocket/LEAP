"""Prepare published point-set data without changing IDs or coordinates."""
import gc
import importlib
import json
from pathlib import Path
import pickle

from .files import save, sha256, require_space


class NumericUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        import numpy as np
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(importlib.import_module("numpy.core.multiarray"), name)
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        raise pickle.UnpicklingError(f"Disallowed pickle object: {module}.{name}")


def numeric_load(path):
    with Path(path).open("rb") as stream:
        return NumericUnpickler(stream).load()


def pack(paths):
    import numpy as np
    lengths = np.asarray([len(p) for p in paths], dtype=np.int64)
    if not len(lengths) or (lengths < 2).any():
        raise ValueError("Every trajectory must contain at least two original vertices")
    offsets = np.r_[0, lengths.cumsum()]
    points = np.empty((offsets[-1], 2), np.float64)
    # Bound temporary conversion memory for the 1.6M Python trajectory records.
    for lo in range(0, len(paths), 32768):
        hi = min(lo + 32768, len(paths))
        block = np.concatenate(paths[lo:hi]).astype(np.float64, copy=False)
        if block.shape != (offsets[hi] - offsets[lo], 2) or not np.isfinite(block).all():
            raise ValueError("Expected finite, two-dimensional original vertices")
        points[offsets[lo]:offsets[hi]] = block
        if len(paths) > 100000 and (lo // 32768) % 8 == 0:
            print(f"  Packed {hi:,}/{len(paths):,} trajectories", flush=True)
    return points, offsets


def top_ids(row, k=20):
    import numpy as np
    if row.ndim != 1 or len(row) < k or not np.isfinite(row).all() or (row < 0).any():
        raise ValueError("Invalid published reference distances")
    threshold = np.partition(row, k - 1)[k - 1]
    ids = np.flatnonzero(row <= threshold)
    return ids[np.lexsort((ids, row[ids]))[:k]]


def pointset_hd(a, b):
    import numpy as np
    distances = np.linalg.norm(a[:, None] - b[None, :], axis=-1)
    return float(max(distances.min(0).max(), distances.min(1).max()))


def check_preparation_memory(minimum_gib=24, dataset="Porto"):
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        available = next(int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        if available < minimum_gib * 1024**3:
            raise MemoryError(f"First-time {dataset} preparation requires {minimum_gib} GiB available system RAM")


def prepare(source, dest, asset, *, expected_database=1597579, expected_queries=500):
    import numpy as np
    source, dest = Path(source), Path(dest)
    required = ("base_list", "query_list", "Haus/haus_test_distance_matrix_result")
    print("  Checking original data hashes...", flush=True)
    inputs = {name: sha256(source / name) for name in required}
    model = json.loads(Path(asset).read_text())
    normalization = {k: model[k] for k in ("geometry_mean", "geometry_std")}
    receipt = dest / "PREPARED.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if (record["inputs"] != inputs or record["normalization"] != normalization
                or record["database_size"] != expected_database or record["queries"] != expected_queries):
            raise ValueError("Cached data belongs to different inputs/normalization; use a new --cache-dir")
        for name, digest in record["outputs"].items():
            if sha256(dest / name) != digest:
                raise ValueError("Corrupted prepared data: " + name)
        print("  Reusing verified prepared data.", flush=True)
        return record
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Incomplete data preparation in {dest}; use a fresh --cache-dir")
    if expected_database == 1597579:
        check_preparation_memory()
    require_space(dest, (4 if expected_database == 1597579 else 1) * 1024**3)
    query_paths = numeric_load(source / "query_list")
    if len(query_paths) != expected_queries:
        raise ValueError(f"Expected {expected_queries} published queries, got {len(query_paths)}")
    qp, qo = pack(query_paths)
    np.savez(dest / "queries.npz", points=qp, offsets=qo, **normalization)
    del qp, qo
    print("  Packing original database vertices...", flush=True)
    base = numeric_load(source / "base_list")
    if len(base) != expected_database:
        raise ValueError(f"Expected {expected_database} database trajectories, got {len(base)}")
    points, offsets = pack(base)
    np.savez(dest / "database.npz", points=points, offsets=offsets, **normalization)
    del base
    gc.collect()
    print("  Reading published Hausdorff reference matrix (not a model prediction)...", flush=True)
    matrix = numeric_load(source / "Haus/haus_test_distance_matrix_result")
    if matrix.shape != (expected_queries, expected_database):
        raise ValueError("Published distance matrix does not match query/database IDs")
    truth = np.stack([top_ids(row) for row in matrix])
    checks = []
    for qi in sorted({0, expected_queries // 2, expected_queries - 1}):
        for bi in (int(truth[qi, 0]), int(truth[qi, -1]), expected_database // 2):
            actual = pointset_hd(np.asarray(query_paths[qi]), points[offsets[bi]:offsets[bi + 1]])
            published = float(matrix[qi, bi])
            if not np.isclose(actual, published, rtol=1e-7, atol=1e-9):
                raise ValueError("Published labels failed the independent point-set HD check")
            checks.append({"query": qi, "database": bi, "published": published, "recomputed": actual})
    np.save(dest / "top20.npy", truth, allow_pickle=False)
    record = {"version": 1, "inputs": inputs, "normalization": normalization,
              "database_size": expected_database, "queries": expected_queries,
              "reference": "published SDD point-set Hausdorff matrix; distance/ID tie order",
              "reference_check": checks, "original_vertex_order": True,
              "outputs": {name: sha256(dest / name) for name in ("database.npz", "queries.npz", "top20.npy")}}
    save(receipt, record)
    return record


def prepare_training(source, dest):
    """Same grouped 80/20 split and train-only normalization as the SDD archive."""
    import hashlib
    import numpy as np
    source, dest = Path(source), Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    paths = numeric_load(source / "train_list")
    queries = numeric_load(source / "query_list")

    def geometry_hash(p):
        return hashlib.sha256(np.asarray(p, np.float64).tobytes()).hexdigest()

    query_hashes = {geometry_hash(p) for p in queries}
    groups, excluded = {}, []
    for i, path in enumerate(paths):
        key = geometry_hash(path)
        if key in query_hashes:
            excluded.append(i)
        else:
            groups.setdefault(key, []).append(i)
    order = np.random.default_rng(2000).permutation(len(groups))
    grouped = list(groups.values())
    boundary = int(round(.2 * len(grouped)))
    va = sorted(i for g in order[:boundary] for i in grouped[g])
    tr = sorted(i for g in order[boundary:] for i in grouped[g])
    train = [np.asarray(paths[i], np.float64) for i in tr]
    val = [np.asarray(paths[i], np.float64) for i in va]
    tp, to = pack(train)
    vp, vo = pack(val)
    mean = tp.mean(0)
    scale = float(np.sqrt(np.mean((tp - mean)**2)))
    if scale <= 0 or len(tr) < 256 or len(va) < 20:
        raise ValueError("Insufficient or degenerate training data")
    np.savez(dest / "raw_inputs.npz", train_points=tp, train_offsets=to,
             validation_points=vp, validation_offsets=vo, geometry_mean=mean, geometry_std=np.full(2, scale))
    labels = numeric_load(source / "Haus/haus_train_distance_matrix_result")
    if labels.shape != (len(paths), len(paths)):
        raise ValueError("Training labels have the wrong shape")
    if not np.allclose(labels, labels.T, rtol=0, atol=1e-12):
        if np.count_nonzero(np.tril(labels, -1)) and np.count_nonzero(np.triu(labels, 1)):
            raise ValueError("Training labels are neither symmetric nor triangular")
        labels = labels + labels.T
    if not np.isfinite(labels).all() or (labels < 0).any() or (np.diag(labels) != 0).any():
        raise ValueError("Invalid training labels")
    for ids, name in ((tr, "train"), (va, "validation")):
        np.save(dest / f"haus_{name}.npy", labels[np.ix_(ids, ids)], allow_pickle=False)
    for a, b in ((tr[0], tr[1]), (tr[len(tr)//2], tr[-1])):
        np.testing.assert_allclose(pointset_hd(np.asarray(paths[a]), np.asarray(paths[b])), labels[a, b],
                                   rtol=1e-7, atol=1e-9)
    save(dest / "protocol.json", {"seed": 2000, "train_ids": tr, "validation_ids": va,
        "excluded_query_overlap_train_ids": excluded, "test_labels_used_for_training": False,
        "grouping": "identical ordered FP64 coordinate bytes", "train_fraction": .8,
        "input_sha256": {name: sha256(source / name) for name in
                         ("train_list", "query_list", "Haus/haus_train_distance_matrix_result")}})
    return dest
