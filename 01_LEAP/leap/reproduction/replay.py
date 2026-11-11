"""Frozen public-API replay. Reference data is read only after both rankings."""
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

from .config import checked_file, digest, ROOT
from .logging import output_path


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def geometry(spec, *, packed=False):
    import numpy as np
    path = Path(spec["path"])
    if path.suffix == ".pkl":
        if spec.get("trusted_pickle") is not True:
            raise ValueError("Pickle geometry requires trusted_pickle=true; prefer NPZ")
        from .pickle_compat import CompatibleUnpickler
        with path.open("rb") as stream:
            value = CompatibleUnpickler(stream).load()
        paths, mean, std = value["paths"], value["mean"], value["std"]
    else:
        with np.load(path, allow_pickle=False) as value:
            points = value["points"]
            offsets = value["offsets"]
            if (points.ndim != 2 or points.shape[1] != 2 or offsets.ndim != 1 or
                    offsets.dtype.kind not in "iu" or len(offsets) < 2 or offsets[0] != 0 or
                    offsets[-1] != len(points) or (np.diff(offsets) < 2).any()):
                raise ValueError("Invalid packed geometry: points[N,2], integer offsets[M+1]")
            if packed:
                from leap.common.data import PackedTrajectories
                paths = PackedTrajectories(points, offsets)
            else:
                paths = [points[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
            mean, std = value["geometry_mean"], value["geometry_std"]
    if not len(paths):
        raise ValueError("Empty geometry")
    arrays = [paths.points] if packed and hasattr(paths, 'points') else paths
    for p in arrays:
        a = np.asarray(p)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
            raise ValueError("Each trajectory needs at least two finite original points")
    return paths, np.asarray(mean), np.asarray(std)


def validate_ids(ids, rows, size, n, same):
    import numpy as np
    ids = np.asarray(ids)
    if ids.shape != (rows, n) or ids.dtype.kind not in "iu":
        raise ValueError("Top-N IDs have wrong shape or dtype")
    if ((ids < 0) | (ids >= size)).any():
        raise ValueError("Top-N contains invalid database IDs")
    if n > 1 and (np.diff(np.sort(ids, axis=1), axis=1) == 0).any():
        raise ValueError("Top-N contains duplicate IDs")
    if same and (ids == np.arange(rows)[:, None]).any():
        raise ValueError("Top-N includes self")


def reference(spec, rows, size, n, same, db_hash, query_hash):
    import numpy as np
    if spec.get("database_sha256") != db_hash:
        raise ValueError("Reference database SHA256/order does not match geometry")
    if not same and spec.get("queries_sha256") != query_hash:
        raise ValueError("Reference query SHA256/order does not match geometry")
    if spec.get("sha256"):
        checked_file(spec)
    if spec.get("complete") and not Path(spec["complete"]).is_file():
        raise ValueError("Reference matrix has no completion marker")
    value = np.load(spec["path"], mmap_mode="r", allow_pickle=False)
    if spec["format"] == "top_ids":
        if value.ndim != 2 or value.shape[0] != rows or value.shape[1] < n:
            raise ValueError("Reference Top-N has insufficient rows or columns")
        ids = np.array(value[:, :n])
    else:
        if value.shape != (rows, size) or value.dtype.kind != "f":
            raise ValueError("Reference matrix shape/dtype mismatch")
        ids = np.empty((rows, n), dtype=np.int64)
        order = np.arange(size)
        for i in range(rows):
            row = np.array(value[i])
            check = np.delete(row, i) if same else row
            if not np.isfinite(check).all():
                raise ValueError("Reference contains nonfinite off-diagonal values")
            if same:
                row[i] = np.inf
            ids[i] = np.lexsort((order, row))[:n]
    validate_ids(ids, rows, size, n, same)
    return ids


def metrics(predicted, truth):
    import numpy as np
    n = predicted.shape[1]
    result = {}
    for k in sorted({1, n} | ({5} if n >= 5 else set()) | ({20} if n >= 20 else set())):
        hit = sum(int((predicted[lo:lo+512, :k, None] == truth[lo:lo+512, None, :k]).any(2).sum())
                  for lo in range(0, len(truth), 512))
        result[f"HR@{k}"] = hit / (len(truth) * k)
        result[f"missing@{k}"] = len(truth) * k - hit
    if n >= 20:
        result["R5@20"] = sum(int((predicted[lo:lo+512, :20, None] == truth[lo:lo+512, None, :5]).any(1).sum())
                              for lo in range(0, len(truth), 512)) / (len(truth) * 5)
    result["ordered_equal_queries"] = int(np.all(predicted == truth, axis=1).sum())
    return result


def run(plan, output=None):
    import numpy as np
    import torch
    from leap import LEAP
    from leap import RetrievalConfig

    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("Replay requires a Linux NVIDIA CUDA host; the workstation can use ui --connect")
    torch.cuda.set_device(plan["device"])
    torch.set_num_threads(plan["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(output) if output else output_path(plan['output_root'], 'evaluation', plan['job'])
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "plan": plan, "created": datetime.now(timezone.utc).isoformat(),
              "host": platform.node(), "gpu": torch.cuda.get_device_name(),
              "versions": {k: importlib.metadata.version(k) for k in ("numpy", "torch", "triton")},
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for folder in ("01_LEAP/leap",)
                                for p in sorted((ROOT / folder).rglob("*.py"))},
              "reference_used_for_search": False}
    save(out / "protocol.json", report)

    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        return value, time.perf_counter() - start

    try:
        db_hash = checked_file(plan["database"])
        query_hash = db_hash if plan["self_query"] else checked_file(plan["queries"])
        paths, mean, std = geometry(plan["database"])
        qpaths, qmean, qstd = (paths, mean, std) if plan["self_query"] else geometry(plan["queries"])
        config = RetrievalConfig(**plan["retrieval"])
        candidate_search = plan.get("candidate_search", "tree" if config.distance == "hausdorff" else "auto")
        model = LEAP.from_asset(path=plan["asset"], device=plan["device"], config=config,
                                  routing_asset=plan.get("routing_asset"))
        for actual, expected in ((mean, model.mean), (std, model.std), (qmean, model.mean), (qstd, model.std)):
            np.testing.assert_array_equal(actual, expected, err_msg="Frozen normalization differs from geometry")
        n, same = plan["top_n"], plan["self_query"]
        if n > len(paths) - int(same):
            raise ValueError("Not enough database trajectories for requested Top-N")
        with torch.no_grad():
            db, db_s = timed(lambda: model.encode(paths))
            q, q_s = (db, 0.0) if same else timed(lambda: model.encode(qpaths))
            tree, build_s = timed(lambda: model.build_search_index(db))
            functions = {
                "initial": lambda: model.candidate_indices(q, None if same else db, candidate_k=n,
                                                           search_index=tree, candidate_search=candidate_search),
                "complete": lambda: model.search_range(q, None if same else db, search_index=tree,
                                                       top_k=n, initial_k=plan["initial_k"],
                                                       query_batch=plan["query_batch"])["indices"]}
            expected, times, warm = {}, {"initial": [], "complete": []}, {}
            for name, fn in functions.items():
                print("WARMUP", name, flush=True)
                warm[name] = []
                for _ in range(plan["warmup"]):
                    ids, elapsed = timed(fn)
                    array = ids.cpu().numpy()
                    validate_ids(array, len(qpaths), len(paths), n, same)
                    if name in expected:
                        np.testing.assert_array_equal(array, expected[name])
                    expected[name] = array.copy()
                    warm[name].append(elapsed)
            for repeat in range(plan["repeats"]):
                for name in (("initial", "complete") if repeat % 2 == 0 else ("complete", "initial")):
                    ids, seconds = timed(functions[name])
                    np.testing.assert_array_equal(ids.cpu().numpy(), expected[name])
                    times[name].append(seconds)
                    print(json.dumps({"repeat": repeat, "mode": name, "seconds": seconds}), flush=True)
            for name, ids in expected.items():
                np.save(out / (name + ".indices.npy"), ids, allow_pickle=False)
        # Ranking and all timing passes finish before opening reference data.
        measured = {}
        if plan["reference"]:
            truth = reference(plan["reference"], len(qpaths), len(paths), n, same, db_hash, query_hash)
            measured = {name: metrics(ids, truth) for name, ids in expected.items()}
        report.update(status="complete", model_asset_id=model.asset_id, retrieval=asdict(config),
                      candidate_search=candidate_search,
                      database_sha256=db_hash, queries_sha256=query_hash, database_size=len(paths),
                      queries=len(qpaths), encoding_database_s=db_s, encoding_queries_s=q_s,
                      index_build_s=build_s, warmup_seconds=warm, resident_seconds=times,
                      mean_seconds={name: statistics.mean(values) for name, values in times.items()},
                      median_seconds={name: statistics.median(values) for name, values in times.items()},
                      metrics=measured, scope="public-API replay; not a replay of paper-only FP64 near-tie repair")
        save(out / "result.json", report)
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["job", "metric", "mode", "queries", "database_size",
                                                        "HR@1", "HR@5", "HR@20", "R5@20", "resident_seconds"])
            writer.writeheader()
            for name in functions:
                writer.writerow(dict(job=plan["job"], metric=plan["metric"], mode=name, queries=len(qpaths),
                                     database_size=len(paths), resident_seconds=report["mean_seconds"][name],
                                     **{k: measured.get(name, {}).get(k, "") for k in ("HR@1", "HR@5", "HR@20", "R5@20")}))
        save(out / "ALL_DONE.json", {"status": "complete", "result": "result.json"})
        print("RESULT", out, flush=True)
        return out
    except Exception as exc:
        save(out / "FAILED.json", {"error": str(exc), "type": type(exc).__name__})
        raise
