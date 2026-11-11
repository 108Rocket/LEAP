"""Warm wall-clock timing through public APIs, with raw samples and output gates."""
import copy
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from .config import ROOT, checked_file, digest
from .logging import output_path
from .replay import geometry, metrics, reference, save, validate_ids


def prepare_plan(plan, scope="both", query_ids=None, memory_fraction=0.45):
    if scope not in ("batch", "single", "both"):
        raise ValueError("Unknown timing scope")
    if not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be in (0, 1]")
    if plan["self_query"] and scope != "batch":
        raise ValueError("Single sessions do not exclude self IDs; use external queries or --scope batch")
    if plan["retrieval"].get("large_k") or plan["retrieval"].get("full_large_k"):
        raise ValueError("This runner measures ordinary Top-1..64 ranking/Full, not large-k profiles")
    if scope != "batch" and (plan["initial_k"] is not None or plan["candidate_search"] == "full"):
        raise ValueError("Single sessions have no initial_k/scan override; put budgets in the profile")
    if query_ids is not None and scope == "batch":
        raise ValueError("query_ids selects single requests only, not batch membership")
    return dict(plan, timing_schema=2, scope=scope, memory_fraction=memory_fraction,
                single_query_ids_file=str(Path(query_ids).expanduser().resolve()) if query_ids else None,
                runtime="selected public API and configured precision; no hidden accuracy repair",
                timing="warm synchronized wall clock; arithmetic means; see 11_docs/timing.md")


def selected_ids(count, path=None):
    import numpy as np
    ids = np.load(path, allow_pickle=False) if path else np.arange(min(count, 64))
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids)
            or len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= count).any()):
        raise ValueError("Single query IDs must be a nonempty unique integer vector within the query file")
    return ids.astype(np.int64)


def timed(fn, synchronize):
    synchronize()
    start = time.perf_counter_ns()
    value = fn()
    synchronize()
    return value, (time.perf_counter_ns() - start) / 1e9


def alternating(round_id):
    return ("ranking", "full") if round_id % 2 == 0 else ("full", "ranking")


def summarize(records):
    groups = {}
    for row in records:
        if row["round"] >= 0:
            groups.setdefault(row["stage"], []).append(row["seconds"])
    return {name: {"samples": len(values), "mean_seconds": statistics.mean(values),
                   "min_seconds": min(values), "max_seconds": max(values),
                   "stdev_seconds": statistics.stdev(values) if len(values) > 1 else None}
            for name, values in groups.items()}


def hardware():
    def command(args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return "unavailable: " + str(exc)
    return {"utc": datetime.now(timezone.utc).isoformat(),
            "cpu_load": os.getloadavg() if hasattr(os, "getloadavg") else None,
            "gpu": command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,memory.free,utilization.gpu,clocks.sm,power.draw", "--format=csv,noheader"]),
            "compute_processes": command(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"])}


def _round(model, plan, paths, qpaths, selection, round_id, record, verify, synchronize):
    """Separate lifetime ensures old database/session buffers are freed between passes."""
    import numpy as np
    n, same = plan["top_n"], plan["self_query"]

    def measure(stage, fn, query_id=None):
        value, seconds = timed(fn, synchronize)
        record({"round": round_id, "stage": stage, "query_id": query_id, "seconds": seconds})
        return value

    db = measure("encode_database", lambda: model.encode(paths))
    tree = measure("build_index", lambda: model.build_search_index(db))
    resolutions = {}
    if plan["scope"] in ("batch", "both"):
        q = db if same else measure("encode_queries", lambda: model.encode(qpaths))
        functions = {
            "ranking": lambda: model.candidate_indices(q, None if same else db, candidate_k=n,
                search_index=tree, candidate_search=plan["candidate_search"]),
            "full": lambda: model.search_range(q, None if same else db, search_index=tree,
                top_k=n, initial_k=plan["initial_k"], query_batch=plan["query_batch"])["indices"]}
        if model.config.parameter_mode == "auto":
            resolutions["batch_full"] = model.config.resolve(len(paths), n, query_count=len(qpaths)).as_dict()
        for mode in alternating(round_id):
            for _ in range(plan["warmup"]):
                ids, seconds = timed(functions[mode], synchronize)
                record({"round": -1, "pass": round_id, "stage": "warmup_batch_" + mode,
                        "query_id": None, "seconds": seconds})
                verify("batch_" + mode, ids.cpu().numpy())
        for mode in alternating(round_id):
            ids = measure("batch_" + mode, functions[mode])
            verify("batch_" + mode, ids.cpu().numpy())
        del functions, ids, q
    if plan["scope"] in ("single", "both"):
        single = copy.copy(model)
        single.config = replace(model.config, query_profile="single_query")
        session = single.query_session(db, search_index=tree, top_k=n,
            initial_capacity=max(256, max(len(qpaths[i]) for i in selection)))
        resolutions["single"] = getattr(session, "configuration", asdict(single.config))
        raws = [qpaths[i] for i in selection]
        for _ in range(plan["warmup"]):
            _, seconds = timed(lambda: session.warmup(raws, modes=alternating(round_id)), synchronize)
            record({"round": -1, "pass": round_id, "stage": "warmup_single",
                    "query_id": None, "seconds": seconds})
        arrays = {mode: [] for mode in alternating(round_id)}
        # Pair methods at each request, and reverse their order on successive passes.
        for query_id, raw in zip(selection, raws):
            for mode in alternating(round_id):
                ids = measure("single_" + mode, lambda: session.search(raw, mode=mode), int(query_id))
                arrays[mode].append(ids.copy())
        for mode, values in arrays.items():
            verify("single_" + mode, np.concatenate(values))
    return resolutions


def run(plan, output=None):
    import numpy as np
    import numba
    import torch
    from leap import RetrievalConfig
    from leap import LEAP
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("Timing requires a Linux NVIDIA CUDA host")
    torch.cuda.set_device(plan["device"])
    torch.cuda.set_per_process_memory_fraction(plan["memory_fraction"])
    torch.set_num_threads(plan["threads"])
    numba.set_num_threads(min(plan["threads"], numba.config.NUMBA_NUM_THREADS))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    synchronize = lambda: torch.cuda.synchronize(plan["device"])
    out = Path(output) if output else output_path(plan['output_root'], 'timing', plan['job'])
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "plan": plan, "host": platform.node(),
              "gpu": torch.cuda.get_device_name(), "training_seed_count": 1,
              "seed_note": "one supplied checkpoint bundle; timing repeats are not training seeds",
              "reference_used_for_search": False,
              "cpu_threads": {"torch": torch.get_num_threads(), "numba": numba.get_num_threads(),
                              "interop": torch.get_num_interop_threads(),
                              "environment": {key: os.environ.get(key) for key in
                                              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
              "versions": {key: importlib.metadata.version(key) for key in ("numpy", "torch", "triton")},
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for folder in ("01_LEAP/leap",)
                                for p in sorted((ROOT / folder).rglob("*.py"))},
              "bundled_assets_sha256": {p.name: digest(p) for p in sorted((ROOT / "01_LEAP/leap/assets").glob("*.json"))},
              "hardware_before": hardware()}
    save(out / "protocol.json", report)
    records, expected = [], {}
    try:
        db_hash = checked_file(plan["database"])
        query_hash = db_hash if plan["self_query"] else checked_file(plan["queries"])
        paths, mean, std = geometry(plan["database"], packed=True)
        qpaths, qmean, qstd = (paths, mean, std) if plan["self_query"] else geometry(plan["queries"], packed=True)
        if plan["top_n"] > len(paths) - int(plan["self_query"]):
            raise ValueError("Insufficient database for Top-N")
        config = replace(RetrievalConfig(**plan["retrieval"]), query_profile="throughput")
        report["explicit_assets_sha256"] = {key: digest(plan[key]) for key in ("asset", "routing_asset") if plan.get(key)}
        model = LEAP.from_asset(plan["asset"], device=plan["device"], config=config,
                                  routing_asset=plan.get("routing_asset"))
        for actual, required in ((mean, model.mean), (std, model.std), (qmean, model.mean), (qstd, model.std)):
            np.testing.assert_array_equal(actual, required, err_msg="Frozen normalization mismatch")
        selection = selected_ids(len(qpaths), plan["single_query_ids_file"]) if plan["scope"] != "batch" else np.array([], dtype=np.int64)
        report.update(model_asset_id=model.asset_id, retrieval=asdict(config),
                      encoding_input=("original packed points/offsets" if hasattr(paths, "points") else "trajectory list"),
                      database_sha256=db_hash, queries_sha256=query_hash,
                      database_size=len(paths), batch_queries=len(qpaths), single_query_ids=selection.tolist(),
                      single_query_ids_sha256=digest(plan["single_query_ids_file"]) if plan["single_query_ids_file"] else None)
        save(out / "protocol.json", report)

        def verify(stage, ids):
            rows = len(selection) if stage.startswith("single_") else len(qpaths)
            validate_ids(ids, rows, len(paths), plan["top_n"], plan["self_query"])
            if stage in expected:
                np.testing.assert_array_equal(ids, expected[stage], err_msg="Repeated output changed: " + stage)
            else:
                expected[stage] = ids.copy()
                np.save(out / (stage + ".indices.npy"), ids, allow_pickle=False)

        with (out / "samples.jsonl").open("x", encoding="utf-8") as stream, \
                (out / "hardware.jsonl").open("x", encoding="utf-8") as load_stream:
            def record(row):
                records.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            with torch.no_grad():
                # One entire unmeasured pass absorbs first-use compilation/allocation.
                for round_id in range(-1, plan["repeats"]):
                    print("TIMING_PASS", round_id, flush=True)
                    snapshot = {"pass": round_id, "hardware": hardware()}
                    save(out / "STATE.json", snapshot)
                    load_stream.write(json.dumps(snapshot) + "\n")
                    load_stream.flush()
                    resolutions = _round(model, plan, paths, qpaths, selection, round_id,
                                         record, verify, synchronize)
                    gc.collect()
        measured = {}
        if plan["reference"]:
            truth = reference(plan["reference"], len(qpaths), len(paths), plan["top_n"],
                              plan["self_query"], db_hash, query_hash)
            measured = {name: metrics(ids, truth[selection] if name.startswith("single_") else truth)
                        for name, ids in expected.items()}
        comparisons = {}
        if plan["scope"] == "both":
            for mode in ("ranking", "full"):
                comparisons[mode] = {"queries": len(selection), "ordered_equal_queries":
                    int(np.all(expected["single_" + mode] == expected["batch_" + mode][selection], axis=1).sum())}
        report.update(status="complete", summary=summarize(records), resolved_configuration=resolutions,
                      metrics=measured, single_batch_comparison=comparisons,
                      hardware_after=hardware(), repeat_outputs_equal=True,
                      outputs_sha256={p.name: digest(p) for p in sorted(out.glob("*.indices.npy"))})
        save(out / "result.json", report)
        with (out / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["stage", "samples", "mean_seconds", "min_seconds", "max_seconds", "stdev_seconds"])
            writer.writeheader()
            for stage, stats in report["summary"].items():
                writer.writerow(dict(stage=stage, **stats))
        save(out / "ALL_DONE.json", {"success": True, "result_sha256": digest(out / "result.json"),
                                    "samples_sha256": digest(out / "samples.jsonl")})
        print("RESULT", out, flush=True)
        return out
    except BaseException as exc:
        save(out / "FAILED.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
