"""Portable paths and explicit reproduction plans; no GPU imports."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = Path(__file__).with_name("example.json")


def read(path):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Expected an entrance configuration with version=1")
    unknown = set(value) - {"version", "device", "threads", "output_root", "ui", "datasets", "jobs", "training"}
    if unknown:
        raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
    return path, value


def resolve(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty path")
    if value.startswith("repo:"):
        path = (ROOT / value[5:]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("repo: path must remain inside the checkout")
        return path
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base).parent / path).resolve()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_spec(base, value):
    if isinstance(value, str):
        value = {"path": value}
    if not isinstance(value, dict) or "path" not in value:
        raise ValueError("Expected a file specification with path")
    result = dict(value, path=str(resolve(base, value["path"])))
    return result


def checked_file(spec):
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = digest(path)
    expected = spec.get("sha256")
    if expected and actual != expected:
        raise ValueError("SHA256 mismatch: " + str(path))
    return actual


def integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def plan(path, cfg, name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Invalid job name")
    jobs = cfg.get("jobs", {})
    if name not in jobs:
        raise ValueError("Unknown job: " + name)
    job = jobs[name]
    if set(job) - {"dataset", "metric", "profile", "asset", "routing_asset", "top_n", "warmup", "repeats", "initial_k", "query_batch", "candidate_search"}:
        raise ValueError("Unknown fields in job " + name)
    metric = job["metric"]
    if metric not in ("hausdorff", "dfd", "dtw"):
        raise ValueError("Unsupported distance: " + str(metric))
    candidate_search = job.get("candidate_search", "tree" if metric == "hausdorff" else "auto")
    if candidate_search not in ("auto", "full", "tree"):
        raise ValueError("candidate_search must be auto, full or tree")
    if metric == "dfd" and candidate_search == "full":
        raise ValueError("DFD initial ranking requires auto or tree")
    data = cfg["datasets"][job["dataset"]]
    if type(data.get("self_query")) is not bool:
        raise ValueError("Dataset must explicitly specify self_query")
    same = data["self_query"]
    if same and data.get("queries"):
        raise ValueError("Self-query datasets must not have a separate queries file")
    database = file_spec(path, data["database"])
    queries = database if same else file_spec(path, data["queries"])
    ref = data.get("references", {}).get(metric)
    ref = file_spec(path, ref) if ref else None
    if ref and ref.get("complete"):
        ref["complete"] = str(resolve(path, ref["complete"]))
    if ref and ref.get("format") not in ("top_ids", "matrix"):
        raise ValueError("Reference format must be top_ids or matrix")
    profile = resolve(path, job.get("profile", f"repo:02_configs/{metric}/{metric}.json"))
    retrieval = json.loads(profile.read_text(encoding="utf-8"))
    if retrieval.get("distance") != metric:
        raise ValueError("Profile distance differs from job distance")
    asset = resolve(path, job["asset"]) if job.get("asset") else None
    routing_asset = resolve(path, job["routing_asset"]) if job.get("routing_asset") else None
    if routing_asset and (metric != "dtw" or retrieval.get("dtw_routing") != "learned"):
        raise ValueError("routing_asset requires a learned DTW profile")
    n = integer(job.get("top_n", 20), "top_n (batch API)", 1, 64)
    initial = job.get("initial_k")
    if initial is not None:
        integer(initial, "initial_k", n, 64)
    batch = job.get("query_batch")
    if batch is not None:
        integer(batch, "query_batch", 1, 100000)
    return dict(job=name, dataset=job["dataset"], metric=metric, database=database,
                queries=queries, self_query=same, reference=ref, profile=str(profile),
                retrieval=retrieval, asset=str(asset) if asset else None,
                routing_asset=str(routing_asset) if routing_asset else None, top_n=n,
                initial_k=initial, query_batch=batch, candidate_search=candidate_search,
                warmup=integer(job.get("warmup", 1), "warmup", 1, 20),
                repeats=integer(job.get("repeats", 3), "repeats", 1, 100),
                device=cfg.get("device", "cuda:0"),
                threads=integer(cfg.get("threads", 4), "threads", 1, 64),
                output_root=str(resolve(path, cfg.get("output_root", "repo:14_logs"))),
                runtime="public FP32 API; empirical pruning slack",
                timing="resident batch search; excludes encode/build/JIT/reference/I-O")
