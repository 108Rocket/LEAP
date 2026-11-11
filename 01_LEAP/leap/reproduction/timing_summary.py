"""Equal-weight aggregation of independently supplied checkpoint timing runs."""
import argparse
import json
from pathlib import Path
import statistics

from .config import digest


def aggregate(runs):
    if len({seed for seed, _ in runs}) != len(runs) or not runs:
        raise ValueError("Supply distinct training-seed labels")
    results, contract, paths = [], None, set()
    for seed, directory in runs:
        directory = Path(directory).expanduser().resolve()
        if directory in paths:
            raise ValueError("The same timing directory cannot stand for multiple training seeds")
        paths.add(directory)
        report = json.loads((directory / "result.json").read_text())
        receipt = json.loads((directory / "ALL_DONE.json").read_text())
        if (not receipt.get("success") or report.get("status") != "complete"
                or not report.get("repeat_outputs_equal")
                or receipt["result_sha256"] != digest(directory / "result.json")
                or receipt["samples_sha256"] != digest(directory / "samples.jsonl")):
            raise ValueError("Incomplete or changed timing result: " + str(directory))
        plan = report["plan"]
        current = {key: report[key] for key in ("database_sha256", "queries_sha256", "database_size",
                   "batch_queries", "single_query_ids", "retrieval", "source_sha256", "gpu", "versions")}
        current["plan"] = {key: plan[key] for key in ("timing_schema", "scope", "top_n", "initial_k",
                            "query_batch", "candidate_search", "warmup", "repeats", "threads", "memory_fraction")}
        current["samples"] = {key: stats["samples"] for key, stats in report["summary"].items()}
        current["cpu_threads"] = report.get("cpu_threads")
        if contract is not None and current != contract:
            raise ValueError("Timing protocols, membership, software or hardware differ")
        contract = current
        results.append({"seed": seed, "directory": str(directory), "result_sha256": receipt["result_sha256"],
                        "model_asset_id": report["model_asset_id"], "summary": report["summary"]})
    means = {stage: [row["summary"][stage]["mean_seconds"] for row in results]
             for stage in results[0]["summary"]}
    return {"training_seed_labels": [row["seed"] for row in results], "runs": results,
            "independent_training_not_verified": True,
            "note": "Caller supplies independent training provenance. Repeated timing is not retraining.",
            "same_model_asset_ids": len({row["model_asset_id"] for row in results}) < len(results),
            "aggregation": "arithmetic mean within each checkpoint, then equal-weight mean across checkpoints",
            "summary": {stage: {"seed_means_seconds": values, "mean_seconds": statistics.mean(values),
                         "seed_stdev_seconds": statistics.stdev(values) if len(values) > 1 else None}
                        for stage, values in means.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", required=True, metavar=("SEED", "DIRECTORY"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate([(int(seed), path) for seed, path in args.run])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
