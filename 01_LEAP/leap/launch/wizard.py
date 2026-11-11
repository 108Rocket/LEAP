"""Consent-based startup, using the public timing runner after preparation."""
import argparse
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import uuid
import urllib.error

from .files import (MEMBERS, RELEASE_REVISION, SDD_BYTES, SDD_SHA256, SDD_URL,
                    GEOLIFE_ARCHIVE, GEOLIFE_BYTES, GEOLIFE_URL, GEOLIFE_MEMBERS,
                    download, download_geolife, extract_geolife, verify_geolife,
                    extract_porto, save, sha256, require_space)

ROOT = Path(__file__).resolve().parents[3]
DATASETS = {"geolife": {"city": "Geolife", "database": 9386, "queries": 1000,
                        "ram_gib": 4, "vram_gib": 4},
            "porto": {"city": "Porto", "database": 1597579, "queries": 500,
                      "ram_gib": 24, "vram_gib": 8}}
LOGO = r"""
  L      EEEEE    A    PPPP            *
  L      E       A A   P   P       *--/
  L      EEEE   AAAAA  PPPP    *--/
  L      E      A   A  P    *-/
  LLLLL  EEEEE  A   A  P   /
                         +------------>
  Learned Probes for Exact Trajectory Similarity Search
"""


class Declined(Exception):
    """The user chose not to perform a required action."""


def choose_mode(args, argv):
    if args.dataset == "geolife":
        return
    explicit = any(a == "--mode" or a.startswith("--mode=") for a in argv)
    if args.yes or args.checkpoint or explicit or not sys.stdin.isatty():
        return
    try:
        print("Use the bundled seed2000 checkpoint? Y = run now, N = train 100 epochs first [Y/n]", flush=True)
        answer = input().strip().lower()
    except EOFError as exc:
        raise Declined("No input received; no training or download was started.") from exc
    if answer in ("n", "no"):
        args.mode = "train"
    elif answer not in ("", "y", "yes"):
        raise Declined("Please answer Y or N. No training or download was started.")


def confirm(question, yes=False):
    if yes:
        print(question + " [Y: --yes]", flush=True)
        return
    try:
        # A full line also reaches the parent's line-buffered bootstrap log.
        print(question + " [y/N]", flush=True)
        answer = input().strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        raise Declined("Stopped without making the requested change. Rerun when ready.")


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return self.console.isatty()


def command(args, extra_env=None):
    print("  Running: " + " ".join(map(str, args)), flush=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "01_LEAP") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update(extra_env or {})
    with subprocess.Popen(list(map(str, args)), cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1, env=environment) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        status = process.wait()
    if status:
        raise RuntimeError(f"Command exited with status {status}")


def dependencies():
    limits = {"numpy": ((1, 24), (3,)), "pandas": ((2, 2), (3,)),
              "torch": ((2, 5), (2, 6)), "triton": ((3, 1), (3, 2)),
              "numba": ((0, 60), (0, 68)), "gdown": ((6, 5), (6, 6))}
    problems = []
    for name, (low, high) in limits.items():
        try:
            raw = importlib.metadata.version(name)
            version = tuple(int(p) for p in raw.split("+")[0].split(".")[:2])
            if not low <= version < high:
                problems.append(name + "=" + raw)
        except (importlib.metadata.PackageNotFoundError, ValueError):
            problems.append(name + " missing/unsupported")
    return problems


def resume_command(args, python):
    command = [python, ROOT / "entrance.py", "--dataset", args.dataset, "--mode", args.mode, "--device", args.device,
               "--scope", args.scope, "--log-root", args.log_root]
    for name in ("source_dir", "archive", "cache_dir", "checkpoint", "wheelhouse"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), value])
    if args.yes:
        command.append("--yes")
    return command


def check_package_network():
    # A parent timeout also bounds DNS resolution, which socket timeouts may not.
    probe = ("import sys, urllib.request; "
             "r=urllib.request.urlopen(sys.argv[1], timeout=10); r.read(1); r.close()")
    try:
        subprocess.run([sys.executable, "-c", probe, "https://download.pytorch.org/whl/cu124/torch/"],
                       timeout=20, check=True, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot reach the official PyTorch download site within 20 seconds. "
                           "Check network access or use --wheelhouse /path/to/linux-wheels; "
                           "no system environment was changed.") from exc


def cache_path(args, asset=None):
    if args.cache_dir is not None:
        return args.cache_dir
    base = ROOT / "04_data/sdd" / args.dataset / "entrance_cache"
    chosen = asset or args.checkpoint
    if chosen is not None and (args.mode == "train" or args.checkpoint):
        return base / sha256(chosen)
    return base


def environment(args):
    print("[1/5] Python, dependencies and GPU", flush=True)
    if platform.system() != "Linux":
        raise RuntimeError("This CUDA/Triton workflow requires Linux (or Linux in WSL2). "
                           "Run python entrance.py there with an NVIDIA GPU; no Windows packages were installed.")
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("Use Python 3.10, 3.11 or 3.12.")
    cache = cache_path(args)
    if args.mode == "train" or not (cache / "PREPARED.json").is_file():
        from .data import check_preparation_memory
        settings = DATASETS[args.dataset]
        check_preparation_memory(settings["ram_gib"], settings["city"])
    missing = dependencies()
    if missing:
        print("  Dependencies: " + ", ".join(missing), flush=True)
        target = ROOT / ".venv-entrance/bin/python"
        managed = Path(sys.prefix).resolve() == target.parent.parent.resolve()
        if managed and os.environ.get("LEAP_BOOTSTRAPPED") == "1":
            raise RuntimeError("Installed environment is still incompatible. Inspect its install log before retrying.")
        if target.exists() and not managed:
            print("  Reusing .venv-entrance.", flush=True)
            command(resume_command(args, target))
            return False
        confirm("Create/repair .venv-entrance and install the pinned CUDA 12.4 PyTorch stack (several GB)?", args.yes)
        if args.wheelhouse:
            if not args.wheelhouse.is_dir() or not any(args.wheelhouse.glob("*.whl")):
                raise ValueError("--wheelhouse must contain compatible Linux Python wheels")
            package_source = ["--no-index", "--find-links", args.wheelhouse]
        else:
            print("  Checking access to the official PyTorch package site...", flush=True)
            check_package_network()
            package_source = ["--index-url", "https://download.pytorch.org/whl/cu124"]
        if not target.exists():
            command([sys.executable, "-m", "venv", ROOT / ".venv-entrance"])
        command([target, "-m", "pip", "install", "--timeout", "30", "--retries", "2", "torch==2.5.1", *package_source])
        offline = package_source if args.wheelhouse else []
        command([target, "-m", "pip", "install", "--timeout", "30", "--retries", "2",
                 "-r", ROOT / "10_requirements/reproduction.txt", *offline])
        command([target, "-m", "pip", "install", "--timeout", "30", "--retries", "2",
                 "setuptools>=68", "wheel", *offline])
        command([target, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", ROOT, *offline])
        command(resume_command(args, target), {"LEAP_BOOTSTRAPPED": "1"})
        return False
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA. Check the NVIDIA driver and GPU access; no CPU timing is substituted.")
    torch.cuda.set_device(args.device)
    free, total = torch.cuda.mem_get_info()
    print(f"  {torch.cuda.get_device_name()}; free VRAM {free / 1024**3:.1f}/{total / 1024**3:.1f} GiB", flush=True)
    required_vram = DATASETS[args.dataset]["vram_gib"]
    if free < required_vram * 1024**3:
        raise RuntimeError(f"At least {required_vram} GiB free VRAM is required for this workflow.")
    if free < total * .85:
        confirm("This GPU already has substantial allocated memory; timing may be affected. Continue?", args.yes)
    return True


def checkpoint(args, run):
    print("[2/5] Public seed2000 checkpoint", flush=True)
    manifest = json.loads((ROOT / "03_checkpoints/manifest.json").read_text())
    city = DATASETS[args.dataset]["city"]
    item = next(x for x in manifest["checkpoints"] if x["id"] == f"sdd.{args.dataset}.hausdorff")
    if args.checkpoint:
        asset = args.checkpoint
        digest = sha256(asset)
        if digest != item["asset_sha256"]:
            receipt = json.loads((asset.parent / "ALL_DONE.json").read_text())
            model = json.loads(asset.read_text())
            if (receipt.get("status") != "trained" or receipt.get("epochs") != 100
                    or receipt.get("asset_sha256") != digest or model.get("seed") != 2000
                    or model.get("city") != city or model.get("representation") != "pointset-nearfar-v1"):
                raise ValueError(f"Custom checkpoint must have this launcher's completed {city} training receipt")
        print("  Verified supplied checkpoint: " + str(asset), flush=True)
        return asset
    asset = ROOT / item["asset"]
    if not asset.exists():
        confirm(f"Recover/download the missing SDD {city} Hausdorff checkpoint (SHA256-verified)?", args.yes)
        asset = run / "downloaded_asset.json"
        recovered = False
        try:
            blob = subprocess.run(["git", "show", f"{RELEASE_REVISION}:{item['asset']}"], cwd=ROOT,
                                  capture_output=True, timeout=15, check=True).stdout
            import hashlib
            if hashlib.sha256(blob).hexdigest() == item["asset_sha256"]:
                asset.write_bytes(blob)
                recovered = True
                print("  Recovered verified checkpoint from local Git history.", flush=True)
        except (OSError, subprocess.SubprocessError):
            pass
        if not recovered:
            url = f"https://raw.githubusercontent.com/DongXinfzu/LEAP/{RELEASE_REVISION}/{item['asset']}"
            try:
                download(url, asset, item["asset_sha256"])
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise RuntimeError("Checkpoint download is not publicly accessible. Restore 03_checkpoints from "
                                       "your original source archive or an authorized Git clone; no credentials are requested.") from exc
                raise
    if sha256(asset) != item["asset_sha256"]:
        raise ValueError("Checkpoint hash mismatch. It will not be overwritten or used.")
    print("  Checkpoint SHA256 verified.", flush=True)
    return asset


def source_data(args):
    if args.dataset == "geolife":
        return geolife_source(args)
    print("[3/5] SDD Porto data", flush=True)
    source = args.source_dir or ROOT / "04_data/sdd/porto/source"
    required = set(MEMBERS) if args.mode == "train" else set(MEMBERS) - {"train_list", "Haus/haus_train_distance_matrix_result"}
    missing = [name for name in required if not (source / name).is_file()]
    if not missing:
        print("  Using " + str(source), flush=True)
        return source
    if args.source_dir:
        raise FileNotFoundError("Missing files in --source-dir: " + ", ".join(sorted(missing)))
    archive = args.archive or ROOT / "04_data/downloads/SequenceDistanceDataset.tar.gz"
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        print("  Official source: https://huggingface.co/datasets/Airchang/SDD", flush=True)
        confirm(f"Download the official SDD archive ({SDD_BYTES / 1e9:.2f} GB; allow about 45 GB disk space)?", args.yes)
        partial = archive.with_name(archive.name + ".part")
        existing = min(partial.stat().st_size, SDD_BYTES) if partial.exists() else 0
        require_space(archive.parent, 45 * 10**9 - existing)
        download(SDD_URL, archive, SDD_SHA256, SDD_BYTES)
    else:
        print("  Verifying existing SDD archive SHA256...", flush=True)
        if sha256(archive) != SDD_SHA256:
            raise ValueError("SDD archive hash mismatch")
    confirm("Extract Porto Hausdorff inputs (about 8.5 GB) into " + str(source) + "?", args.yes)
    return extract_porto(archive, source, training=args.mode == "train")


def geolife_source(args):
    print("[3/5] SDD Geolife data", flush=True)
    source = args.source_dir or ROOT / "04_data/sdd/geolife/source"
    if all((source / name).is_file() for name in GEOLIFE_MEMBERS):
        print("  Verifying original Geolife input SHA256 values...", flush=True)
        return verify_geolife(source)
    if args.source_dir:
        raise FileNotFoundError("Incomplete Geolife --source-dir; point it to the extracted Geolife folder")
    archive = args.archive or ROOT / "04_data/downloads" / GEOLIFE_ARCHIVE
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        print("  Public data: " + GEOLIFE_URL, flush=True)
        confirm(f"Download the Geolife Hausdorff ZIP ({GEOLIFE_BYTES / 1e6:.0f} MB, no Google login)?", args.yes)
        download_geolife(archive)
    confirm("Extract verified Geolife data (300 MB) into " + str(source) + "?", args.yes)
    return extract_geolife(archive, source)


def evaluate(args, run, source, asset):
    from .data import prepare
    from leap.reproduction.config import plan
    from leap.reproduction.timing import prepare_plan, run as measure
    print("[4/5] Prepare original vertices and independent published references", flush=True)
    cache = cache_path(args, asset)
    settings = DATASETS[args.dataset]
    city, nq = settings["city"], settings["queries"]
    prepared = prepare(source, cache, asset, expected_database=settings["database"], expected_queries=nq)
    hashes = prepared["outputs"]
    ref = {"path": str(cache / "top20.npy"), "format": "top_ids", "sha256": hashes["top20.npy"],
           "database_sha256": hashes["database.npz"], "queries_sha256": hashes["queries.npz"]}
    cfg = {"version": 1, "device": args.device, "threads": 4, "output_root": str(run),
           "datasets": {"sdd_" + args.dataset: {"self_query": False,
               "database": {"path": str(cache / "database.npz"), "sha256": hashes["database.npz"]},
               "queries": {"path": str(cache / "queries.npz"), "sha256": hashes["queries.npz"]},
               "references": {"hausdorff": ref}}},
           "jobs": {args.dataset + "_hd": {"dataset": "sdd_" + args.dataset, "metric": "hausdorff",
                "profile": "repo:02_configs/hausdorff/batch_query_pointset_hd.json",
                "asset": str(asset), "top_n": 20, "warmup": 1, "repeats": 3}}}
    path = run / "config.json"
    save(path, cfg)
    print("[5/5] Warm up, check repeated outputs, measure and evaluate HR@20", flush=True)
    print(f"  Batch = all {nq:,} queries, resident search only.", flush=True)
    if args.scope == "both":
        print("  Single = complete requests for the first 64 queries.", flush=True)
    print("  Download, preparation, transfer and JIT are outside search timing. One checkpoint, not three training seeds.", flush=True)
    output = measure(prepare_plan(plan(path, cfg, args.dataset + "_hd"), scope=args.scope,
                                  memory_fraction=.9), output=run / "evaluation")
    result = json.loads((output / "result.json").read_text())
    for mode, comparison in result["single_batch_comparison"].items():
        if comparison["ordered_equal_queries"] != comparison["queries"]:
            raise ValueError("Single/batch output mismatch: " + mode + "; inspect evaluation/result.json")
    lines = ["", f"SDD {city} | point-set Hausdorff | k=20 | seed2000",
             f"Database: {result['database_size']:,} | Batch queries: {result['batch_queries']}",
             f"Mode          HR@20 ({nq} queries)    Batch total (ms)    Single request (ms)"]
    for name, label in (("ranking", "Ranking"), ("full", "Full")):
        hr = result["metrics"]["batch_" + name]["HR@20"]
        batch = result["summary"]["batch_" + name]["mean_seconds"] * 1000
        single = result["summary"].get("single_" + name)
        single_text = f"{single['mean_seconds'] * 1000:.3f}" if single else "not measured"
        lines.append(f"{label:<13} {hr:>8.2%}              {batch:>10.3f}          {single_text:>10}")
    lines += ["", f"HR@20 = overlap with the published top-20, averaged over {nq:,} queries.",
              "FP32 search is compared with supplied SDD labels; near-ties may differ.",
              "Times are arithmetic means of 3 warm repeats, with no outliers removed.",
              "Encoding and index construction are recorded separately in evaluation/result.json.",
              "Measured on " + result["gpu"] + "; times are not guaranteed to match the paper.",
              "Log directory: " + str(run)]
    summary = "\n".join(lines) + "\n"
    print(summary, flush=True)
    (run / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    save(run / "ALL_DONE.json", {"status": "complete", "result_sha256": sha256(output / "result.json"),
                               "checkpoint_sha256": sha256(asset), "mode": args.mode})


def parser():
    ap = argparse.ArgumentParser(description="Guided Hausdorff reproduction. Default: SDD Geolife, trained seed2000 checkpoint.")
    ap.add_argument("--dataset", choices=("geolife", "porto"), default="geolife")
    ap.add_argument("--mode", choices=("checkpoint", "train"), default="checkpoint")
    ap.add_argument("--checkpoint", type=Path, help="Bundled checkpoint or final_asset.json with a completed training receipt")
    ap.add_argument("--wheelhouse", type=Path, help="Offline Linux/Python-compatible dependency wheels; install only after confirmation")
    ap.add_argument("--source-dir", type=Path, help="Published selected-city directory containing base_list, query_list and Haus/")
    ap.add_argument("--archive", type=Path, help="Geolife ZIP, or official SequenceDistanceDataset.tar.gz for --dataset porto")
    ap.add_argument("--cache-dir", type=Path, help="Generated packed inputs; never overwrite source data")
    ap.add_argument("--log-root", type=Path, default=ROOT / "14_logs/entrance")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scope", choices=("batch", "both"), default="both")
    ap.add_argument("--yes", action="store_true", help="Explicitly consent to required downloads, installation and selected training")
    ap.add_argument("--check-only", action="store_true", help="Print readiness without downloading, installing or running CUDA")
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser().parse_args(argv)
    if args.mode == "train" and args.checkpoint:
        parser().error("--checkpoint loads a trained model; it cannot be combined with --mode train")
    if args.dataset == "geolife" and args.mode == "train":
        parser().error("Geolife entrance currently uses the verified seed2000 checkpoint. From-scratch training is not part of this quick start.")
    for name in ("source_dir", "archive", "cache_dir", "log_root", "checkpoint", "wheelhouse"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    run = args.log_root / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    old_out, old_err = sys.stdout, sys.stderr
    with (run / "console.log").open("x", encoding="utf-8") as stream:
        sys.stdout, sys.stderr = Tee(old_out, stream), Tee(old_err, stream)
        try:
            print(LOGO, flush=True)
            print("Logs: " + str(run), flush=True)
            if not args.check_only:
                choose_mode(args, argv)
            save(run / "START.json", {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
            if args.check_only:
                state = {"platform": platform.system(), "python": platform.python_version(),
                         "dependencies": dependencies(), "mode": args.mode,
                         "dataset": args.dataset, "dimensions": DATASETS[args.dataset],
                         "runtime_requires": "Linux, Python 3.10-3.12, CUDA NVIDIA GPU, selected SDD city inputs",
                         "executed": False}
                print(json.dumps(state, indent=2), flush=True)
                save(run / "CHECK_ONLY.json", state)
                return 0
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[key] = "4"
            if not environment(args):
                save(run / "DELEGATED.json", {"status": "child completed; see child log path above"})
                return 0
            if args.mode == "train":
                print("[2/5] New seed2000 model: training selected; bundled checkpoints remain unchanged.", flush=True)
            asset = checkpoint(args, run) if args.mode == "checkpoint" else None
            source = source_data(args)
            if args.mode == "train":
                confirm("Train new Porto nearest/farthest probes from scratch: seed2000, 50+50 epochs, then save and evaluate?", args.yes)
                command([sys.executable, "-m", "leap.launch.train", "--source-dir", source,
                         "--output", run / "training", "--device", args.device])
                asset = run / "training/final_asset.json"
            evaluate(args, run, source, asset)
            return 0
        except (Declined, KeyboardInterrupt) as exc:
            print("\nStopped: " + str(exc), flush=True)
            save(run / "STOPPED.json", {"reason": str(exc), "resume": "Rerun the same command; verified downloads are reusable."})
            return 130 if isinstance(exc, KeyboardInterrupt) else 2
        except Exception as exc:
            traceback.print_exc()
            save(run / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
            print("\nNot completed. Details: " + str(run / "console.log"), flush=True)
            return 1
        finally:
            sys.stdout, sys.stderr = old_out, old_err
