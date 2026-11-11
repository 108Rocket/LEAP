"""Resolve preparation inputs without machine-specific paths or import-time I/O."""
import argparse
import json
from pathlib import Path


STAGES = {
    "prepare_100k": {
        "porto_source": "file", "germany_source": "file", "porto_ids": "file",
        "germany_formal": "file", "porto_asset": "file", "germany_asset": "file",
    },
    "prepare_sdd": {"source_dir": "directory", "archive_manifest": "file"},
    "audit_cities": {
        "source_dir": "directory", "prepared_dir": "directory", "porto_protocol": "file",
    },
    "prepare_city_training": {"source_dir": "directory"},
}


def resolve_paths(config, stage):
    config = Path(config).expanduser().resolve()
    document = json.loads(config.read_text(encoding="utf-8-sig"))
    section = document[stage]
    required = {**STAGES[stage], "output_root": "output"}
    paths = {}
    for key, kind in required.items():
        value = section.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{stage}.{key}: configure a nonempty path")
        path = Path(value).expanduser()
        path = (config.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if kind == "file" and not path.is_file():
            raise FileNotFoundError(f"{stage}.{key}: missing input file {path}")
        if kind == "directory" and not path.is_dir():
            raise FileNotFoundError(f"{stage}.{key}: missing input directory {path}")
        paths[key] = path
    output = paths["output_root"]
    for key, kind in STAGES[stage].items():
        source = paths[key]
        if (output == source or source.is_relative_to(output)
                or (kind == "directory" and output.is_relative_to(source))):
            raise ValueError(f"{stage}.output_root overlaps input {key}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    return paths


def load_paths(stage, argv=None):
    parser = argparse.ArgumentParser(description=f"LEAP {stage}: explicit input paths")
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON paths, resolved relative to this config file")
    parser.add_argument("--check-only", action="store_true",
                        help="Check paths without reading data, creating files, or running computation")
    args = parser.parse_args(argv)
    paths = resolve_paths(args.config, stage)
    if args.check_only:
        print(json.dumps({"stage": stage, "status": "paths_verified",
                          "paths": {k: str(v) for k, v in paths.items()}}, indent=2))
        return None
    return paths
