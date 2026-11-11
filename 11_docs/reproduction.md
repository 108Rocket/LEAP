# Reproduction

## Outputs and Logs

The supplied configuration uses `output_root: "repo:14_logs"`. Training,
evaluation, timing and code tests have separate output categories. Every public
CLI run keeps `configuration.json`, `console.log`, `status.json` and an
`artifacts/` subdirectory together. `--output PATH` selects a new run directory.
Relative overrides are resolved from the current working directory.

Existing runs are never overwritten. Failures during execution preserve the
traceback and failed status. Dry runs create no logs. Generated runs are ignored
by Git, while selected Porto Hausdorff examples are bundled separately by RQ.
See the [log guide](../14_logs/README.md) for file names and examples.

1. Install the editable package from the release root.
2. Obtain and prepare the source data described in `04_data/README.md`.
3. Keep the supplied seed-2000 assets and their recorded normalization.
4. Edit `05_examples/reproduce_seed2000.json` only where your dataset paths differ.
5. Run `python -m leap check --require-data` and inspect a `--dry-run` plan.
6. Run one distance at a time and retain its raw output files.

## Frozen Parameters

`03_checkpoints/manifest.json` lists seven trained models: SDD Geolife Hausdorff
and three distances for 100K Porto and SDD Porto. The seven probe assets and two paired DTW routers
are complete inference checkpoints in JSON form. They
contain the learned parameter values, normalization and internal checksums.
Optimizer state and intermediate training epochs are not part of this release.

`python -m leap check` verifies every published file against its SHA256.
Completion receipts identify each checkpoint's training stages and seed.
Other cities require separately supplied checkpoints and job configurations.

## Accuracy References

Each dataset can declare a `references` mapping in its configuration:

```json
"references": {
  "dtw": {
    "path": "repo:04_data/sdd/porto/references/dtw.top1000.npy",
    "format": "top_ids",
    "database_sha256": "SHA256_OF_DATABASE_NPZ",
    "queries_sha256": "SHA256_OF_QUERIES_NPZ"
  }
}
```

Top-ID rows must use the same query and database order as the geometry files,
with ties resolved by original distance then database ID. Use FP64 reference
distances under the selected target. Hashes refer to your actual geometry files.
Without references, timing can establish repeated-output consistency, but it
cannot calculate retrieval accuracy.

`python -m leap evaluate --job sdd.porto.dtw` uses the
existing public replay API. It reads the reference after search, and records
configuration, geometry hashes, model fingerprints and outputs.

## Scope

The shipped checkpoints reproduce one trained seed. They do not stand for three
independently trained models. Keep training seeds separate from repeated warm
measurements. Hardware and library versions can affect both floating-point ties
and time; compare complete results when changing execution environments.
