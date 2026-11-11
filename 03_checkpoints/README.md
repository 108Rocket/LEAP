# Seed 2000 Checkpoints

[First-query example](../11_docs/quick_start.md) / [Training guide](../11_docs/training.md)

The manifest contains seven complete inference bundles: **SDD Geolife Hausdorff**
for `python entrance.py`, plus Hausdorff, DFD and DTW for **100K Porto** and
**SDD Porto**. Each DTW bundle includes its paired summary-network router.

The files are JSON inference checkpoints containing learned probes, normalization,
and, for DTW, learned network weights. They are sufficient to construct the
inference models. Intermediate epochs and optimizer states are omitted.

```bash
python -m leap checkpoints
python -m leap check
```

Load one with `leap.load_checkpoint("sdd.porto.dtw")`. The loader verifies the
parameter checksums, selects a matching distance profile, and supplies the paired
DTW router explicitly, preserving the workload-specific normalization.

Each manifest row identifies the seed, dataset, city, target, file paths and
SHA256 values, together with a summary of its completed training receipt.
For `sdd.porto.dtw`, the radial and ordered filtering probes have an additional
completed training stage (50 sampling rounds, 3,700 updates). Index anchors,
signal probes and router weights are unchanged. Re-encode trajectories and
rebuild indexes when switching checkpoints; cached fields are model-specific.
The manifest retains both training stages. Other bundles are unchanged.
No external-model weights or full datasets are included.
