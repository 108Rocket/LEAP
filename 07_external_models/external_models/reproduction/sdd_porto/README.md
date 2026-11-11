# SDD Interfaces

The encoder interface imports the original author classes from the pinned
snapshots under each `07_external_models/external_models/learned/<model>/upstream/`. Supply trusted checkpoints compatible
with those classes. State dictionaries are loaded strictly. No attention,
padding, position-encoding or other model replacement is installed.

## Checkpoint Inputs

```text
assets/
  Porto.<distance>.ConvTraj/best.pt, inference.json
  Porto.<distance>.SIMformer/best.pt, inference.json
  Porto.<distance>.TSMini/checkpoint.pt, statistics.json
  Porto.<distance>.TrajAgg/checkpoint.pt, statistics.json
  Porto.<distance>.TrajCL/checkpoint.pt
  Porto.<distance>.MovSemCL/checkpoint.pt
  city_grid/cellspace.pkl, embeddings.pkl
```

`<distance>` is `hausdorff`, `discret_frechet` or `dtw`. The checkpoint containers
use `model` for ConvTraj/SIMformer/TrajAgg, `encoder` for TSMini, and `encoder_q`
plus `trajsimi` for TrajCL/MovSemCL. `inference.json` records the training
normalization (`mean`/`std` for SIMformer, `raster_spec` for ConvTraj).
`statistics.json` contains training coordinate bounds and, for TrajAgg, mean/std.
The adapter converts the SDD coordinate units by multiplying by 111320. It does
not resample, truncate, extend position embeddings or repair incompatible inputs.

## Encoding

Expose exactly one GPU. Follow the author requirements in its own environment.

```bash
CUDA_VISIBLE_DEVICES=0 python 07_external_models/external_models/reproduction/sdd_porto/encode_benchmark.py \
  --model ConvTraj --metric hausdorff \
  --points /path/to/points.npy --offsets /path/to/offsets.npy \
  --asset-root /path/to/assets --mode total --output runs/convtraj-total
```

Packed XY points and offsets follow the [data format](../../../../04_data/README.md).
Use `--mode gpu` in a separate profiling process. Complete encoding includes
CPU input preparation and transfer. GPU activity excludes them. These runs
are separate from the [LEAP search timing protocol](../../../../11_docs/timing.md).
Three measurements of one checkpoint are timing repeats, not training seeds.

## TraSS

[NativeBench.java](../../exact/TraSS/NativeBench.java) reads packed data and calls the author
Java/HBase implementation. Follow the author's build and service setup. Run its
`regression` and `gate` checks before timing. If the original source fails a
check, report that failure without changing its algorithm or collecting an
accepted timing. The adapter does not patch the original parser or filter.

[External Models](../../README.md) | [Overview](../../../../README.md)
