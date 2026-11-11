# Quick Start

## Python API

```python
from leap import load_checkpoint

model = load_checkpoint("sdd.porto.dtw", device="cuda:0")
# Each path contains the original ordered 2D coordinates.
database = model.encode(database_paths)
queries = model.encode(query_paths)
tree = model.build_search_index(database)
neighbors = model.search_range(queries, database, search_index=tree)
```

The same encoded database serves Ranking and Full search. For repeated raw
requests, load with `mode="single"` and create `model.query_session(database, ...)`.
See [configuration and API details](configuration.md) before changing k.

## Reproduction

```bash
python -m leap timing --job sdd.porto.hausdorff --dry-run
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m leap timing --job sdd.porto.hausdorff \
  --scope batch
```

Use `sdd.porto.dfd` or `sdd.porto.dtw` for the other distances. The supplied
[job configuration](../05_examples/reproduce_seed2000.json) names all datasets,
checkpoints, routers and profiles explicitly. Paths starting with `repo:`
resolve from the checkout and remain valid when the folder is moved.

To evaluate accuracy, add FP64 reference IDs and geometry hashes to the job
configuration, then run `python -m leap evaluate --job JOB`.
The command prints its run directory under `14_logs/evaluation/`. Numerical
results are in `artifacts/`, beside the saved configuration and console log.
Read the [reproduction guide](reproduction.md) and [timing rules](timing.md)
for the input and output contracts.

100K Hausdorff uses vertex-to-segment distances. SDD Hausdorff uses point-set
distances and nearest/farthest fields. Each city has its own parameters, and
the checkpoint loader chooses the matching profile and DTW router.

FP32 is the default, with an explicit FP64 option. Full search is exact under
exact arithmetic. Floating-point near ties may change the ordered results.
The release supplies one training seed, while the paper reports three-seed means.

[Overview](../README.md)
