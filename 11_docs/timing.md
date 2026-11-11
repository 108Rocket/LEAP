# Timing Rules

Use the same original geometry, trained parameters, target, k, candidate budgets,
CPU thread count, GPU and precision for every paired comparison. The paper's
reference host is NVIDIA L20. A different device produces a different timing
result.

| Timer | Includes | Excludes |
| --- | --- | --- |
| Database encoding | Validation, normalization, packing when needed, CPU/GPU transfer and all encoded fields/summaries | Disk loading, model loading, index construction |
| Index construction | Building the index from the encoded database | Database encoding and query work |
| Batch Ranking | Ranking all preencoded queries and returning GPU IDs | Query encoding, index build, CPU result transfer |
| Batch Full | Initialization, filtering, original-distance refinement and selection for all preencoded queries | Same exclusions as Batch Ranking |
| Single Ranking/Full | One raw CPU query through preparation, encoding, transfer, search and CPU return | Startup, file loading, network and database construction |

The Batch value is the total for all queries, including all 500 queries in the
SDD Porto experiment. Dividing by 500 does not give independent request latency.
The Single value describes an in-process request, not an HTTP round trip.

## Warm Measurements

The timing runner first warms complete search modes outside the timer. CUDA
synchronization surrounds a monotonic wall-clock interval. The default is three
measured rounds. Ranking and Full alternate order across rounds, and single
requests also alternate modes across queries. All successful samples enter the
arithmetic mean. Warmup, JIT and diagnostic work do not enter search timings.

Each pass separately measures raw database encoding and index construction.
Direct packed arrays avoid millions of intermediate Python views, but the
normalization and representation computation still execute. Precomputed encoded
features cannot be substituted for raw input in an encoding timer.

Use `--scope batch` for self-query data, whose self IDs are explicitly excluded.
Use separate query and database files for `--scope both` or `--scope single`.
For exact reproduction of a Single cohort, supply its frozen `--query-ids` file.
The default cohort is the first at most 64 query rows.

## Receipts

The CLI saves each run in `14_logs/timing/` with its configuration and console
log. The following worker files are in that run's `artifacts/` subdirectory.
Output files contain seconds. `samples.jsonl` retains raw observations,
`summary.csv` and `result.json` report arithmetic means, `protocol.json` records
resolved settings and hashes, and `hardware.jsonl` records hardware snapshots
outside timers. A successful receipt hashes the completed output files.

Repeated runs must retain the same ordered IDs within each search mode. Accuracy
references are loaded after search. A failed check produces an explicit failure
record; the runner does not silently reduce k, remove samples or switch targets.

The bundled generic timing CLI handles Top-1 through Top-64. Full Top-1000 and
component controls have separate APIs and input contracts. A sample at one k or
from one ablation cannot be relabeled as another configuration.
