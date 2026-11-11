# Configuration and API

Browse the [distance profiles](../02_configs/README.md) for all configuration files.

`load_checkpoint("sdd.porto.hausdorff")` loads point-set HD with nearest/farthest
fields. `load_checkpoint("100k.porto.hausdorff")` loads vertex-to-segment HD.
The city and distance are part of the checkpoint identifier. Do not mix a
checkpoint's normalization with another city's data or a different target.

`mode="batch"` chooses the adopted throughput profile, `mode="single"` chooses
the persistent-query profile, and `mode="auto"` chooses the size/k budget policy.
Automatic configuration does not train or tune against query answers.

`search_engine="original"` retains the original implementation and is the default.
`search_engine="fast"` selects the optional accelerated candidate search. Separate
[fast profiles and usage](search_engine.md) preserve the existing config files.
The engine choice also works with `parameter_mode="auto"`.

`dynamic_threshold` defaults to `false`. Set it to `true` to tighten the Full
search radius using newly refined neighbors. Initial exact distances are reused
by default with either setting; this switch never disables that reuse.
The optimized dynamic paths support FP32 Top-20 batches: point-set near/far HD
with 50 initial candidates, DFD with at most 64, and optimized ragged DTW with
at most 64. Single-query sessions, other k values, and unsupported profiles keep
their existing Full path. Ranking and its budgets are unchanged.

```python
from dataclasses import replace
model.config = replace(model.config, dynamic_threshold=True)
```

## Search

`model.encode(paths)` accepts ordered original-coordinate arrays or
`PackedTrajectories(points, offsets)`. Keep source coordinates in float64 until
encoding. `model.build_search_index(database)` constructs the reusable index.
`candidate_indices` returns the candidate ranking and `search_range` performs
the complete geometric search, including candidates outside that ranking.

For single requests, construct one `model.query_session(database, ...)` and
warm the required lengths and modes before timing. Session buffers and CUDA
graphs are reused between requests. See the original method signatures and
tests for optional return values and self-query ID handling.

## Result Counts

The generic timing CLI accepts Top-1 through Top-64. Larger Top-k experiments
use the separate `full_large_k` configuration and public API. The configs retain
the tested large-k tree path and candidate capacities; k and s are independent
of tree leaf size. Never silently truncate candidates to fit a buffer.

The published proportional-budget experiment uses
`s_H=ceil(2.5*k)`, `s_F=ceil(1.6*k)`, `s_W=k`,
`p_F=ceil(3.2*k)` and `p_W=ceil(6.4*k)`.
These experimental budgets are explicit settings, not a claim that every
automatic profile must use those ratios.

## Precision

`RetrievalConfig.precision` accepts `fp32` and `fp64`. Select precision before
encoding the database and constructing the index. Existing FP32 features and
indexes cannot be relabeled as FP64. For numerical investigations, compare the
same arithmetic with and without filtering before attributing a miss to pruning.

## Defaults

The JSON profiles are the executable reference. DFD and Hausdorff batch profiles
use a query batch of 512. DTW uses 2000, so a 500-query workload can run as one
batch. Large-k memory protection has separate limits. A changed memory budget
must not change k, s, the model, the target or the results.
