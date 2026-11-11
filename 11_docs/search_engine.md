# Search Engine

Keep an existing config unchanged to use the original engine. To try the new
engine, use a separate config file with one extra field:

```json
"search_engine": "fast"
```

Set it to `"original"` to select the original implementation. The default is
`"original"`; this does not disable any preexisting optimizations.

New examples: [HD](../02_configs/hausdorff/fast_pointset_hd.json),
[DFD](../02_configs/dfd/fast_dfd.json), [DTW](../02_configs/dtw/fast_dtw.json).
Load the matching checkpoint as usual. Build the search index after loading the
config. Rebuild the index/session when changing engines; encoded database fields
and model weights can be reused. `parameter_mode="auto"` retains this choice.

To preserve the loaded profile's budgets and change only the engine, set it
before building any index or session:

```python
from dataclasses import replace
from leap import load_checkpoint

model = load_checkpoint("sdd.porto.dfd", mode="auto")
model.config = replace(model.config, search_engine="fast")
# Encode the database and build its index as usual.
```

| Distance | New candidate-search implementation |
| --- | --- |
| Point-set HD, near/far | Joint field partition, blockwise stopping, fewer merges |
| DFD | Endpoint-first blockwise checks, fewer merges |
| DTW, signal ranking | Same eight-term routing score, fewer and smaller merges |

Scores, tie-breaking IDs, candidate budgets and original-distance refinement
are unchanged. Existing specialized Single sessions use the selected engine.
This switch does not automatically introduce a reusable Batch CUDA graph; that
is a separate session optimization. It does not change the target metric, train
a model, guarantee a speedup, or make coarse ranking exact under raw distance.
Fast mode requires FP32 and does not support legacy `large_k=True`. Current
`full_large_k` and auto profiles remain available. HD requires point-set near/far
fields and ordinary Full (`hausdorff_execution="auto"`); DTW requires signal
ranking and optimized execution. Unsupported settings raise a configuration error.

All 20 original JSON profiles are unchanged. See [validation](validation.md#optional-fast-search)
for the checked environment and scope; Single speedups are workload dependent.
