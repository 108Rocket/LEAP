# Distance Configurations

[Configuration guide](../11_docs/configuration.md) / [Example run plan](../05_examples/README.md)

Choose a distance folder, then a query mode. The original profiles and their
parameter values are unchanged; optional accelerated profiles are listed below.

| Folder | Profiles |
| --- | --- |
| [Hausdorff](hausdorff/) | Polyline HD and point-set HD |
| [DFD](dfd/) | Discrete Frechet distance |
| [DTW](dtw/) | Dynamic time warping |

`auto_*` selects parameters by database size and k. `batch_query_*` is the
resident batch profile, `single_query_*` is the complete-request profile,
and `*full_top1000*` enables the large-k Full-search path.
For Hausdorff, names containing `pointset_hd` or `sdd_` use the SDD point-set
target; the other Hausdorff profiles use the polyline target.

| Distance | Configuration |
| --- | --- |
| dfd | [auto_dfd.json](dfd/auto_dfd.json) |
| dtw | [auto_dtw.json](dtw/auto_dtw.json) |
| hausdorff | [auto_hausdorff.json](hausdorff/auto_hausdorff.json) |
| hausdorff | [auto_pointset_hd.json](hausdorff/auto_pointset_hd.json) |
| dfd | [batch_query_dfd.json](dfd/batch_query_dfd.json) |
| dtw | [batch_query_dtw.json](dtw/batch_query_dtw.json) |
| hausdorff | [batch_query_hausdorff.json](hausdorff/batch_query_hausdorff.json) |
| hausdorff | [batch_query_pointset_hd.json](hausdorff/batch_query_pointset_hd.json) |
| dfd | [dfd.json](dfd/dfd.json) |
| dtw | [dtw.json](dtw/dtw.json) |
| dfd | [full_top1000_dfd.json](dfd/full_top1000_dfd.json) |
| dtw | [full_top1000_dtw.json](dtw/full_top1000_dtw.json) |
| hausdorff | [full_top1000_hausdorff.json](hausdorff/full_top1000_hausdorff.json) |
| hausdorff | [hausdorff.json](hausdorff/hausdorff.json) |
| hausdorff | [sdd_full_top1000_hausdorff.json](hausdorff/sdd_full_top1000_hausdorff.json) |
| dtw | [sdd_porto_dtw_aligned.json](dtw/sdd_porto_dtw_aligned.json) |
| dfd | [single_query_dfd.json](dfd/single_query_dfd.json) |
| dtw | [single_query_dtw.json](dtw/single_query_dtw.json) |
| hausdorff | [single_query_hausdorff.json](hausdorff/single_query_hausdorff.json) |
| hausdorff | [single_query_pointset_hd.json](hausdorff/single_query_pointset_hd.json) |

## Optional Fast Search

Existing profiles use `search_engine="original"`. Select one of these separate
profiles to enable `search_engine="fast"` without editing an original file:

| Distance | Accelerated profile |
| --- | --- |
| Point-set Hausdorff | [fast_pointset_hd.json](hausdorff/fast_pointset_hd.json) |
| DFD | [fast_dfd.json](dfd/fast_dfd.json) |
| DTW | [fast_dtw.json](dtw/fast_dtw.json) |

These change candidate-search execution, not trained weights or ranking scores.
See [Search Engine](../11_docs/search_engine.md) for usage and supported settings.

Checkpoint loading selects the original matching distance profile automatically.
Explicit job paths use `repo:02_configs/<distance>/<file>.json`.

[Configuration guide](../11_docs/configuration.md) | [Overview](../README.md)

