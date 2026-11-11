# Paper to Code

[Home](../../README.md) / [Code structure](../../01_LEAP/README.md) / [Dataset preparation](../../04_data/README.md)

Each equation links to its module README, which links to the implementation.
The numbering below follows the November 10, 2026 manuscript. LaTeX labels identify
the equations if later revisions change their numbers.

## Numbered Equations

| Eq. | LaTeX label | What it computes | Module |
| --- | --- | --- | --- |
| 1 | `eq:directed` | Directed vertex-to-segment Hausdorff distance | [Hausdorff fields](../../01_LEAP/leap/hausdorff/README.md#vertex-and-segment-fields) |
| 2 | `eq:target` | Symmetric Hausdorff distance | [Hausdorff fields](../../01_LEAP/leap/hausdorff/README.md#vertex-and-segment-fields) |
| 3 | `eq:dfd-target` | Original DFD | [DFD](../../01_LEAP/leap/dfd/README.md#formulas-and-code) |
| 4 | `eq:dtw-target` | Original DTW | [DTW](../../01_LEAP/leap/dtw/README.md#formulas-and-code) |
| 5 | `eq:dual-fields` | Nearest-position and segment fields | [Hausdorff fields](../../01_LEAP/leap/hausdorff/README.md#vertex-and-segment-fields) |
| 6 | `eq:dtw-routing-summary` | Learned ordered summary positions | [DTW](../../01_LEAP/leap/dtw/README.md#formulas-and-code) |
| 7 | `eq:lower-bound` | Cross-field trajectory bound | [Hausdorff fields](../../01_LEAP/leap/hausdorff/README.md#vertex-and-segment-fields) |
| 8 | `eq:pointset-bound` | Nearest/farthest point-set bound | [Point-set fields](../../01_LEAP/leap/hausdorff/README.md#point-set-fields) |
| 9 | `eq:node-bound` | Cross-field group bound | [Hausdorff fields](../../01_LEAP/leap/hausdorff/README.md#vertex-and-segment-fields) |
| 10 | `eq:dfd-bound` | DFD field and endpoint bound | [DFD](../../01_LEAP/leap/dfd/README.md#formulas-and-code) |
| 11 | `eq:dfd-node` | DFD group bound from field ranges and endpoint boxes | [DFD](../../01_LEAP/leap/dfd/README.md#formulas-and-code) |
| 12 | `eq:dtw-seed-scan` | Summary retrieval score, not an original-DTW lower bound | [DTW](../../01_LEAP/leap/dtw/README.md#formulas-and-code) |
| 13 | `eq:dtw-individual` | DTW trajectory box bound | [DTW](../../01_LEAP/leap/dtw/README.md#formulas-and-code) |
| 14 | `eq:dtw-node` | DTW group bound | [DTW](../../01_LEAP/leap/dtw/README.md#formulas-and-code) |
| 15 | `eq:dfd-subset` | Ordered retained-vertex recurrence | [DFD](../../01_LEAP/leap/dfd/README.md#formulas-and-code) |
| 16 | `eq:dfd-progressive` | Combined DFD reranking bound | [DFD](../../01_LEAP/leap/dfd/README.md#formulas-and-code) |
| 17 | `eq:dtw-signal` | Probe-signal lower bound for DTW candidate reranking | [DTW](../../01_LEAP/leap/dtw/README.md#formulas-and-code) |
| 18 | `eq:shared-threshold` | Initial upper bound used as the pruning threshold | [Shared search](../../01_LEAP/leap/common/README.md#search-procedure) |
| 19 | `eq:dtw-block-gap` | Block-pair cost lower bound from probe-distance ranges | [DTW block filtering](../../01_LEAP/leap/dtw/README.md#summary-and-ordered-filters) |
| 20 | `eq:dtw-weighted-block` | Weighted block lower bound with residual ordered alignment | [DTW block filtering](../../01_LEAP/leap/dtw/README.md#summary-and-ordered-filters) |
| 21 | `eq:boundary-loss` | Probe-learning loss for lower-bound targets | [Probe learning](../../01_LEAP/leap/training/README.md#probe-objectives) |
| 22 | `eq:dtw-signal-loss` | Signal-probe ranking loss | [DTW objectives](../../01_LEAP/leap/training/README.md#dtw-objectives) |
| 23 | `eq:witness-gap` | Theoretical bound gap from critical-position proximity | [Probe learning](../../01_LEAP/leap/training/README.md#probe-objectives) |
| 24 | `eq:dtw-summary-loss` | Summary-network ranking loss for candidate-pool inclusion | [DTW objectives](../../01_LEAP/leap/training/README.md#dtw-objectives) |

## Other Expressions and Algorithm

| Paper content | Code explanation |
| --- | --- |
| Vertices, segments, point-to-set distance, clipped segment projection | [Hausdorff](../../01_LEAP/leap/hausdorff/README.md) |
| Cached features, k/p/s, top-k, distance/ID ties, and Algorithm 1 | [Shared search](../../01_LEAP/leap/common/README.md) |
| Point-set farthest field and node ranges | [Point-set fields](../../01_LEAP/leap/hausdorff/README.md#point-set-fields) |
| DFD sampled bound `G_g`, retained vertices, and final minimum of the recurrence | [DFD](../../01_LEAP/leap/dfd/README.md) |
| DTW box sum `beta`, positive weights, cumulative progress, probe signals, and ordered block bounds | [DTW](../../01_LEAP/leap/dtw/README.md#summary-and-ordered-filters) |
| Training radius, mined negatives, auxiliary loss, sufficient probe regions, and fixed-count placement | [Probe learning](../../01_LEAP/leap/training/README.md#probe-objectives) |
| DTW summary inclusion loss and signal ranking loss | [DTW objectives](../../01_LEAP/leap/training/README.md#dtw-objectives) |
| Validity, neighbor inclusion, exactness, and complexity | Mathematical properties of the [bounds and search](../../01_LEAP/leap/common/README.md). There is no separate function that computes a proof or a complexity guarantee. |

## Figures and Modules

| Fig. | Paper figure label | Content | Code README |
| --- | --- | --- | --- |
| 1 | `fig:overview`, `fig:progressive` | (a) fields and bounds, (b) probe placement, (c) visiting order | [Hausdorff](../../01_LEAP/leap/hausdorff/README.md), [learning](../../01_LEAP/leap/training/README.md), [DFD](../../01_LEAP/leap/dfd/README.md) |
| 2 | `fig:architecture` | Initialization, three filtering levels, and exact top-k selection | [Shared search](../../01_LEAP/leap/common/README.md) |
| 3 | `fig:dtw-routing-alignment` | DTW summary-based candidate retrieval | [DTW](../../01_LEAP/leap/dtw/README.md) |
| 4 | `fig:dtw_box`, `fig:dtw-box-bound` | (a) position-to-box bound for filtering, (b) probe-signal bound for ranking | [DTW](../../01_LEAP/leap/dtw/README.md) |
| 5 | `fig:porto-probe-learning` | Sufficient probe regions for five Hausdorff training pairs | [Learning](../../01_LEAP/leap/training/README.md) |
| 6 | `fig:level-ablation` | Initialization and filtering ablations, retaining early stopping | [Shared search](../../01_LEAP/leap/common/README.md), [RQ3 tests](../../08_tests/tests/README.md) |
| 7 | `fig:database-scalability` | Database-size sweep of ranking and exact-search time | [RQ4 tests](../../08_tests/tests/README.md), [Porto Hausdorff records](../../14_logs/examples/README.md) |
| 8 | `fig:query-sensitivity`, `fig:initial-budget`, `fig:result-count` | Initialization budget (a,c) and returned neighbors (b,d): time and initial coverage | [Configuration](../configuration.md), [RQ5 tests](../../08_tests/tests/README.md) |

GEF, TBF, and RGR in the repository illustrations denote the paper's group filter,
trajectory filter, and original-distance refinement. [Editable illustration sources](../figures/README.md)
are kept separately. They illustrate the same mechanisms but are not all identical
to the manuscript's composite artwork. RQ tests are implementation checks, not
scripts that regenerate the paper's plots.

## Reported Quantities

| Quantity | Implementation or record |
| --- | --- |
| HR@k and R5@20 | [replay.metrics](../../01_LEAP/leap/reproduction/replay.py#L99) |
| Initial coverage | Fraction of reference top-k IDs present in the initialization set. Training examples: [phase1 validation](../../06_reproduction/training_sources/phase1_train.py#L18). Sweep outputs: [Porto Hausdorff records](../../14_logs/examples/README.md). |
| Search, encoding, and index time | [Timing rules](../timing.md), [timing.py](../../01_LEAP/leap/reproduction/timing.py) |
| Original-distance evaluations (Evals.) | [Full-search counters](../../01_LEAP/leap/common/dual_field.py#L484) and each distance's search. Counts include early-stopped evaluations and exclude reuse of cached initialization distances. See [timing definitions](../timing.md). |
| Across-seed aggregation | [Reproduction protocol](../reproduction.md). Repeated timing runs are not independent training seeds. |

Original mapping source (historical): `LEAP__Learned_Probes_for_Exact_Trajectory_Similarity_Search.zip`,
SHA256 `379d9c762d18aeba23f17497b8d2c6cb4d0b9aabfbcf1d52714218a25be8a4d5`.
The equation and figure tables above have been synchronized with the November 10 manuscript.
