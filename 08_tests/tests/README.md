# Tests and Paper Research Questions

Tests follow the five RQs in the current paper. Each file has one primary
folder, and its header lists related RQs. Shared checks live in `shared/`.

```text
08_tests/tests/
  rq1/     Overall performance
  rq2/     Full search
  rq3/     Component contributions
  rq4/     Database scalability
  rq5/     Initialization budget and result count
  shared/  Data, release, runtime and timing
```

| RQ | Paper section |
| --- | --- |
| [RQ1](rq1/) | Overall Performance |
| [RQ2](rq2/) | Role of Full Search |
| [RQ3](rq3/) | Contributions of C1--C3 |
| [RQ4](rq4/) | Database Scalability |
| [RQ5](rq5/) | Initialization Budget and Result Count |

RQ4 varies database size with fixed queries and k. RQ5 varies the initial
budget s at fixed k, and then varies k with proportional candidate budgets.
The rq4 directory checks the large-database execution boundary. The rq5
directory checks budget configuration, candidate capacity and large-k results.

These are supporting regression tests. Their synthetic inputs do not reproduce
the paper curves or replace the dataset experiments. See the
[reproduction workflow](../../11_docs/reproduction.md) for dataset evaluation.
`Shared` covers infrastructure used by RQ1--RQ5.

Run one group from the release root:

```bash
python -m pytest 08_tests/tests/rq4
python -m pytest 08_tests/tests/rq5
```

| Test file | Related RQ | Checks |
| --- | --- | --- |
| [test_preparation_paths.py](shared/test_preparation_paths.py) | Shared | Config-relative preparation paths, missing inputs, read-only sources, and protected output folders. |
| [test_pack_data.py](shared/test_pack_data.py) | Shared | Historical lengths-to-offsets conversion, source IDs, unchanged coordinates, dependency declarations, and invalid-input rejection. |
| [test_adopted_profiles.py](rq1/test_adopted_profiles.py) | RQ1 | Published profiles select the documented retrieval paths. |
| [test_dfd_member_buckets.py](rq1/test_dfd_member_buckets.py) | RQ1, RQ2 | DFD length bucketing preserves original distances. |
| [test_dtw_batch_frontier.py](rq1/test_dtw_batch_frontier.py) | RQ1, RQ2 | Batched DTW frontier execution preserves original-distance results. |
| [test_dtw_execution.py](rq1/test_dtw_execution.py) | RQ1, RQ2 | Optimized DTW execution preserves public Full-search outputs. |
| [test_dtw_member_buckets.py](rq1/test_dtw_member_buckets.py) | RQ1, RQ2 | DTW leaf bucketing preserves results across length boundaries. |
| [test_dtw_parallel.py](rq1/test_dtw_parallel.py) | RQ1, RQ2 | Parallel DTW candidate checks preserve neighbor order. |
| [test_dtw_single_layout.py](rq1/test_dtw_single_layout.py) | RQ1, RQ2 | Single-request DTW compaction and storage reuse preserve results. |
| [test_dtw_speed_audit.py](rq1/test_dtw_speed_audit.py) | RQ1, RQ2 | DTW execution optimizations preserve outputs; no timing result is produced. |
| [test_execution_opt.py](rq1/test_execution_opt.py) | RQ1, RQ2 | Execution paths, length handling and overflow preserve search results. |
| [test_external_interfaces.py](rq1/test_external_interfaces.py) | RQ1 | Baseline source provenance and native interfaces; no baseline inference. |
| [test_packed_encoding.py](rq1/test_packed_encoding.py) | RQ1 | Packed input encoding, batching and search-output equivalence. |
| [test_query_batch_defaults.py](rq1/test_query_batch_defaults.py) | RQ1 | Published batch defaults, overrides and memory limits. |
| [test_query_batch_gpu.py](rq1/test_query_batch_gpu.py) | RQ1, RQ2 | Batch-size changes preserve exhaustive-search results. |
| [test_query_encoding.py](rq1/test_query_encoding.py) | RQ1, RQ3 | DTW query summaries and routing preserve values and fallback gradients. |
| [test_routing_sparse.py](rq1/test_routing_sparse.py) | RQ1, RQ3 | Sparse DTW summary routing preserves native arithmetic and outputs. |
| [test_routing_workspace.py](rq1/test_routing_workspace.py) | RQ1, RQ3 | Compact DTW routing workspace preserves summaries and ownership. |
| [test_search_engine.py](rq1/test_search_engine.py) | RQ1, RQ2, RQ5 | Original/fast config isolation, candidate ties and paging, self exclusion, and auto Full output parity. |
| [test_single_plan_dispatch.py](rq1/test_single_plan_dispatch.py) | RQ1 | Single-request scheduling, length buckets and buffer reuse. |
| [test_single_query_profile.py](rq1/test_single_query_profile.py) | RQ1 | Single-request profiles preserve normalization and search configuration. |
| [test_single_query_session.py](rq1/test_single_query_session.py) | RQ1, RQ2 | Single-request sessions preserve results, state isolation and validation. |
| [test_single_query_specialized.py](rq1/test_single_query_specialized.py) | RQ1 | Specialized single-request execution preserves outputs and session state. |
| [test_summary_encoding.py](rq1/test_summary_encoding.py) | RQ1, RQ3 | DFD/DTW optimized encoders preserve reference geometric summaries. |
| [test_dfd_seed_reuse.py](rq2/test_dfd_seed_reuse.py) | RQ1, RQ2 | Initial-candidate distance reuse and complete refinement under retries. |
| [test_dtw_subtree_parallel.py](rq2/test_dtw_subtree_parallel.py) | RQ1, RQ2 | Parallel DTW subtree traversal preserves candidate enumeration. |
| [test_dtw_threshold_refinement.py](rq2/test_dtw_threshold_refinement.py) | RQ1, RQ2 | Threshold-aware DTW scheduling preserves exact refinement. |
| [test_entrance_gpu.py](rq2/test_entrance_gpu.py) | RQ1, RQ2 | Public replay parity and separation of reference labels from search. |
| [test_fp64_retrieval.py](rq2/test_fp64_retrieval.py) | RQ2 | Higher-precision filtering and final ordering for the remaining-miss control. |
| [test_hausdorff_threshold_refinement.py](rq2/test_hausdorff_threshold_refinement.py) | RQ1, RQ2 | Hausdorff early refinement preserves survivors and strict threshold ties. |
| [test_interval_index.py](rq2/test_interval_index.py) | RQ2, RQ3 | Probe fields, tree retrieval and exact candidate refinement. |
| [test_pointset_execution.py](rq2/test_pointset_execution.py) | RQ1, RQ2, RQ3 | Point-set near/far fields, Ranking/Full execution and session safety. |
| [test_precision.py](rq2/test_precision.py) | RQ2 | Coordinate normalization preserves precision before floating-point casting. |
| [test_range_buffers.py](rq2/test_range_buffers.py) | RQ2 | Candidate buffers preserve complete enumeration through overflow. |
| [test_range_search.py](rq2/test_range_search.py) | RQ2 | Full range search agrees with raw distances for ties and external queries. |
| [test_aligned_adoption.py](rq3/test_aligned_adoption.py) | RQ3 | DTW learning derivatives, matched assets and alignment controls. |
| [test_dfd.py](rq3/test_dfd.py) | RQ2, RQ3 | DFD recurrence, ordered bounds and Full-search equivalence. |
| [test_dtw.py](rq3/test_dtw.py) | RQ2, RQ3 | DTW recurrence, external queries and bounded-search correctness. |
| [test_dtw_box_shortcircuit.py](rq3/test_dtw_box_shortcircuit.py) | RQ2, RQ3 | Two-direction DTW box filtering preserves decisions and reduces checks. |
| [test_dtw_filter_fastpaths.py](rq3/test_dtw_filter_fastpaths.py) | RQ2, RQ3 | DTW filtering shortcuts preserve bound decisions. |
| [test_dtw_routing.py](rq3/test_dtw_routing.py) | RQ1, RQ3 | Learned DTW candidate routing and its interface to exact refinement. |
| [test_dtw_signal.py](rq3/test_dtw_signal.py) | RQ3 | Ordered probe signals, radial DTW bounds and learning gradients. |
| [test_dtw_signal_memory.py](rq3/test_dtw_signal_memory.py) | RQ1, RQ3 | Memory-bounded probe signals preserve encodings, ties and gradients. |
| [test_ordered_dtw.py](rq3/test_ordered_dtw.py) | RQ3 | Ordered DTW summaries and alignment across shape boundaries. |
| [test_paper_components.py](rq3/test_paper_components.py) | RQ2, RQ3 | Geometric field equations, filtering hierarchy and component wiring. |
| [test_weighted_dtw.py](rq3/test_weighted_dtw.py) | RQ3 | Weighted DTW summary recurrence, ordered ranking and bounded cascade. |
| [test_database_capacity.py](rq4/test_database_capacity.py) | RQ4 | Candidate generation remains valid beyond one million database rows. |
| [test_auto_config.py](rq5/test_auto_config.py) | RQ1, RQ5 | Candidate-budget constraints and automatic configuration; no budget sweep. |
| [test_auto_config_gpu.py](rq5/test_auto_config_gpu.py) | RQ1, RQ2, RQ5 | Automatic budgets preserve the resolved manual Full-search outputs. |
| [test_candidate_capacity.py](rq5/test_candidate_capacity.py) | RQ1, RQ2, RQ5 | Candidate capacity and top-k ordering across leaf sizes. |
| [test_full_large_k.py](rq5/test_full_large_k.py) | RQ1, RQ2, RQ5 | Large-k Full search agrees with exhaustive original-distance results. |
| [test_full_large_k_dispatch.py](rq5/test_full_large_k_dispatch.py) | RQ1, RQ2, RQ5 | Large-k requests select the configured Full-search path. |
| [test_query_search.py](rq5/test_query_search.py) | RQ1, RQ2, RQ5 | Full-database versus initial-pool search with varying result count k. |
| [test_data.py](shared/test_data.py) | Shared | Input formats, data validation and reproducible dataset loading. |
| [test_release.py](shared/test_release.py) | Shared | Checkpoint integrity, published profiles and portable release paths. |
| [test_runtime.py](shared/test_runtime.py) | Shared | Shared session context, configuration serialization and module ownership. |
| [test_timing.py](shared/test_timing.py) | Shared | Timing boundaries, warmup exclusion and arithmetic seed aggregation. |
| [test_logs.py](shared/test_logs.py) | Shared | Grouped run outputs, console capture, failure records and dry-run isolation. |

[Validation results](../../11_docs/validation.md) | [Overview](../../README.md)
