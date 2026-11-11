# Dynamic Time Warping

[Shared search](../common/README.md) / [Formula index](../../../11_docs/paper-map/README.md)

DTW minimizes the accumulated Euclidean cost along a monotone matching path.
LEAP retrieves candidates with a small learned summary, ranks them by probe
signal alignment, and uses bounds on original DTW to filter full search.

## Formulas and Code

| Paper equation | Meaning | Implementation |
| --- | --- | --- |
| (4) `eq:dtw-target` | Original DTW, without a warping window or length normalization | [dtw_kernel](dtw.py#L8), [bounded_kernel](dtw_bounded.py#L18) |
| (6) `eq:dtw-routing-summary` | Cumulative learned weights assign vertices to eight ordered summary positions | [OrderedMass.forward](dtw_routing.py#L51) |
| (12) `eq:dtw-seed-scan` | Compare corresponding summary positions and scale by trajectory length | [pair_scores](dtw_routing.py#L131), [routing_kernel](dtw_routing_tree.py#L82) |
| (13) `eq:dtw-individual` | Larger sum of vertex-to-box distances in the two directions | [dtw_individual_bound_kernel](../common/trajectory_bound_filter.py#L27), [enumerate_sum](dtw_range.py#L63) |
| (14) `eq:dtw-node` | Stronger of the cross-field and query-to-group-box bounds | [dtw_node_bound_kernel](../common/group_envelope_filter.py#L36), [enumerate_sum](dtw_range.py#L63) |
| (17) `eq:dtw-signal` | Ordered probe-signal alignment gives a lower bound for candidate reranking | [signal_kernel](dtw_signal.py#L41), [pairs](dtw_signal.py#L108) |
| (19) `eq:dtw-block-gap` | Block-pair cost lower bound from probe-distance ranges | [stats_torch and bound_torch](dtw_probe_intervals.py) |
| (20) `eq:dtw-weighted-block` | Weighted minimum costs plus residual ordered block alignment | [bound_torch](dtw_probe_intervals.py#L48) |

Eq. (12) is a learned retrieval score, not a lower bound on original DTW. The routing tree bounds **that score**.
Full search uses Eqs. (13)--(14) and additional valid DTW bounds.
The complete proof of the signal bound in Eq. (17) is in
[Appendix D: DTW Signal Lower Bound](../../../README.md#d-dtw-signal-lower-bound).

## Summary and Ordered Filters

- `beta(A, X)` sums distances from vertices of A to the bounding box of X. [box_distance](dtw_range.py#L56) computes one term.
- [prepare](dtw_routing.py#L17) computes six geometry/progress inputs. [OrderedMass](dtw_routing.py#L43) is the 6-to-16-to-1 MLP. Its positive masses, cumulative progress, interval overlap, and weighted positions implement Eq. (6) and its preceding definitions.
- Optimized encoding reuses the same computation through [_routing_sparse.py](_routing_sparse.py), [_routing_transport.py](_routing_transport.py), and [_routing_encoding.py](_routing_encoding.py).
- The pointwise signal `sigma_j(x)` is the distance from a vertex to a probe. [encode](dtw_signal.py#L16) caches these signals.
- Ordered block filtering is implemented by [stats_torch](dtw_probe_intervals.py#L20) and [bound_torch](dtw_probe_intervals.py#L48). Row and column minimum charges are subtracted before residual block alignment so a local cost is not counted twice. [keep_pairs](dtw_probe_intervals.py#L175) applies the threshold.
- [search](dtw_range.py#L175) coordinates full search. [dtw_bounded.py](dtw_bounded.py) and [dtw_frontier.py](dtw_frontier.py) perform original-distance refinement and early stopping.

[Learning](../training/README.md) covers summary and signal objectives.
[Routing tests](../../../08_tests/tests/rq3/test_dtw_routing.py),
[signal tests](../../../08_tests/tests/rq3/test_dtw_signal.py), and
[ordered-filter tests](../../../08_tests/tests/rq3/test_ordered_dtw.py)
check the corresponding computations.
