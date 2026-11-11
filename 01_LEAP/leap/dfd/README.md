# Discrete Frechet Distance

[Shared search](../common/README.md) / [Formula index](../../../11_docs/paper-map/README.md)

DFD matches vertices in order and minimizes the largest matched distance.
Cheap field and endpoint bounds retrieve a pool. Order-aware bounds rerank this
pool before full search evaluates the initialization candidates.

## Formulas and Code

| Paper equation | Meaning | Implementation |
| --- | --- | --- |
| (3) `eq:dfd-target` | Minimum bottleneck cost over monotone matching paths | [dfd_kernel](dfd.py#L14), [original_value](dfd_refinement.py#L22) |
| (10) `eq:dfd-bound` | Maximum of vertex-field differences and the two endpoint distances | [bound_kernel](dfd.py#L109), [candidate_kernel](dfd_tree.py#L11) |
| (11) `eq:dfd-node` | Group lower bound from nearest-position field ranges and endpoint boxes | [make_native](dfd_tree.py#L128), [native_enum](dfd_tree.py#L88) |
| (15) `eq:dfd-subset` | Match retained source vertices to nondecreasing target positions | [direction](dfd_geometry.py#L59), using [prefix_min](dfd_geometry.py#L54) |
| (16) `eq:dfd-progressive` | Combine field, endpoint, sampled-vertex, and both ordered bounds | [initial_candidates](dfd_tree.py#L179) |

The retained vertices keep both endpoints and add the vertex farthest from the
current segments. See [simplify_kernel](dfd_geometry.py#L20). The sampled bound
`G_g` is [witness_bound](dfd_geometry.py#L110). The final minimum of the ordered
recurrence is computed in [direction](dfd_geometry.py#L59).

## Search and Learning

- Group field ranges and endpoint boxes: [make_native](dfd_tree.py#L128).
- Threshold and cached refinement: [dfd_range.py](dfd_range.py).
- Ordered feasibility check before full dynamic programming: [ordered_fast](dfd_refinement.py#L7).
- Probe training with the separation loss: [sdd_dfd_train.py](../../../06_reproduction/training_sources/sdd/sdd_dfd_train.py). Endpoint terms remain fixed.

[DFD tests](../../../08_tests/tests/rq3/test_dfd.py) check the distance and bounds.
[Seed reuse tests](../../../08_tests/tests/rq2/test_dfd_seed_reuse.py) check reuse of
initialization distances.
