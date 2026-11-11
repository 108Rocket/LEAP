# Architecture and Maintenance

[Module READMEs](../01_LEAP/README.md) / [Paper formulas and figures](paper-map/README.md)

The public entry is `leap.load_checkpoint`. It binds a trained parameter bundle
to a retrieval configuration and then constructs the common retriever in
`01_LEAP/leap/common/dual_field.py`. The historical class name `DualFieldRetriever`
remains in the implementation and is exported as `leap.LEAP`.

## Shared Search

`common/data.py` owns packed input validation. `common/encoding.py` and
`common/_query_encoding.py` handle common normalization and encoding dispatch.
`common/config.py` defines `RetrievalConfig`. `common/query_search.py` dispatches
the selected distance, and `common/runtime/` owns sessions, private buffers,
automatic budgets and execution switches.

`common/group_envelope_filter.py` and `common/trajectory_bound_filter.py`
provide the shared geometric filtering interfaces used by the distance-specific
traversals. They also retain their small DTW bound helpers beside these interfaces.

`common/interval_index.py` also provides sorting, top-k and index helpers used
by several distances. `common/raw_geometry_refinement.py` contains the shared
legacy refinement interface. These modules keep their existing responsibilities
where splitting them would change the validated execution path.

## Distance Implementations

Hausdorff uses `hausdorff/range_search.py` and the point-set modules. The common
retriever still contains the original field-encoding kernel because it also
constructs representations for DFD and DTW. Point-set HD uses nearest and
farthest vertex fields; 100K HD uses vertex and segment fields.

DFD owns its `dfd_geometry.py`, `dfd_tree.py`, `dfd_range.py` and refinement
modules. Ordered screening uses original vertices before the original DFD
dynamic program.

DTW owns the summary network, routing tree, probe signals, ordered alignment,
parallel traversal, bounded refinement and block-frontier kernels. A bound on
the routing score is distinct from a lower bound on the original DTW distance.

## Changing Code

Keep a single implementation of each numerical kernel. Shared orchestration
belongs in `common/`; a change to one distance belongs in its directory.
Preserve normalization, distance definitions, candidate budgets, tie handling,
precision and output ownership when optimizing execution.

All runtime modules live under `leap`. Shared state is imported from
`leap.common.runtime`, with no second package or module-alias layer.

Use the unit tests for configuration and shape changes. Validate numerical
changes with original-distance references and compare complete ordered IDs and
distances before accepting timings. A source refactor is not a new accuracy or
performance result.
