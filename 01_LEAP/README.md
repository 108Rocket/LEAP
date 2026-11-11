# LEAP Code

[Home](../README.md) / [Paper to code](../11_docs/paper-map/README.md) / [Supplementary proofs](../README.md#appendix)

Start with `leap.load_checkpoint`, then encode a database and issue queries.
See the [first-query example](../11_docs/quick_start.md).

| Module | Responsibility |
| --- | --- |
| [Shared search](leap/common/README.md) | Encoding, candidate initialization, filtering, and query sessions |
| [Hausdorff](leap/hausdorff/README.md) | Vertex/segment and point-set fields and bounds |
| [DFD](leap/dfd/README.md) | Endpoint bounds, ordered vertex screening, and refinement |
| [DTW](leap/dtw/README.md) | Learned summaries, probe signals, box bounds, and refinement |
| [Learning](leap/training/README.md) | Probe losses, placement, and summary training |

## Repository Guide

| Need | Location |
| --- | --- |
| Settings and trained parameters | [Configurations](../02_configs/) / [checkpoints](../03_checkpoints/README.md) |
| Inputs and preparation | [Datasets](../04_data/README.md) / [examples](../05_examples/) / [tools](../09_tools/) |
| Experiments and checks | [Reproduction](../06_reproduction/README.md) / [tests by RQ](../08_tests/tests/README.md) / [outputs](../14_logs/README.md) |
| Baselines | [Author sources and interfaces](../07_external_models/external_models/README.md) |
| Setup and reference | [Dependencies](../10_requirements/) / [guides](../11_docs/) |
| Attribution | [Licenses](../12_licenses/) |

Install from the repository root. Import the package as `leap`.
