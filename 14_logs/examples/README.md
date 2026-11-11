# Porto Hausdorff Log Examples

These selected records come from completed historical experiments. They show
the configuration, output and validation format, not a new execution of this
source release or a replacement for the complete paper evidence. Only Porto
Hausdorff records are bundled.

| Category | Included examples |
| --- | --- |
| [Training](training/hausdorff/) | SDD Porto seed2000 histories, phase configurations and checkpoint receipt |
| [RQ1](rq1/hausdorff/) | Ranking, Full and LB accuracy/timing at full SDD Porto size |
| [RQ2](rq2/hausdorff/) | GPU exhaustive scan versus Full, with original gates and raw timings |
| [RQ3](rq3/hausdorff/) | 100K Porto filtering-level ablations, with their original configurations |
| [RQ4](rq4/hausdorff/) | Database scaling endpoints at 100K and 1597579, including scorer calls |
| [RQ5](rq5/hausdorff/) | k=50 candidate-path adaptation, with output parity checks |

These six folders are representative selections, not every seed, ablation or sweep
point. The original five-repeat timing logs are not relabeled as the later
three-seed, three-repeat paper aggregation. See [INDEX.json](INDEX.json).

Each example keeps its configuration and result together. `SOURCE.json` records
the source task, selected source files and checksums, plus any public-export
redactions. Numerical values and training seeds are preserved. Historical timing
repetitions retain their original protocol and are not additional training seeds.
`configuration.json` in the historical experiment folders is an extract of
recorded settings, not an executable job file. Runnable configurations remain in
[05_examples](../../05_examples/) and [02_configs](../../02_configs/).

Do not write new runs here. Use the adjacent `training/`, `evaluation/`, `timing/`
and `tests/` categories instead.

[Log guide](../README.md) | [Overview](../../README.md)
