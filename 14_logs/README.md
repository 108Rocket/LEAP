# Logs and Run Outputs

[Reproduction guide](../11_docs/reproduction.md) / [Timing definitions](../11_docs/timing.md) / [Porto Hausdorff examples](examples/README.md)

`python entrance.py` creates `entrance/<timestamp-id>/` with `console.log`,
`SUMMARY.txt`, the evaluation configuration, actual returned IDs and raw timings.
Its `--check-only` mode records readiness without running a model.

The supplied configuration sets `output_root` to `repo:14_logs`. Public CLI
commands (`python -m leap`) group generated output by purpose. Relative `--output` overrides are
resolved from the current working directory. Existing run directories are never
overwritten. A dry run does not create logs or start a model.

```text
14_logs/
  entrance/     Guided runs, including optional training
  training/     Recipe name and timestamp
  evaluation/   Job name and timestamp
  timing/       Job name and timestamp
  tests/        Test suite and timestamp
  examples/     Porto Hausdorff reference logs grouped by paper RQ
```

Each new CLI run contains:

```text
RUN/
  configuration.json   Input configuration and resolved plan
  console.log          Standard output, warnings and Python traceback
  status.json          Running, complete or failed, with timestamps
  artifacts/           Worker output in its original format
```

| Command | Files under `artifacts/` |
| --- | --- |
| `python -m leap train --execute ...` | Training history, configuration and checkpoints; archived Hausdorff stages keep their own subdirectories |
| `python -m leap evaluate ...` | `protocol.json`, `result.json`, `summary.csv`, neighbor IDs and completion/failure record |
| `python -m leap timing ...` | Evaluation files plus `samples.jsonl`, `hardware.jsonl` and progress state |
| `python 09_tools/run_checks.py ...` | `junit.xml`; selected test paths and source roots are in the run configuration |

The logging wrapper changes output organization only. It does not train a new
model, change search budgets, or include logging in measured GPU operations.
Low-level Python functions and standalone author scripts retain their explicit
output parameters. Use the public commands above for automatic console capture.

Generated runs are ignored by Git. The selected [Porto Hausdorff examples](examples/README.md)
are shipped separately and never used as a new run's output directory.

[Overview](../README.md)
