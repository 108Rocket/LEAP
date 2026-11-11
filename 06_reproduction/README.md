# Learning Objective Sources

The executable review interface is `python -m leap`, described in the
[reproduction guide](../11_docs/reproduction.md) and
[training guide](../11_docs/training.md).

This directory retains the historical objective implementations used to train
the published probes. `training_sources/` contains the two-phase Hausdorff
learner, selection, placement and original orchestration. `training_sources/sdd/`
contains the city-specific DFD learner, point-set Hausdorff learner and its
point-field implementation collected from the completed L20 training snapshots.

These source snapshots keep their original injected configuration dependencies.
The placement stage receives its paths from the public adapter, with no internal
server defaults. The archived orchestration files are not public launch
commands. They make the actual learning formulas inspectable; the public path
adapter supports the explicitly documented recipes. No new end-to-end training
replay is claimed for this packaging task.

Verify the objective snapshots without loading GPU code:

```bash
python 06_reproduction/verify_sources.py
```

The manifest records current byte hashes and original source hashes. Import
paths have been updated to the LEAP package. Historical experiment logs and
host-specific orchestration are not part of this release.
