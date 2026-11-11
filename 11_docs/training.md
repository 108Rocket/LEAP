# Training Sources

The public checkpoints can be used directly for inference. The sources here
also retain the original learning code and explicit training adapters.

For SDD Porto point-set Hausdorff, `python entrance.py --dataset porto --mode train` prepares
the published inputs, runs the two 50-epoch stages, saves the new checkpoint,
and then evaluates it.
This path has passed the L20 training preflight, including the adapted placement
kernel, but a fresh complete 100-epoch replay has not yet been validated.

| Part | Source |
| --- | --- |
| Two-phase Hausdorff learner and final placement | `06_reproduction/training_sources/` |
| Portable adapter for the archived Hausdorff phases | `01_LEAP/leap/reproduction/training.py`, recipe `paper-probes` |
| DTW signal probes | `01_LEAP/leap/training/train_dtw_probes.py` |
| DTW learned summary weights | `01_LEAP/leap/training/train_dtw_routing.py` |
| City-specific DFD objective | `06_reproduction/training_sources/sdd/sdd_dfd_train.py` |
| SDD point-set Hausdorff objective | `06_reproduction/training_sources/sdd/sdd_probe_train.py` and `sdd_point_fields.py` |

The archived two-phase source manifest preserves the original source hashes.
These historical scripts depend on an injected `config` module.
The public adapter supplies the configured paths and seed before
calling their stages. They are objective-reference files, not standalone
launch commands.

```bash
python -m leap train --config your-training.json --recipe paper-probes
python -m leap train --config your-training.json --recipe paper-probes --execute
```

Configure `training.raw_inputs`, `training.hausdorff_teacher`, `training.seed`
and `output_root` first. The teacher must use the same original-distance target
as the intended checkpoint. Input labels and split membership are separate
required artifacts. The adapter records their hashes and never overwrites an
existing training output.

The generic archived Hausdorff adapter is not claimed to reproduce
the later city-specific SDD point-set retraining pipeline. DFD's historical
training pipeline also needs its original prepared label inputs and orchestration.
The shipped inference bundles provide those final trained parameters directly.

This organization task does not rerun training or claim that a fresh full
training replay has completed. A successful import or dry-run is a software
check, not a trained-model result. For the paper, hyperparameters are selected
on Porto and remain fixed across cities, while each city trains its own weights.
