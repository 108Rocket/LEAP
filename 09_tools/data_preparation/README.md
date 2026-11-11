# Data Preparation Code

[100K guide](../../04_data/100k/README.md) / [SDD guide](../../04_data/sdd/README.md)

## Runnable Converter

[pack_data.py](../pack_data.py) converts already selected JSON trajectories or a
packed NPZ to LEAP's input format. It attaches the chosen checkpoint's
normalization without projecting, resampling, or changing the coordinates.
NPZ inputs accept `points` with either `offsets` or `lengths`. The latter is the
100K preparation script's output. Supplied `source_ids` are retained unchanged.
Install the reproduction requirements before using the preparation scripts,
including `pandas` for the processed 100K input files.

```bash
python 09_tools/pack_data.py --input prepared_porto.json \
  --checkpoint sdd.porto.hausdorff --output 04_data/sdd/porto/database.npz
```

## Generation Sources

| Stage | Source |
| --- | --- |
| Raw Porto preprocessing and original distance labels | Author's [preprocessing_porto.py](../../07_external_models/external_models/learned/TrajCL/upstream/utils/preprocessing_porto.py) |
| Original supervised 7K/1K/2K splits | Author's [data_loader.py](../../07_external_models/external_models/learned/TrajCL/upstream/utils/data_loader.py) |
| Frozen 100K databases and external queries | [prepare_100k.py](archive/prepare_100k.py) |
| SDD numeric input conversion and distance checks | [prepare_sdd.py](archive/prepare_sdd.py) |
| SDD city inventory and coordinate hashes | [audit_cities.py](archive/audit_cities.py) |
| SDD train/validation split and label extraction | [prepare_city_training.py](archive/prepare_city_training.py) |

## Configure and Check

Edit [paths.example.json](paths.example.json) to point to your inputs, or copy it
to your own location. All relative paths are resolved from the config file,
independently of the working directory. Defaults write generated data and reports
under `14_logs/data_preparation/`. Inputs remain read-only. A nonempty output
directory is rejected to protect completed runs.

```bash
python 09_tools/data_preparation/archive/prepare_sdd.py \
  --config 09_tools/data_preparation/paths.example.json --check-only
python 09_tools/data_preparation/archive/prepare_sdd.py \
  --config 09_tools/data_preparation/paths.example.json
```

The same options work for `prepare_100k.py`, `audit_cities.py`, and
`prepare_city_training.py`. Run SDD preparation before the city audit.
`--check-only` checks paths without loading datasets or running distance checks.
It does not certify input contents or reproduce experiment results.

| Stage | Required input |
| --- | --- |
| `prepare_100k` | Processed Porto compact and Germany files, frozen Porto source IDs, Germany formal geometry, and the two original normalization assets |
| `prepare_sdd` | Fully extracted SDD trajectory folder and a verified archive manifest with a `sha256` field |
| `audit_cities` | SDD trajectory folder, prepared Porto points/offsets, and Porto's `PROTOCOL.json` |
| `prepare_city_training` | SDD trajectory folder including original training distance matrices |

The Porto ID JSON needs `compact_iloc` and `raw_float64_sha256`. Germany's
formal NPZ needs `points` and `lengths`. Preserve the original input versions
and frozen selections. Do not substitute newly sampled IDs for missing sidecars.
The 100K script draws external queries using the historical procedure. It is not
a replacement for the original Self-100K input construction.

The four historical routines now use explicit paths and local helper imports.
[sources.json](archive/sources.json) records both original and adapted hashes.
Selection seeds, geometry, normalization, distance formulas, and split rules are
unchanged. The packaged path checks have not rerun full dataset preparation or
training. Identical paper replay still requires the separate inputs above.

The external author code remains unmodified in its own model directory.
Dataset files, frozen-ID sidecars, and full distance matrices are separate inputs.
Only load pickle inputs from a trusted source whose checksum you have verified.
