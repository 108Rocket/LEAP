# External Models

Eight primary baselines include their original author source in separate model
directories. Each directory has an `upstream/` source snapshot, `SOURCE.json`
with commit/blob hashes, and a small LEAP interface outside the author code.
External checkpoints and datasets are not bundled. Model internals are unchanged.

| Method | Author source | Interface |
| --- | --- | --- |
| [ConvTraj](learned/ConvTraj/README.md) | [Proudc/ConvTraj](https://github.com/Proudc/ConvTraj) | Neural encoder |
| [TrajCL](learned/TrajCL/README.md) | [changyanchuan/TrajCL](https://github.com/changyanchuan/TrajCL) | Neural encoder |
| [MovSemCL](learned/MovSemCL/README.md) | [ryanlaics/MovSemCL](https://github.com/ryanlaics/MovSemCL) | Neural encoder |
| [SIMformer](learned/SIMformer/README.md) | [SUSTC-ChuangYANG/SIMformer](https://github.com/SUSTC-ChuangYANG/SIMformer) | Neural encoder |
| [TSMini](learned/TSMini/README.md) | [changyanchuan/TSMini](https://github.com/changyanchuan/TSMini) | Neural encoder |
| [TrajAgg](learned/TrajAgg/README.md) | [zhx66741/TrajAgg](https://github.com/zhx66741/TrajAgg) | Neural encoder |
| [LB-TrajRep](bounds/LB-TrajRep/README.md) | [LIWEIDENG0830/LB-TrajRep](https://github.com/LIWEIDENG0830/LB-TrajRep) | Lower bounds. The authors' pinned release does not provide `tools/landmark_embedding.py`. |
| [TraSS](exact/TraSS/README.md) | [huajunge/k-sim](https://github.com/huajunge/k-sim) | Exact Java/HBase search |

## Additional Entries

| Directory | Role |
| --- | --- |
| [HNSW](indexes/HNSW/README.md) | Paper vector-index control using native FAISS over ConvTraj vectors |
| [TrajGAT](optional/TrajGAT/README.md) | Optional author-source entry from the earlier project |
| [DITA](optional/DITA/README.md) | Optional author-source entry from the earlier project |
| [RePo](optional/RePo/README.md) | Optional author-source entry from the earlier project |

The optional methods are not primary baselines reported in the current paper.
Their entry points remain separate from the eight bundled source snapshots.

## Author Sources

[sources.json](sources.json) records the exact upstream commits and model paths.
The primary source snapshots are already included and can be verified offline
without Git. Omitted datasets, binaries and supplementary outputs are recorded
in each `SOURCE.json`. The verifier checks retained file bytes against SHA256
and their original Git blob hashes. No patches or models are executed.

```bash
python 07_external_models/external_models/install_sources.py --list
python 07_external_models/external_models/install_sources.py --verify-only
python 07_external_models/external_models/install_sources.py --model TrajGAT
```

Optional author repositories are fetched only when explicitly selected. Use
`--include-optional` to include all optional entries. `--source-root DIRECTORY`
selects a separate flat directory of full Git checkouts, leaving bundled code
untouched. Existing source is verified and never overwritten.

Follow each author's README to install its dependencies and prepare its inputs.
For Python imports of these LEAP interfaces, add `07_external_models` to
`PYTHONPATH`. The `external_models` package name remains unchanged.
Run Python baselines in separate processes because their packages share names
such as `model`, `config` and `utils`. To pass arguments to an author entry point:

```bash
python 07_external_models/external_models/run.py TrajCL --entrypoint train_trajsimi.py -- --help
```

The entry-point filename and arguments are those of the selected author's
release. Java/HBase setup and commands follow the TraSS README. Upstream
licenses remain applicable.

## Data and Evaluation

| File | Purpose |
| --- | --- |
| [common.py](common.py) | 100K split validation and deterministic retrieval metrics |
| [prepare_teachers.py](prepare_teachers.py) | Convert and verify supplied reference distances |
| [ConvTraj inputs](learned/ConvTraj/inputs.py) | Pack coordinates into sequence/raster tensors |
| [SDD encoder interfaces](reproduction/sdd_porto/sdd_models.py) | Load author classes and supplied inference metadata |
| [SDD encoding runner](reproduction/sdd_porto/encode_benchmark.py) | Complete encoding or separate GPU activity measurement |
| [TraSS dataset adapter](exact/TraSS/NativeBench.java) | Packed trajectories and native CPU query calls |

SDD checkpoint metadata and commands are documented in the
[SDD interface guide](reproduction/sdd_porto/README.md). The interfaces do not
replace missing author modules or change model internals. Original input limits
and dependency requirements apply. An interface or a verified source checkout
does not by itself establish a reproduced benchmark result.

[Overview](../../README.md)
