# Release Validation

## Optional Fast Search

The optional [search engine](search_engine.md) was checked on 2026-11-03 with
RTX 4090, PyTorch 2.11.0+cu128 and Triton Windows 3.6.0.post26. The original
engine remains the default; all 20 original JSON profiles are unchanged.

Five new test methods and 175 existing regression tests passed. For each of
HD, DFD and DTW, all 500 SDD Porto queries against 1,597,579 trajectories
returned identical candidates at budgets 1, 20, 50, 100 and 1000. Full top-20
IDs, order and FP32 distances also agreed exactly, as did Single outputs.
Synthetic tests additionally cover ties, self exclusion, paging and auto Full
at k = 1, 50 and 1000. No model was retrained.
After integration, 209 targeted regression tests passed on a fresh release
snapshot, including checkpoint loading, profile resolution and runtime checks;
there were no failures or skips.

Timing checks used one trained seed (2000), not three training seeds. Batch
improvements do not establish a stable Full Single speedup: DFD/DTW Single
medians were essentially unchanged. These checks do not replace the earlier
L20 validation or the paper's timing tables, and do not establish fast-engine
compatibility with every other PyTorch/Triton version.

## Guided Porto Launcher

The [guided entrance](entrance.md) adds setup and data preparation around the
existing Hausdorff timing runner. Its L20 acceptance uses the bundled seed2000
model, all 500 SDD Porto queries and all 1,597,579 database trajectories.

| Check | Result |
| --- | --- |
| Complete CUDA suite before final setup/UX guards | 915 tests and 54 subtests passed; no failures or skips |
| Final launcher-specific suite | 40 tests passed in both the existing and fresh environments |
| Fresh environment setup | Launcher created its own environment, installed verified official wheels, and passed `pip check` |
| CPU-only regression suite in the fresh environment | 165 passed; 103 skipped in CPU-only mode |
| Original archive extraction | Porto Hausdorff inputs extracted and verified; original archive unchanged |
| Fresh preparation versus cached preparation | Identical input hashes, packed arrays, reference IDs and returned IDs |
| Repeated search and single versus batch | Identical ordered Ranking/Full IDs |
| Training preflight | Teacher distances, fields, gradients and adapted placement kernel passed |

HR@20 is 0.5830 for Ranking and 0.9999 for Full against the published SDD
point-set distance matrix. It is not a newly recomputed FP64 reference.
Batch timing covers all 500 resident queries; complete single requests use the
first 64 queries. Three timing repeats reuse one trained seed.

The training path is implemented, but a fresh 100-epoch run has **not** been
completed in this acceptance. The preflight executes no optimizer steps.
Fresh installation was tested through `entrance.py` on L20 using official
wheels transferred from an internet-connected machine, because the server
could not reach the package sites. The official dataset endpoint was checked
with a verified range response; the 28 GB archive was not downloaded again.
The clean-room run extracted the existing original archive, prepared the full
dataset, and then repeated retrieval in the newly installed environment. Both
runs returned exactly the same ordered IDs as the earlier accepted run.
These are functional checks on a shared GPU, not new paper timings. An unauthenticated
checkpoint HTTP request returned 404; the bundled checkpoint works, and a
hash-checked local Git recovery path is provided.

## Full L20 Validation

The release was validated on 2026-10-31 using NVIDIA L20, PyTorch 2.5.1+cu124,
Triton 3.1.0, NumPy 2.1.3 and pandas 2.2.3. The environment is isolated from
other projects. These historical checks used the then-published 15 seed-2000
checkpoints and their five DTW routers without retraining. The current release
ships six Porto checkpoints and the Geolife Hausdorff quick-start checkpoint; the original validation results below
are retained as a historical record.

| Check | Result |
| --- | --- |
| Complete code test suite, with CUDA enabled | 890 passed, zero failures and zero skips |
| Every checkpoint at k = 1, 20 and 1000 | 45 cases passed against unpruned original-distance evaluation |
| Original versus organized code on full datasets | All 15 dataset/distance pairs passed |
| Single versus batch search on SDD | All 7,500 query/distance cases passed in both Ranking and Full |
| Full results that differ from the existing FP64 reference | All 703 affected queries matched an unpruned FP32 scan exactly |
| Shipped file checksums | 703 files verified |
| Author source and archived training-source integrity | Eight author snapshots and nine archived training files verified |
| Editable installation and CLI outside the checkout | Passed |

The checkpoint checks use synthetic geometry to compare Full IDs and FP32
distances with the unpruned scorer. They also check single-request Full at all
three k values, and single/batch Ranking at k = 1 and 20. They are correctness
checks, not retrieval-accuracy measurements on an additional dataset.

## Full Dataset Replay

Each row below was evaluated under Hausdorff, DFD and DTW, using both Ranking
and Full at k = 20. The original and organized implementations use identical
input files, checkpoints and inference configurations.

| Dataset | Queries | Database trajectories |
| --- | ---: | ---: |
| Self-100K Porto | 100,000 | 100,000 |
| Self-100K Germany | 100,000 | 100,000 |
| SDD Porto | 500 | 1,597,579 |
| SDD Geolife | 1,000 | 9,386 |
| SDD Chengdu | 1,000 | 3,000 |

This covers 607,500 query/distance cases per implementation. All ordered
Ranking IDs, Full IDs, FP32 distances, scorer counts and thresholds agree
exactly before and after organization. Self-100K excludes the query's own
record. The raw single-request comparison covers SDD, whose external-query
contract matches the session API.

The existing FP64 reference was read only after retrieval. Every query whose
Full top-20 order or membership differed from that reference was then checked
against an unpruned scan using the same FP32 geometry and scorer. All 703
queries matched that scan bit for bit in IDs and distances. No incorrect
pruning was found in these checks.

For example, the public Chengdu HD seed-2000 model gives HR@1 = 0.997 and
HR@20 = 1.000 under default FP32. Its three top-1 disagreements involve nearly
equal original distances. This does not reproduce the separately reported
HR@1 = 1.000. The reported paper/homepage tables are unchanged, and this run
does not establish the precision setting used for that reported value.

## Repairs and Scope

The full suite exposed an outdated homepage-navigation assertion and an
endpoint-bound rounding difference between DFD candidate scorers. The
navigation test now follows the configuration guide to the profile index.
The DFD bound uses the same round-to-nearest square root as the tree and dense
paths. An independent FP32/FP64 regression checks that rounding. The original
DFD distance recurrence, checkpoints and training objectives are unchanged.

This validation does not rerun training or external-model inference. External
checks cover interface behavior and the integrity of retained author sources.
The remaining training-replay requirements are described in
[Training Sources](training.md). No elapsed time from this correctness run
replaces a paper benchmark measurement.

The initial 2026-10-30 packaging checks included 234 CPU tests, 73 targeted
CUDA tests and five synthetic output comparisons. The full CUDA suite and
full-data replay above supersede their limited execution coverage.

Run `python -m pytest 08_tests/tests -q` in the installed CUDA environment for
the complete code suite, or `python 09_tools/run_checks.py --cpu` for CPU-only
checks. Checkpoint loading verifies files against the current checkpoint catalog.
Dataset replay requires the original inputs and explicit inference configuration.
