# 100K Porto and Germany

[Dataset index](../README.md) / [Preparation code](../../09_tools/data_preparation/README.md)

## Sources

| Input | Source and code |
| --- | --- |
| Porto raw trajectories | [ECML/PKDD taxi trajectory dataset](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i). The bundled author [Porto preprocessing code](../../07_external_models/external_models/learned/TrajCL/upstream/utils/preprocessing_porto.py) builds the processed input. |
| Germany raw trajectories | [OpenStreetMap public GPS traces](https://wiki.openstreetmap.org/wiki/Planet.gpx), identified as the Germany source in [TrajCL, Section V-A and reference 51](https://arxiv.org/html/2210.05155v3). |
| Processed trajectories and original 7K/1K/2K convention | [TrajCL repository](https://github.com/changyanchuan/TrajCL), its [dataset download](https://drive.google.com/drive/folders/1wvFSdi4T1RvG1ww7TlobQJoTSBdJ7zWq?usp=sharing), and the bundled [data loader](../../07_external_models/external_models/learned/TrajCL/upstream/utils/data_loader.py). |

LEAP uses the processed `merc_seq` coordinates. Source row numbers refer to a
specific processed file, not directly to the raw GPS archive.

## Selection

| Set | Frozen selection |
| --- | --- |
| Porto training / validation / historical test | Original processed rows `[960907,967907)`, `[967907,968907)`, `[968907,970907)` |
| Porto Self-100K | Original processed rows `[1098180,1198180)` |
| Germany training / validation / historical test | Processed rows `[100391,107391)`, `[107391,108391)`, `[108391,110391)` |
| Germany Self-100K | Sampled from the processed source, excluding the original 10K and duplicate geometries. The archived [selection routine](../../09_tools/data_preparation/archive/prepare_100k.py) records the procedure. |
| External 1K queries | The same [selection routine](../../09_tools/data_preparation/archive/prepare_100k.py) selects distinct queries and excludes database and supervised-set geometries. |

Ranges are zero-based and end-exclusive. The Porto compact file has different
row numbers. Its source-ID mapping must be used instead of applying the original
ranges to the compact file.

The old 2K set is a separate supervised similarity test. It is not Self-100K.
The inherited Germany train and validation sets contain ten identical point
sequences under different record IDs. Hyperparameters are selected on Porto,
and Germany is not used for model selection.

## Prepare the Runtime Files

1. Obtain the processed trajectories and the matching frozen source IDs.
2. Select the original ordered `merc_seq` arrays. Preserve vertices, direction, and database order.
3. Convert the selected records using [pack_data.py](../../09_tools/pack_data.py) and the matching `100k.porto.<distance>` checkpoint. Germany requires a separately supplied checkpoint and configuration.
4. Put `database.npz` in the city folder and references in `references/`. For Self-100K, query the database and exclude the same record ID.

The historical selector accepts the explicit [path configuration](../../09_tools/data_preparation/README.md#configure-and-check).
Frozen compact-ID sidecars and full reference arrays are not shipped.
Raw Germany-to-`germany_20200` preprocessing is not implemented by the bundled
Porto preprocessor. These inputs are required for an identical benchmark replay.
