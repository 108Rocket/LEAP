# SDD Porto, Geolife, and Chengdu

[Dataset index](../README.md) / [Preparation code](../../09_tools/data_preparation/README.md)

## Download

`python entrance.py` downloads the [Geolife Hausdorff subset (211 MB)](https://drive.google.com/file/d/1a-CzXT0ud6JcAKAfplpej_nyHtuDEM8T/view?usp=sharing)
after confirmation, verifies SHA256, and runs 1,000 queries against 9,386 trajectories.
Already downloaded? Use `python entrance.py --archive /path/to/SDD_Geolife_Hausdorff.zip`.
This subset retains the five original SDD Geolife geometry/Hausdorff files byte for byte.

Use the published [SequenceDistanceDataset](https://huggingface.co/datasets/Airchang/SDD).
Its trajectory section contains Porto, Geolife, and Chengdu, with train/query/base
splits and distance matrices. Use the published benchmark coordinates and ordering.
Reprocessing a raw city dataset does not necessarily reproduce these splits.

| Published input | LEAP use | Code |
| --- | --- | --- |
| `base_list` | Search database | [prepare_sdd.py: pack](../../09_tools/data_preparation/archive/prepare_sdd.py#L75) |
| `query_list` | Fixed external queries, 500 for the Porto timing experiments | [audit_cities.py](../../09_tools/data_preparation/archive/audit_cities.py) |
| `train_list` | City-specific training and validation | [prepare_city_training.py](../../09_tools/data_preparation/archive/prepare_city_training.py) |
| `Haus`, `DFD`, and `DTW` matrices | Training labels and reference-distance checks | [prepare_sdd.py: exact](../../09_tools/data_preparation/archive/prepare_sdd.py#L47), [audit_cities.py](../../09_tools/data_preparation/archive/audit_cities.py) |

Hausdorff is **point set** here. DFD preserves vertex order. DTW sums Euclidean
costs without a warping window. The archived checker distinguishes point-set
Hausdorff from vertex-to-segment Hausdorff explicitly.

## Training Split

[prepare_city_training.py](../../09_tools/data_preparation/archive/prepare_city_training.py)
removes training records whose complete ordered FP64 coordinate sequences also
appear among the test queries. It keeps identical remaining sequences together,
then uses seed 2000 to assign 20% of these groups to validation. It records all
selected IDs and extracts the matching label submatrices. Test query and database
records are unchanged.

The script computes a training-only translation and one shared scale for both
coordinate axes. For inference, use the normalization saved in the selected
checkpoint. Do not refit it to the database or query set.

## Runtime Files

Convert selected coordinates to the [packed schema](../README.md#packed-geometry)
using [pack_data.py](../../09_tools/pack_data.py). Place `database.npz`,
`queries.npz`, and Top-ID references under the matching city folder. Reference
rows and columns must follow the exact query and database order.

The published matrices provide reference distances. Their dtype alone does not
certify a newly computed FP64 exhaustive reference. Keep published-label checks
and an independently recomputed reference distinguishable in run records.

The preparation scripts accept explicit [path configuration](../../09_tools/data_preparation/README.md#configure-and-check).
Inputs are read-only and outputs default to `14_logs/data_preparation/`.
Full preparation and training still require the published data and reference matrices.
