# LB-TrajRep

**Using Lower-Bound Representations for Trajectory Similarity Learning** (Accepted by VLDB 2026)

---

## Overview

LB-TrajRep is a unified framework for trajectory similarity retrieval based on **admissible lower-bound representations**.
Instead of learning neural embeddings to approximate trajectory distances, LB-TrajRep constructs vector representations from pivot-induced lower-bound components. The resulting representations:

- Preserve similarity ordering under classical trajectory distances  
- Enable efficient large-scale top-k retrieval  
- Require no neural network training  
- Remain stable across different distance measures  

The current implementation supports retrieval under:

- Hausdorff distance  
- Discrete Fréchet Distance (DFD)  
- Dynamic Time Warping (DTW)  

Once pivot points are selected, trajectory representations can be reused across distance measures without re-training.

Extensive experiments on multiple real-world trajectory datasets demonstrate that LB-TrajRep consistently achieves strong ranking performance compared with recent neural trajectory embedding methods, while offering better interpretability and significantly lower training overhead.

## Experimental Results as Examples

### GeoLife

<p align="center">
  <img src="figures/Geolife-results.png" width="750">
</p>

On GeoLife, LB-TrajRep consistently outperforms recent neural trajectory embedding methods under Hausdorff, DFD, and DTW. The improvements are particularly significant for alignment-based distances.

---

### Porto

<p align="center">
  <img src="figures/Porto-results.png" width="750">
</p>

On the large-scale Porto dataset, LB-TrajRep maintains strong ranking performance and stability across distance measures, demonstrating its scalability and robustness.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

We use the same processed datasets and splits as in SIMformer [https://github.com/SUSTC-ChuangYANG/SIMformer] and ConvTraj[https://github.com/Proudc/ConvTraj].

Please download the preprocessed datasets from the official repositories of those works. 

Geolife、Porto、ChengDu (From ConvTraj): https://huggingface.co/datasets/Airchang/SDD

T-Drive (From SIMformer): https://drive.google.com/drive/folders/1Mp4rx9wS4ToTiqliLiB4gPtqwJrbEpqa

AIS (From SIMformer): https://drive.google.com/drive/folders/1PhjiJaqGAlq8Bp-9hJIks3_XaFnpXRiX

`--root_read_path` should point to one dataset directory such as `.../Trajectory/Chengdu`.

The current code supports both of the following dataset layouts.

Layout A: split trajectory lists + nested distance folders

```text
<dataset_root>/
  train_list
  query_list
  base_list
  dfd/dfd_train_distance_matrix_result
  dfd/dfd_test_distance_matrix_result
  dtw/dtw_train_distance_matrix_result
  dtw/dtw_test_distance_matrix_result
  haus/haus_train_distance_matrix_result
  haus/haus_test_distance_matrix_result
```

Layout B: single `traj_list` + flat distance files

```text
<dataset_root>/
  traj_list
  <dist_type>_train_distance_matrix_result
  <dist_type>_test_distance_matrix_result
```

Important notes:

- `--root_write_path` must be a writable output directory.
- `--dist_type` must match the actual folder / file names in your dataset.
- For the ConvTraj SDD datasets used in our tests, valid examples are `dfd`, `dtw`, `haus`, and `edr`.

---

## Reproducing Results

Run experiments with explicit arguments instead of relying on the default paths in `train.py`.

Below is a tested example for the **PSE** setting (Pivot + Start-End lower bound) with hardness-aware pivot selection on the Chengdu dataset:

```bash
python train.py \
  --network_type PSE \
  --pivot_strategy point-hard \
  --landmark_num 32 \
  --dist_type dfd \
  --root_read_path ../ConvTrajData/SequenceDistanceDataset/Trajectory/Chengdu \
  --root_write_path ./Chengdu \
  --device cuda:0
```

If your dataset is stored elsewhere, replace only the two path arguments:

```bash
python train.py \
  --network_type PSE \
  --pivot_strategy point-hard \
  --landmark_num 32 \
  --dist_type dfd \
  --root_read_path <dataset_path> \
  --root_write_path <output_path> \
  --device cuda:0
```

After running, the script prints evaluation metrics:

```
[Eval] R@1  R@5  R@10  R@50  R@100  R@10|50
```

These correspond to the ranking metrics reported in the paper.

Method names in the paper map to the following code arguments:

```text
PIVOT -> --network_type PIVOT
IPSE  -> --network_type IPSE
PSE   -> --network_type PSE
```

```text
Random     -> --pivot_strategy point-random
FPS        -> --pivot_strategy point-fps
Facility   -> --pivot_strategy point-facility
KMedoids   -> --pivot_strategy point-kmedoids
Tightness  -> --pivot_strategy point-tightness
Hardness   -> --pivot_strategy point-hard
```

To reproduce other configurations, modify:

```
--network_type
--pivot_strategy
--dist_type
```

---

## Acknowledgement

This implementation partially builds upon the publicly available ConvTraj codebase.
We sincerely thank the authors for open-sourcing their implementation, which served as a valuable foundation for our experimental framework.
