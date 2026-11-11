import torch
import argparse
import pickle
import os
import numexpr as ne 
from tools import function
from tools import test_methods
from tools.landmark_embedding import LandmarkConfig, embed_traj_list_by_landmarks_gpu, embed_traj_list_by_landmarks_with_se_gpu, sample_landmark_points, embed_traj_list_by_landmarks
from tools.start_end_lb import start_end_lb_gpu


from config.new_config import ConfigClass

import numpy as np


from tools.pivot_selection import PivotSelectConfig, select_pivot_points

def build_true_knn_from_dist(test_distance_matrix: np.ndarray, topk: int = 100) -> np.ndarray:
    """Build ground-truth KNN indices from a query-base distance matrix."""
    return np.argsort(test_distance_matrix, axis=1)[:, :topk].astype(np.int64)


def resolve_distance_matrix_paths(root_read_path: str, dist_type: str) -> tuple[str, str]:
    """Support both flat and nested distance-matrix layouts."""
    nested_dir = os.path.join(root_read_path, dist_type)
    nested_train = os.path.join(nested_dir, f"{dist_type}_train_distance_matrix_result")
    nested_test = os.path.join(nested_dir, f"{dist_type}_test_distance_matrix_result")
    if os.path.exists(nested_train) and os.path.exists(nested_test):
        return nested_train, nested_test

    flat_train = os.path.join(root_read_path, f"{dist_type}_train_distance_matrix_result")
    flat_test = os.path.join(root_read_path, f"{dist_type}_test_distance_matrix_result")
    return flat_train, flat_test


def load_trajectory_dataset(root_read_path: str, dist_type: str):
    """Support both split-list and concatenated traj_list layouts."""
    train_list_path = os.path.join(root_read_path, "train_list")
    query_list_path = os.path.join(root_read_path, "query_list")
    base_list_path = os.path.join(root_read_path, "base_list")

    if all(os.path.exists(path) for path in (train_list_path, query_list_path, base_list_path)):
        train_list = pickle.load(open(train_list_path, "rb"))
        query_list = pickle.load(open(query_list_path, "rb"))
        base_list = pickle.load(open(base_list_path, "rb"))
        train_set, query_set, base_set = len(train_list), len(query_list), len(base_list)
        traj_list = train_list + query_list + base_list
    else:
        train_set, query_set, base_set = function.set_dataset(root_read_path)
        traj_list_path = os.path.join(root_read_path, "traj_list")
        traj_list = pickle.load(open(traj_list_path, "rb"))

    train_dist_matrix_path, test_dist_matrix_path = resolve_distance_matrix_paths(root_read_path, dist_type)
    train_distance_matrix = pickle.load(open(train_dist_matrix_path, "rb"))
    test_distance_matrix = pickle.load(open(test_dist_matrix_path, "rb"))

    return traj_list, train_set, query_set, base_set, train_distance_matrix, test_distance_matrix

def get_args():
    parser = argparse.ArgumentParser(description="HyperParameters for Traj Embedding")

    parser.add_argument("--network_type",                   type=str,   default="PIVOT", help="network type: PSE|IPSE|PIVOT|LANDMARK")
    parser.add_argument("--train_flag",                     type=str,   default="test", help="train flag")
    parser.add_argument("--device",                         type=str,   default="cuda:0", help="device")
    parser.add_argument("--random_seed",                    type=int,   default=666, help="random seed")
    parser.add_argument("--mode",                           type=str,   default="train-directly", help="mode")

    parser.add_argument("--root_write_path",                type=str,   default="/data3/menghaotian/LB-TrajRep_geo_porto/0_geolife", help="root write path")
    parser.add_argument("--root_read_path",                 type=str,   default="/data3/menghaotian/LB-TrajRep_geo_porto/0_geolife", help="root read path")
    parser.add_argument("--dist_type",                      type=str,   default="dfret", help="distance type: dtw|frechet|haus")

    # -------- landmark / pivot embedding (training-free) --------
    parser.add_argument("--landmark_num",                  type=int,   default=32, help="#pivot/landmark points")
    parser.add_argument("--pivot_strategy",                type=str,   default="point-hard",
                        help="Pivot selection strategy: legacy | point-random | point-fps | point-hard | point-ndcg | point-tightness | point-kmedoids | point-facility | traj-random | traj-fps | traj-tightness")
    parser.add_argument("--traj_point_subsample",          type=int,   default=0, help="subsample points per traj for embedding (0=disable)")
    parser.add_argument("--landmark_batch",                type=int,   default=2048, help="batch size for embedding computation")

    # point-* pivot selection params
    parser.add_argument("--pivot_pool_per_traj",           type=int,   default=16, help="[point-*] candidate points sampled per trajectory")
    parser.add_argument("--pivot_pool_include_se",         type=int,   default=1, help="[point-*] include start/end points in candidate pool (1/0)")

    # point-hard / point-ndcg (require train distance matrix)
    parser.add_argument("--pivot_select_traj_subsample",   type=int,   default=0, help="[point-hard/point-ndcg] subsample points per traj when computing proxy phi (0=disable)")
    parser.add_argument("--hard_anchors",                  type=int,   default=512, help="[point-hard] #anchors sampled")
    parser.add_argument("--hard_k",                        type=int,   default=50, help="[point-hard] hard region defined by top-k true neighbors")
    parser.add_argument("--hard_pairs_per_anchor",         type=int,   default=5, help="[point-hard] pairs per anchor")
    parser.add_argument("--hard_cand_pool",                type=int,   default=512, help="[point-hard] candidate pool per greedy step")

    parser.add_argument("--ndcg_anchors",                  type=int,   default=256, help="[point-ndcg] #anchors sampled")
    parser.add_argument("--ndcg_eval_k",                   type=int,   default=50, help="[point-ndcg] true top-k neighbors included per anchor")
    parser.add_argument("--ndcg_at",                       type=int,   default=10, help="[point-ndcg] report NDCG@k during selection")
    parser.add_argument("--ndcg_neg",                      type=int,   default=50, help="[point-ndcg] #random negatives per anchor")
    parser.add_argument("--ndcg_cand_pool",                type=int,   default=256, help="[point-ndcg] candidate pool per greedy step")

    # traj-* pivot selection params (requires train distance matrix)
    parser.add_argument("--pivot_point_rule",              type=str,   default="mid", help="[traj-*] map selected pivot trajectories to points: mid|start|end|max_dev")
    # reuse point-tightness
    parser.add_argument("--tight_cand_pool",               type=int,   default=512, help="[traj-tightness] candidate pool size per greedy step")
    parser.add_argument("--tight_num_pairs",               type=int,   default=20000, help="[traj-tightness] sampled pairs for tightness estimation")

    # -------- pivot lower bound (training-free, metric-only) --------
    # parser.add_argument("--pivot_num",                     type=int,   default=128, help="#pivot trajectories")
    # parser.add_argument("--pivot_sample",                  type=str,   default="random", help="random|fps_start_end")

    args = parser.parse_args()

    return args
    

if __name__ == "__main__":
    
    args   = get_args()

    function.setup_seed(args.random_seed)
    resolved_device = function.resolve_torch_device(args.device)
    device_str = str(resolved_device)

    os.makedirs(args.root_write_path, exist_ok=True)
    traj_list, train_set, query_set, base_set, train_distance_matrix, test_distance_matrix = load_trajectory_dataset(
        args.root_read_path,
        args.dist_type,
    )

    ConfigClass(network_type=args.network_type,
                root_write_path=args.root_write_path,
                root_read_path=args.root_read_path,
                mode=args.mode,
                dist_type=args.dist_type,
                device=resolved_device,
                train_flag=args.train_flag,
                train_set=train_set,
                query_set=query_set,
                base_set=base_set)

    # Method map used in the LB-TrajRep paper terminology:
    # - PIVOT: pivot lower bound only, i.e. use landmark/pivot embedding and
    #   evaluate the max-|phi_p(a)-phi_p(b)| lower bound.
    # - IPSE: landmark embedding already contains start/end information.
    # - PSE: take the maximum of pivot lower bound and start-end lower bound.
    #
    # The pivot selection strategy comes from args.pivot_strategy and is
    # dispatched in tools/pivot_selection.py:
    # - point-random    -> Random
    # - point-fps       -> FPS
    # - point-facility  -> Facility
    # - point-kmedoids  -> KMedoids
    # - point-tightness -> Tightness
    # - point-hard      -> Hardness

    # ===================== Pivot Lower Bound Baseline =====================
    # Run when mode == 'pivot_lb' OR network_type contains 'PIVOT'
    if "SE"  == args.network_type.upper():
        # split trajectoriesW
        train_traj = traj_list[:train_set]
        query_traj = traj_list[train_set:train_set + query_set]
        base_traj  = traj_list[train_set + query_set:]

        pred_dist_se = start_end_lb_gpu(query_traj, base_traj, device=device_str)
        pred_dist = pred_dist_se
        true_knn = build_true_knn_from_dist(test_distance_matrix, topk=100)
        metrics = test_methods.test_all_print_new(
            query_id_list=list(range(query_set)),
            base_id_list=list(range(base_set)),
            true_knn=true_knn,
            test_distance_matrix=pred_dist,
        )
        print("[SEEmbedding] dist_type={}  landmarks={}  sample={}  subsample={}".format(
            args.dist_type, args.landmark_num, args.pivot_strategy, None if args.traj_point_subsample <= 0 else args.traj_point_subsample
        ))
        print("[Eval] R@1={:.4f} R@5={:.4f}  R@10={:.4f}  R@50={:.4f}  R@100={:.4f}  R@10|50={:.4f}".format(
            metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]
        ))
        raise SystemExit(0)
    # PIVOT branch: only keep the pivot-based lower bound from selected landmarks.
    if args.mode.lower() == "pivot_lb" or "PIVOT" in args.network_type.upper():
        # split trajectories
        train_traj = traj_list[:train_set]
        query_traj = traj_list[train_set:train_set + query_set]
        base_traj  = traj_list[train_set + query_set:]

        pool_traj = train_traj + base_traj
        cfg = LandmarkConfig(
            num_landmarks=args.landmark_num,
            sample_method=args.pivot_strategy,
            traj_point_subsample=(None if args.traj_point_subsample <= 0 else args.traj_point_subsample),
            random_seed=args.random_seed,
        )
        if args.pivot_strategy.lower() == "legacy":
            landmarks = sample_landmark_points(pool_traj, cfg)
        else:
            piv_cfg = PivotSelectConfig(
                strategy=args.pivot_strategy,
                K=args.landmark_num,
                seed=args.random_seed,
                pool_per_traj=args.pivot_pool_per_traj,
                pool_include_start_end=bool(args.pivot_pool_include_se),
                select_traj_subsample=args.pivot_select_traj_subsample,
                hard_anchors=args.hard_anchors,
                hard_k=args.hard_k,
                hard_pairs_per_anchor=args.hard_pairs_per_anchor,
                hard_cand_pool=args.hard_cand_pool,
                ndcg_anchors=args.ndcg_anchors,
                ndcg_eval_k=args.ndcg_eval_k,
                ndcg_at=args.ndcg_at,
                ndcg_neg=args.ndcg_neg,
                ndcg_cand_pool=args.ndcg_cand_pool,
                traj_point_rule=args.pivot_point_rule,
                tight_cand_pool=args.tight_cand_pool,
                tight_num_pairs=args.tight_num_pairs,
                device=device_str,
            )

            landmarks = select_pivot_points(train_traj, piv_cfg, D_train=train_distance_matrix)
            
        base_emb = embed_traj_list_by_landmarks_gpu(
            base_traj,
            landmarks,
            traj_point_subsample=cfg.traj_point_subsample,
            batch_size=1024,
            device=device_str
        )
        
        # embed query/base
        query_emb = embed_traj_list_by_landmarks_gpu(
            query_traj,
            landmarks,
            traj_point_subsample=cfg.traj_point_subsample,
            batch_size=1024,
            device=device_str
        )

        pred_dist = test_methods.pivot_lb_scores_from_phi_gpu(query_emb, base_emb, base_batch_size=1600000, device=device_str)
        true_knn = build_true_knn_from_dist(test_distance_matrix, topk=100)
        metrics = test_methods.test_all_print_new(
            query_id_list=list(range(query_set)),
            base_id_list=list(range(base_set)),
            true_knn=true_knn,
            test_distance_matrix=pred_dist,
        )

        print("[PivotEmbedding] dist_type={}  landmarks={}  sample={}  subsample={}".format(
            args.dist_type, args.landmark_num, args.pivot_strategy, cfg.traj_point_subsample
        ))
        print("[Eval] R@1={:.4f} R@5={:.4f}  R@10={:.4f}  R@50={:.4f}  R@100={:.4f}  R@10|50={:.4f}".format(
            metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]
        ))
        raise SystemExit(0) 

    # IPSE branch: use the landmark embedding variant that explicitly includes
    # start/end information inside the embedding representation.
    if args.mode.lower() == "pivot_lb+ISE" or "IPSE" in args.network_type.upper():
        # split trajectories
        train_traj = traj_list[:train_set]
        query_traj = traj_list[train_set:train_set + query_set]
        base_traj  = traj_list[train_set + query_set:]

        pool_traj = train_traj + base_traj
        cfg = LandmarkConfig(
            num_landmarks=args.landmark_num,
            sample_method=args.pivot_strategy,
            traj_point_subsample=(None if args.traj_point_subsample <= 0 else args.traj_point_subsample),
            random_seed=args.random_seed,
        )
        if args.pivot_strategy.lower() == "legacy":
            landmarks = sample_landmark_points(pool_traj, cfg)
        else:
            piv_cfg = PivotSelectConfig(
                strategy=args.pivot_strategy,
                K=args.landmark_num,
                seed=args.random_seed,
                pool_per_traj=args.pivot_pool_per_traj,
                pool_include_start_end=bool(args.pivot_pool_include_se),
                traj_point_rule=args.pivot_point_rule,
                tight_cand_pool=args.tight_cand_pool,
                tight_num_pairs=args.tight_num_pairs,
            )
            landmarks = select_pivot_points(train_traj, piv_cfg, D_train=train_distance_matrix)

        # embed query/base
        base_emb = embed_traj_list_by_landmarks_with_se_gpu(
            base_traj,
            landmarks,
            traj_point_subsample=cfg.traj_point_subsample,
            batch_size=1024,
            device=device_str
        )

        query_emb = embed_traj_list_by_landmarks_with_se_gpu(
            query_traj,
            landmarks,
            traj_point_subsample=cfg.traj_point_subsample,
            batch_size=1024,
            device=device_str
        )
        # compute predicted distance matrix in embedding space
        pred_dist = test_methods.pivot_lb_scores_from_phi_gpu(query_emb, base_emb, base_batch_size=1600000, device=device_str)
        true_knn = build_true_knn_from_dist(test_distance_matrix, topk=100)
        metrics = test_methods.test_all_print_new(
            query_id_list=list(range(query_set)),
            base_id_list=list(range(base_set)),
            true_knn=true_knn,
            test_distance_matrix=pred_dist,
        )

        print("[IPSEEmbedding] dist_type={}  landmarks={}  sample={}  subsample={}".format(
            args.dist_type, args.landmark_num, args.pivot_strategy, cfg.traj_point_subsample
        ))
        print("[Eval] R@1={:.4f} R@5={:.4f}  R@10={:.4f}  R@50={:.4f}  R@100={:.4f}  R@10|50={:.4f}".format(
            metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]
        ))
        raise SystemExit(0) 
    
    # PSE branch: combine two lower bounds by taking their pointwise maximum:
    # pivot lower bound and start-end lower bound.
    if args.mode.lower() == "pivot_lb+se" or "PSE" in args.network_type.upper():
  
        # split trajectories
        train_traj = traj_list[:train_set]
        query_traj = traj_list[train_set:train_set + query_set]
        base_traj  = traj_list[train_set + query_set:]

        pool_traj = train_traj + base_traj
        cfg = LandmarkConfig(
            num_landmarks=args.landmark_num,
            sample_method=args.pivot_strategy,
            traj_point_subsample=(None if args.traj_point_subsample <= 0 else args.traj_point_subsample),
            random_seed=args.random_seed,
        )
        if args.pivot_strategy.lower() == "legacy":
            landmarks = sample_landmark_points(pool_traj, cfg)
        else:
            piv_cfg = PivotSelectConfig(
                strategy=args.pivot_strategy,
                K=args.landmark_num,
                seed=args.random_seed,
                pool_per_traj=args.pivot_pool_per_traj,
                pool_include_start_end=bool(args.pivot_pool_include_se),
                traj_point_rule=args.pivot_point_rule,
                tight_cand_pool=args.tight_cand_pool,
                hard_cand_pool=args.hard_cand_pool,
                tight_num_pairs=args.tight_num_pairs,
                hard_k=args.hard_k,
                device=device_str,
            )
            landmarks = select_pivot_points(train_traj, piv_cfg, D_train=train_distance_matrix)

        # embed query/base
        query_emb = embed_traj_list_by_landmarks_gpu(
            query_traj,
            landmarks,
            traj_point_subsample=cfg.traj_point_subsample,
            batch_size=1024,
            device=device_str
        )
        base_emb = embed_traj_list_by_landmarks_gpu(
            base_traj,
            landmarks,
            traj_point_subsample=cfg.traj_point_subsample,
            batch_size=1024,
            device=device_str
        )

        # compute predicted distance matrix in embedding space
        pred_dist_p = test_methods.pivot_lb_scores_from_phi_gpu(query_emb, base_emb, base_batch_size=1600000, device=device_str)
        pred_dist_se = start_end_lb_gpu(query_traj, base_traj, device=device_str)
        pred_dist = ne.evaluate("where(pred_dist_p > pred_dist_se, pred_dist_p, pred_dist_se)")
        true_knn = build_true_knn_from_dist(test_distance_matrix, topk=100)
        metrics = test_methods.test_all_print_new(
            query_id_list=list(range(query_set)),
            base_id_list=list(range(base_set)),
            true_knn=true_knn,
            test_distance_matrix=pred_dist,
        )
        print("[PSEEmbedding] dist_type={}  landmarks={}  sample={}  subsample={}".format(
            args.dist_type, args.landmark_num, args.pivot_strategy, cfg.traj_point_subsample
        ))
        print("[Eval] R@1={:.4f} R@5={:.4f}  R@10={:.4f}  R@50={:.4f}  R@100={:.4f}  R@10|50={:.4f}".format(
            metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]
        ))
        raise SystemExit(0) 

    # ===================== Landmark Embedding Baseline =====================
    # Run when mode == 'landmark' OR network_type contains 'LANDMARK'
    if args.mode.lower() == "landmark" or "LANDMARK" in args.network_type.upper():
        # split trajectories
        train_traj = traj_list[:train_set]
        query_traj = traj_list[train_set:train_set + query_set]
        base_traj  = traj_list[train_set + query_set:]

        pool_traj = train_traj + base_traj
        cfg = LandmarkConfig(
            num_landmarks=args.landmark_num,
            sample_method=args.pivot_strategy,
            traj_point_subsample=(None if args.traj_point_subsample <= 0 else args.traj_point_subsample),
            random_seed=args.random_seed,
        )
        if args.pivot_strategy.lower() == "legacy":
            landmarks = sample_landmark_points(pool_traj, cfg)
        else:
            piv_cfg = PivotSelectConfig(
                strategy=args.pivot_strategy,
                K=args.landmark_num,
                seed=args.random_seed,
                pool_per_traj=args.pivot_pool_per_traj,
                pool_include_start_end=bool(args.pivot_pool_include_se),
                traj_point_rule=args.pivot_point_rule,
                tight_cand_pool=args.tight_cand_pool,
                tight_num_pairs=args.tight_num_pairs,
            )
            landmarks = select_pivot_points(train_traj, piv_cfg, D_train=train_distance_matrix)

        # embed query/base
        query_emb = embed_traj_list_by_landmarks(
            query_traj,
            landmarks,
            traj_point_subsample=cfg.traj_point_subsample,
            batch_size=args.landmark_batch,
        )
        base_emb = embed_traj_list_by_landmarks(
            base_traj,
            landmarks,
            traj_point_subsample=cfg.traj_point_subsample,
            batch_size=args.landmark_batch,
        )

        # compute predicted distance matrix in embedding space
        pred_dist = test_methods.get_feature_distance(query_emb, base_emb)
        true_knn = build_true_knn_from_dist(test_distance_matrix, topk=100)
        metrics = test_methods.test_all_print_new(
            query_id_list=list(range(query_set)),
            base_id_list=list(range(base_set)),
            true_knn=true_knn,
            test_distance_matrix=pred_dist,
        )

        print("[LandmarkEmbedding] dist_type={}  landmarks={}  sample={}  subsample={}".format(
            args.dist_type, args.landmark_num, args.pivot_strategy, cfg.traj_point_subsample
        ))
        print("[Eval] R@1={:.4f} R@5={:.4f}  R@10={:.4f}  R@50={:.4f}  R@100={:.4f}  R@10|50={:.4f}".format(
            metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]
        ))
        raise SystemExit(0)
