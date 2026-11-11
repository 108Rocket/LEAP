import time
import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Callable

from tools import function

Array = np.ndarray

def traj_to_np(traj) -> Array:
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Trajectory must have shape (L,2), got {arr.shape}")
    return arr

def _build_traj_stack(traj_list, S: int, seed: int = 0):
    """
    Convert trajectories into a padded stack of shape (N, S, 2) with NaN padding.
    Assumes each traj is (L,2). We subsample to at most S points.
    """
    rng = np.random.default_rng(seed)
    N = len(traj_list)
    stack = np.full((N, S, 2), np.nan, dtype=np.float32)

    for i, t in enumerate(traj_list):
        pts = traj_to_np(t).astype(np.float32)  # (L,2)
        if pts.shape[0] == 0:
            continue
        if S > 0 and pts.shape[0] > S:
            idx = rng.choice(pts.shape[0], size=S, replace=False)
            pts = pts[idx]
        # 如果少于 S，就前面填满，后面 NaN
        stack[i, :pts.shape[0], :] = pts
    return stack

def _phi_min_dist_all_trajs(traj_stack: np.ndarray, p: np.ndarray, eps: float = 1e-12):
    """
    traj_stack: (N,S,2) with NaN padding
    p: (2,)
    return phi: (N,) where phi[i] = min ||x - p|| over points in traj i
    """
    diff = traj_stack - p.astype(np.float32)              # (N,S,2)
    d2 = np.sum(diff * diff, axis=-1)                     # (N,S)
    # NaN padding -> use nanmin
    m = np.nanmin(d2, axis=1)                             # (N,)
    # 可能某条轨迹全是 NaN（空），nanmin 会出 nan，这里转成 +inf
    m = np.where(np.isfinite(m), m, np.inf).astype(np.float32)
    return np.sqrt(m + eps).astype(np.float32)

# -------------------------
# Candidate point pool
# -------------------------
def build_point_pool(
    traj_list: List,
    per_traj: int = 16,
    seed: int = 0,
    include_start_end: bool = True,
) -> Array:
    """Build a candidate point pool from trajectories.

    The pool is only used for *point-based* pivot selection strategies.

    Args:
        traj_list: list of trajectories, each convertible to (L,2) float array.
        per_traj: number of random points sampled from each trajectory (0 disables).
        seed: RNG seed.
        include_start_end: always include start/end points of each trajectory.

    Returns:
        points: (C,2) float32
    """
    rng = np.random.default_rng(seed)
    pts_all = []
    for traj in traj_list:
        pts = traj_to_np(traj)
        L = pts.shape[0]
        if L <= 0:
            continue
        if include_start_end:
            pts_all.append(pts[0])
            if L > 1:
                pts_all.append(pts[-1])
        if per_traj > 0 and L > 2:
            idx = rng.choice(L, size=min(per_traj, L), replace=False)
            pts_all.append(pts[idx])
    if not pts_all:
        raise ValueError("Empty point pool: check traj_list.")
    points = np.vstack([p if p.ndim == 2 else p[None, :] for p in pts_all]).astype(np.float32)
    return points

# -------------------------
# Point-based selection
# -------------------------
def select_points_random(points: Array, K: int, seed: int = 0) -> Array:
    rng = np.random.default_rng(seed)
    C = points.shape[0]
    K = min(K, C)
    idx = rng.choice(C, size=K, replace=False)
    return points[idx]

def select_points_fps(points: Array, K: int, seed: int = 0) -> Array:
    """Farthest Point Sampling (Euclidean) on a point pool."""
    rng = np.random.default_rng(seed)
    C = points.shape[0]
    K = min(K, C)
    p0 = int(rng.integers(0, C))
    piv_idx = [p0]
    diff = points - points[p0]
    min_d2 = (diff * diff).sum(axis=1)
    for _ in range(1, K):
        p = int(np.argmax(min_d2))
        piv_idx.append(p)
        diff = points - points[p]
        d2 = (diff * diff).sum(axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return points[np.array(piv_idx, dtype=np.int64)]


def select_points_fps_gpu(
    points: Array,
    K: int,
    seed: int = 0,
    device: str = "cuda:0",
) -> Array:
    """GPU-accelerated Farthest Point Sampling (Euclidean) on a point pool.
    
    Key optimization: Vectorized distance computation on GPU.
    
    Args:
        points: (C, 2) candidate points
        K: number of points to select
        seed: random seed for initial point selection
        device: "cuda:0" / "mps" / "cpu"
    
    Returns:
        selected: (K, 2) selected pivot points
    """
    import torch
    
    dev = function.resolve_torch_device(device)
    
    rng = np.random.default_rng(seed)
    C = points.shape[0]
    K = min(K, C)
    
    # Transfer points to GPU
    points_t = torch.as_tensor(points, dtype=torch.float32, device=dev)  # (C, 2)
    
    # Initialize
    p0 = int(rng.integers(0, C))
    piv_idx = [p0]
    
    # Initial distances
    diff = points_t - points_t[p0]  # (C, 2)
    min_d2 = (diff * diff).sum(dim=1)  # (C,)
    
    for _ in range(1, K):
        # Find farthest point
        p = int(torch.argmax(min_d2).item())
        piv_idx.append(p)
        
        # Update minimum distances
        diff = points_t - points_t[p]
        d2 = (diff * diff).sum(dim=1)
        min_d2 = torch.minimum(min_d2, d2)
    
    return points[np.array(piv_idx, dtype=np.int64)]


# -------------------------
# Point-based learning/selection using D_train (hard / NDCG)
# -------------------------

def _subsample_points(pts: Array, k: int, rng: np.random.Generator) -> Array:
    """Uniformly subsample points in a trajectory for speed.

    k<=0 disables subsampling.
    """
    if k is None or int(k) <= 0:
        return pts
    k = int(k)
    L = pts.shape[0]
    if L <= k:
        return pts
    idx = rng.choice(L, size=k, replace=False)
    return pts[idx]


def _min_dist_to_point(traj_pts: Array, p: Array) -> float:
    """min_{x in traj} ||x-p||_2 (float)."""
    diff = traj_pts - p[None, :]
    d2 = (diff * diff).sum(axis=1)
    return float(np.sqrt(d2.min()))


def _phi_min_dist_for_ids(
    traj_pts_list: List[Array],
    ids: Array,
    p: Array,
) -> Array:
    """Compute phi(T)=min||x-p|| for a list of trajectory ids."""
    out = np.empty((ids.shape[0],), dtype=np.float32)
    for i, tid in enumerate(ids):
        out[i] = _min_dist_to_point(traj_pts_list[int(tid)], p)
    return out

import numpy as np
from typing import List, Tuple

def _traj_to_array(traj) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(arr.shape)
    return arr

# def _subsample_points(pts: np.ndarray, m: int) -> np.ndarray:
#     if m <= 0 or pts.shape[0] <= m:
#         return pts
#     # uniform-ish subsample
#     idx = np.linspace(0, pts.shape[0]-1, m).astype(np.int64)
#     return pts[idx]

def _phi_min_dist(traj_pts: np.ndarray, p: np.ndarray) -> float:
    diff = traj_pts - p[None, :]
    d2 = (diff * diff).sum(axis=1)
    return float(np.sqrt(d2.min()))

def _sample_pairs(N: int, num_pairs: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, N, size=num_pairs, endpoint=False)
    b = rng.integers(0, N, size=num_pairs, endpoint=False)
    mask = a != b
    a, b = a[mask], b[mask]
    if a.shape[0] < num_pairs:
        extra = num_pairs - a.shape[0]
        a2, b2 = _sample_pairs(N, extra, seed + 1)
        a = np.concatenate([a, a2])
        b = np.concatenate([b, b2])
    return a[:num_pairs], b[:num_pairs]

def select_points_tightness_greedy(
    traj_list: List,
    D_train: np.ndarray,          # (N,N) true distance matrix
    points_pool: np.ndarray,      # (C,2) candidate points
    K: int = 32,
    cand_pool: int = 512,
    num_pairs: int = 20000,
    traj_point_subsample: int = 64,
    seed: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    point-tightness:
      maximize mean( LB_P(a,b) / (D[a,b]+eps) ) over sampled pairs
    where LB_P is max over pivots of |phi_p(a)-phi_p(b)|.
    """
    rng = np.random.default_rng(seed)
    N = len(traj_list)
    C = points_pool.shape[0]
    K = min(K, C)

    # sample training pairs
    a, b = _sample_pairs(N, num_pairs=num_pairs, seed=seed)
    dab = D_train[a, b].astype(np.float32) + eps

    # preconvert & subsample trajectories once
    traj_np = []
    for t in traj_list:
        pts = _traj_to_array(t)
        if traj_point_subsample and traj_point_subsample > 0:
            pts = _subsample_points(pts, traj_point_subsample)
        traj_np.append(pts)

    # current LB on pairs
    cur = np.zeros(a.shape[0], dtype=np.float32)

    # candidate indices
    rem = np.arange(C, dtype=np.int64)
    chosen = []

    for it in range(K):
        cand = rng.choice(rem, size=min(cand_pool, rem.shape[0]), replace=False)

        cur_obj = float((cur / dab).mean())
        best_ci, best_gain = None, -1e18

        for ci in cand:
            p = points_pool[ci]
            # compute phi for all sampled a,b (loop is simplest; can be vectorized later)
            phi_a = np.array([_phi_min_dist(traj_np[int(x)], p) for x in a], dtype=np.float32)
            phi_b = np.array([_phi_min_dist(traj_np[int(x)], p) for x in b], dtype=np.float32)
            diff = np.abs(phi_a - phi_b)
            new = np.maximum(cur, diff)
            obj = float((new / dab).mean())
            gain = obj - cur_obj
            if gain > best_gain:
                best_gain = gain
                best_ci = int(ci)

        # commit
        chosen.append(best_ci)
        p = points_pool[best_ci]
        phi_a = np.array([_phi_min_dist(traj_np[int(x)], p) for x in a], dtype=np.float32)
        phi_b = np.array([_phi_min_dist(traj_np[int(x)], p) for x in b], dtype=np.float32)
        cur = np.maximum(cur, np.abs(phi_a - phi_b))

        rem = rem[rem != best_ci]

    return points_pool[np.array(chosen, dtype=np.int64)]

def select_points_tightness_greedy_fast(
    traj_list,
    D_train: np.ndarray,
    points_pool: np.ndarray,
    K: int = 32,
    cand_pool: int = 512,
    num_pairs: int = 20000,
    traj_point_subsample: int = 64,
    seed: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = len(traj_list)
    C = points_pool.shape[0]
    K = min(K, C)

    # 采样 pairs
    a, b = _sample_pairs(N, num_pairs=num_pairs, seed=seed)
    dab = D_train[a, b].astype(np.float32) + eps

    # 关键：一次性堆成 (N,S,2)
    S = int(traj_point_subsample) if traj_point_subsample and traj_point_subsample > 0 else 0
    if S <= 0:
        raise ValueError("For fast path, set traj_point_subsample > 0 (e.g., 64).")
    # S = max([len(traj) for traj in traj_list])
    print("max traj length:", S)
    traj_stack = _build_traj_stack(traj_list, S=S, seed=seed)

    cur = np.zeros(a.shape[0], dtype=np.float32)
    rem = np.arange(C, dtype=np.int64)
    chosen = []

    for _ in range(K):
        cand = rng.choice(rem, size=min(cand_pool, rem.shape[0]), replace=False)
        cur_obj = float((cur / dab).mean())

        best_ci, best_gain = None, -1e18

        # 对每个候选 pivot：一次广播算 phi_all，然后直接 index 到 a,b
        for ci in cand:
            p = points_pool[int(ci)]
            phi_all = _phi_min_dist_all_trajs(traj_stack, p, eps=eps)  # (N,)
            diff = np.abs(phi_all[a] - phi_all[b])                     # (num_pairs,)
            new = np.maximum(cur, diff)
            obj = float((new / dab).mean())
            gain = obj - cur_obj
            if gain > best_gain:
                best_gain = gain
                best_ci = int(ci)

        # commit best
        chosen.append(best_ci)
        p = points_pool[best_ci]
        phi_all = _phi_min_dist_all_trajs(traj_stack, p, eps=eps)
        cur = np.maximum(cur, np.abs(phi_all[a] - phi_all[b]))

        rem = rem[rem != best_ci]

    return points_pool[np.array(chosen, dtype=np.int64)]
def select_points_tightness_greedy_gpu(
    traj_list,
    D_train: np.ndarray,          # (N,N) true distance matrix
    points_pool: np.ndarray,      # (C,2) candidate points
    K: int = 32,
    cand_pool: int = 512,
    num_pairs: int = 20000,
    traj_point_subsample: int = 64,
    seed: int = 0,
    eps: float = 1e-12,
    stack_mode: str = "bucket",   # "pad" / "bucket"
    bucket_edges: tuple = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
    cand_eval_bs: int = 64,       # Batch size for candidate evaluation
    device: str = "cuda:0",       # "cuda:0" / "mps" / "cpu"
) -> np.ndarray:
    """
    GPU-accelerated point-tightness greedy selection.
    
    Tightness objective in the paper terminology:
    maximize the average normalized lower bound over sampled trajectory pairs,
    i.e. mean( LB_P(a,b) / (D[a,b]+eps) ).

    Here LB_P(a,b) is the current pivot lower bound induced by the selected
    pivot set P:
    LB_P(a,b) = max_{p in P} |phi_p(a) - phi_p(b)|
    with phi_p(T) = min_{x in T} ||x-p||.
    
    Key optimizations (same as select_points_hard_greedy_fast_mac):
      - Stack trajectories with padding or bucketing
      - Batch evaluate candidates on GPU
      - MPS-compatible nanmin implementation
    """
    import numpy as np
    import torch

    if stack_mode not in {"pad", "bucket"}:
        raise ValueError(f"stack_mode must be 'pad' or 'bucket', got {stack_mode}")

    # Device resolution
    dev = function.resolve_torch_device(device)

    rng = np.random.default_rng(seed)
    N = len(traj_list)
    C = int(points_pool.shape[0])
    K = min(int(K), C)

    # -------------------------
    # Tightness uses uniformly sampled trajectory pairs to estimate the
    # average bound quality over the whole training distribution.
    # -------------------------
    a, b = _sample_pairs(N, num_pairs=num_pairs, seed=seed)
    num_pairs_actual = a.shape[0]

    # -------------------------
    # Get unique trajectory IDs to reduce work
    # -------------------------
    uniq_ids, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    inv_a = inv[: a.shape[0]]
    inv_b = inv[a.shape[0]:]
    
    U = int(uniq_ids.shape[0])

    # -------------------------
    # Transfer constants to GPU
    # -------------------------
    points_pool_t = torch.as_tensor(points_pool, dtype=torch.float32, device=dev)  # (C, 2)
    inv_a_t = torch.as_tensor(inv_a, dtype=torch.long, device=dev)
    inv_b_t = torch.as_tensor(inv_b, dtype=torch.long, device=dev)
    
    # dab = D[a, b] + eps on GPU
    dab = D_train[a, b].astype(np.float32) + eps
    dab_t = torch.as_tensor(dab, dtype=torch.float32, device=dev)  # (num_pairs,)
    
    # cur_t stores the current lower bound LB_P(a,b) for each sampled pair
    # under the already selected pivot set P.
    cur_t = torch.zeros((num_pairs_actual,), dtype=torch.float32, device=dev)

    remaining = np.arange(C, dtype=np.int64)
    chosen = []

    # -------------------------
    # Prepare unique trajectories (with optional subsampling)
    # -------------------------
    if traj_point_subsample and int(traj_point_subsample) > 0:
        S_fix = int(traj_point_subsample)
    else:
        S_fix = None

    traj_pts_list = []
    lengths = np.empty((U,), dtype=np.int32)

    for i, tid in enumerate(uniq_ids.tolist()):
        pts = traj_to_np(traj_list[int(tid)]).astype(np.float32)  # (L, 2)
        L = int(pts.shape[0])
        if L <= 0:
            pts = np.empty((0, 2), dtype=np.float32)
            L = 0
        else:
            if S_fix is not None and L > S_fix:
                idx = rng.choice(L, size=S_fix, replace=False)
                pts = pts[idx]
                L = S_fix
        traj_pts_list.append(pts)
        lengths[i] = L

    # -------------------------
    # Helper: MPS-friendly "nanmin" over last dim
    # -------------------------
    def _min_ignore_nan(x: torch.Tensor, dim: int):
        inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        x_safe = torch.where(torch.isnan(x), inf, x)
        return torch.min(x_safe, dim=dim).values

    # stack_t: (Ub, Sb, 2), piv_t: (B, 2) -> phi: (B, Ub)
    def _phi_bucket(stack_t: torch.Tensor, piv_t: torch.Tensor) -> torch.Tensor:
        diff = stack_t.unsqueeze(0) - piv_t[:, None, None, :]   # (B, Ub, Sb, 2)
        d2 = (diff * diff).sum(dim=-1)                          # (B, Ub, Sb)
        m = _min_ignore_nan(d2, dim=-1)                         # (B, Ub)
        # empty traj => all NaN => m=inf, ok
        return torch.sqrt(m + eps)

    # -------------------------
    # Build bucket_infos
    # -------------------------
    bucket_infos = []

    if stack_mode == "pad":
        if S_fix is None:
            S = int(lengths.max()) if int(lengths.max()) > 0 else 1
        else:
            S = int(S_fix)

        stack = np.full((U, S, 2), np.nan, dtype=np.float32)
        for ui in range(U):
            pts = traj_pts_list[ui]
            L = int(pts.shape[0])
            if L <= 0:
                continue
            stack[ui, :L, :] = pts

        idx_t = torch.arange(U, dtype=torch.long, device=dev)
        stack_t = torch.as_tensor(stack, dtype=torch.float32, device=dev)
        bucket_infos.append((idx_t, stack_t))

    else:  # bucket mode
        edges = np.array(bucket_edges, dtype=np.int32)
        bucket_id = np.searchsorted(edges, lengths, side="right")

        for bid in np.unique(bucket_id):
            idx_cpu = np.where(bucket_id == bid)[0]
            if idx_cpu.size == 0:
                continue
            Sb = int(lengths[idx_cpu].max())
            if Sb <= 0:
                stack = np.full((idx_cpu.size, 1, 2), np.nan, dtype=np.float32)
            else:
                stack = np.full((idx_cpu.size, Sb, 2), np.nan, dtype=np.float32)
                for j, ui in enumerate(idx_cpu.tolist()):
                    pts = traj_pts_list[ui]
                    L = int(pts.shape[0])
                    if L <= 0:
                        continue
                    stack[j, :L, :] = pts

            idx_t = torch.as_tensor(idx_cpu, dtype=torch.long, device=dev)
            stack_t = torch.as_tensor(stack, dtype=torch.float32, device=dev)
            bucket_infos.append((idx_t, stack_t))

    # -------------------------
    # Precompute permutation for assembling phi (MPS-safe, no index_copy_)
    # -------------------------
    all_idx = torch.cat([idx_t for idx_t, _ in bucket_infos], dim=0)
    perm = torch.argsort(all_idx)

    def _phi_all_for_pivots(piv_t: torch.Tensor) -> torch.Tensor:
        # piv_t: (B, 2) -> (B, U)
        phis = []
        for idx_t, stack_t in bucket_infos:
            phis.append(_phi_bucket(stack_t, piv_t))  # (B, Ub)
        phi_cat = torch.cat(phis, dim=1)              # (B, U) but columns follow all_idx order
        phi_all = phi_cat[:, perm]                    # reorder to 0..U-1 order
        return phi_all

    # -------------------------
    # Greedy loop
    # -------------------------
    for _ in range(K):
        # Sample candidate indices from remaining
        cand_idx = rng.choice(remaining, size=min(int(cand_pool), remaining.shape[0]), replace=False)
        cand_idx_t = torch.as_tensor(cand_idx, dtype=torch.long, device=dev)
        piv_all_t = points_pool_t.index_select(0, cand_idx_t)  # (Ncand, 2)

        # Current Tightness score: average normalized lower bound.
        cur_obj = (cur_t / dab_t).mean()

        best_gain = None
        best_ci = None

        Ncand = int(piv_all_t.shape[0])
        bs = int(cand_eval_bs)

        for s in range(0, Ncand, bs):
            e = min(Ncand, s + bs)
            piv_t = piv_all_t[s:e]  # (B, 2)
            B = int(piv_t.shape[0])

            phi_all = _phi_all_for_pivots(piv_t)  # (B, U)

            # For each candidate pivot p, compute the extra lower-bound term
            # |phi_p(a)-phi_p(b)| on every sampled pair.
            diff = torch.abs(phi_all[:, inv_a_t] - phi_all[:, inv_b_t])  # (B, num_pairs)
            # Adding a new pivot updates LB_P by a max operation, exactly
            # matching the pivot lower bound definition in the paper.
            new = torch.maximum(cur_t.unsqueeze(0), diff)                # (B, num_pairs)
            
            # Greedy gain = improvement of the Tightness objective after
            # inserting this candidate pivot into the current set P.
            obj = (new / dab_t.unsqueeze(0)).mean(dim=1)  # (B,)
            gains = obj - cur_obj  # (B,)

            gmax, arg = torch.max(gains, dim=0)
            gmax_val = float(gmax.item())
            if (best_gain is None) or (gmax_val > best_gain):
                best_gain = gmax_val
                best_ci = int(cand_idx[s + int(arg.item())])

        # Commit the best pivot, then materialize the updated lower bound
        # state so the next greedy round starts from the new pivot set P.
        chosen.append(best_ci)

        piv_best = points_pool_t[best_ci].unsqueeze(0)  # (1, 2)
        phi_best = _phi_all_for_pivots(piv_best)[0]     # (U,)

        diff_best = torch.abs(phi_best[inv_a_t] - phi_best[inv_b_t])  # (num_pairs,)
        cur_t = torch.maximum(cur_t, diff_best)

        remaining = remaining[remaining != best_ci]

    return points_pool[np.asarray(chosen, dtype=np.int64)]


def _sample_hard_pairs_from_D(
    D: Array,
    anchors: int,
    hard_k: int,
    pairs_per_anchor: int,
    seed: int,
) -> Tuple[Array, Array]:
    """Sample hard pairs (a,b) where b is among top-hard_k true neighbors of a."""
    N = D.shape[0]
    rng = np.random.default_rng(seed)
    A = rng.choice(N, size=min(int(anchors), N), replace=False)
    aa, bb = [], []
    for a in A:
        row = D[a].copy()
        row[a] = np.inf
        k = min(int(hard_k), N - 1)
        nn = np.argpartition(row, k)[:k]
        take = min(int(pairs_per_anchor), nn.shape[0])
        chosen = rng.choice(nn, size=take, replace=False)
        aa.extend([a] * take)
        bb.extend(list(chosen))
    return np.asarray(aa, dtype=np.int64), np.asarray(bb, dtype=np.int64)


def select_points_hard_greedy_fast_mac(
    traj_list,
    D_train,
    points_pool,
    K: int,
    seed: int,
    pool_cand: int = 512,
    anchors: int = 512,
    hard_k: int = 50,
    pairs_per_anchor: int = 5,
    traj_subsample: int = 64,
    stack_mode: str = "bucket",   # "pad" / "bucket"
    bucket_edges: tuple = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
    cand_eval_bs: int = 64,       # 32/64 通常适合 MPS
    device: str = "cuda:0",          # "mps" / "cpu"
    eps: float = 1e-12,
):
    """
    Hardness-aware greedy pivot selection.

    In paper terms, this variant does not optimize over uniformly sampled
    pairs. Instead, it concentrates the greedy objective on "hard" neighbor
    pairs sampled from the true distance matrix, so the selected pivots spend
    more capacity on genuinely close trajectories that are harder to separate.

    Key points:
      - No torch.nanmin (use isnan->inf + min)
      - No index_copy_/scatter_ (use concat+argsort to assemble phi)
      - Supports:
          stack_mode="pad": global padded stack (fastest when traj_subsample>0)
          stack_mode="bucket": length-bucket padding (best when long-tail lengths exist, esp traj_subsample=0)

    Recommended settings on Mac:
      - Fast & stable: traj_subsample=64 or 128, stack_mode="pad", cand_eval_bs=32/64
      - Exact-ish with long tail: traj_subsample=0, stack_mode="bucket", cand_eval_bs=32/64
    """
    import numpy as np
    import torch

    if stack_mode not in {"pad", "bucket"}:
        raise ValueError(f"stack_mode must be 'pad' or 'bucket', got {stack_mode}")

    # device resolve
    dev = function.resolve_torch_device(device)

    rng = np.random.default_rng(seed)
    C = int(points_pool.shape[0])
    K = min(int(K), C)

    # -------------------------
    # Hardness samples hard pairs (a,b) where b lies in the true near-neighbor
    # region of a according to D_train, instead of using uniform random pairs.
    # -------------------------
    a, b = _sample_hard_pairs_from_D(D_train, anchors, hard_k, pairs_per_anchor, seed=seed)
    if a.size == 0:
        return select_points_random(points_pool, K, seed=seed)

    # unique ids to reduce work
    uniq_ids, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    inv_a = inv[: a.shape[0]]
    inv_b = inv[a.shape[0] :]

    U = int(uniq_ids.shape[0])
    num_pairs = int(a.shape[0])

    # -------------------------
    # torch constants
    # -------------------------
    points_pool_t = torch.as_tensor(points_pool, dtype=torch.float32, device=dev)  # (C,2)
    inv_a_t = torch.as_tensor(inv_a, dtype=torch.long, device=dev)
    inv_b_t = torch.as_tensor(inv_b, dtype=torch.long, device=dev)
    # cur_t stores the current pivot lower bound LB_P(a,b) on the sampled hard
    # pairs under the already selected pivot set P.
    cur_t = torch.zeros((num_pairs,), dtype=torch.float32, device=dev)

    remaining = np.arange(C, dtype=np.int64)
    selected = []

    # -------------------------
    # prepare uniq trajectories on CPU (variable length)
    # -------------------------
    if traj_subsample and int(traj_subsample) > 0:
        S_fix = int(traj_subsample)
    else:
        S_fix = None

    traj_pts_list = []
    lengths = np.empty((U,), dtype=np.int32)

    for i, tid in enumerate(uniq_ids.tolist()):
        pts = traj_to_np(traj_list[int(tid)]).astype(np.float32)  # (L,2)
        L = int(pts.shape[0])
        if L <= 0:
            pts = np.empty((0, 2), dtype=np.float32)
            L = 0
        else:
            if S_fix is not None and L > S_fix:
                idx = rng.choice(L, size=S_fix, replace=False)
                pts = pts[idx]
                L = S_fix
        traj_pts_list.append(pts)
        lengths[i] = L

    # -------------------------
    # helper: MPS-friendly "nanmin" over last dim via isnan->inf + min
    # -------------------------
    def _min_ignore_nan(x: torch.Tensor, dim: int):
        inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        x_safe = torch.where(torch.isnan(x), inf, x)
        return torch.min(x_safe, dim=dim).values

    # stack_t: (Ub,Sb,2), piv_t: (B,2) -> phi: (B,Ub)
    def _phi_bucket(stack_t: torch.Tensor, piv_t: torch.Tensor) -> torch.Tensor:
        diff = stack_t.unsqueeze(0) - piv_t[:, None, None, :]   # (B,Ub,Sb,2)
        d2 = (diff * diff).sum(dim=-1)                          # (B,Ub,Sb)
        m = _min_ignore_nan(d2, dim=-1)                         # (B,Ub)
        # empty traj => all NaN => m=inf, ok
        return torch.sqrt(m + eps)

    # -------------------------
    # build bucket_infos as list of (idx_t_sorted, stack_t)
    # idx_t_sorted are UNIQUE and cover [0..U-1] across all buckets.
    # -------------------------
    bucket_infos = []

    if stack_mode == "pad":
        if S_fix is None:
            S = int(lengths.max()) if int(lengths.max()) > 0 else 1
        else:
            S = int(S_fix)

        stack = np.full((U, S, 2), np.nan, dtype=np.float32)
        for ui in range(U):
            pts = traj_pts_list[ui]
            L = int(pts.shape[0])
            if L <= 0:
                continue
            stack[ui, :L, :] = pts

        idx_t = torch.arange(U, dtype=torch.long, device=dev)  # already sorted
        stack_t = torch.as_tensor(stack, dtype=torch.float32, device=dev)
        bucket_infos.append((idx_t, stack_t))

    else:
        edges = np.array(bucket_edges, dtype=np.int32)
        bucket_id = np.searchsorted(edges, lengths, side="right")

        for bid in np.unique(bucket_id):
            idx_cpu = np.where(bucket_id == bid)[0]
            if idx_cpu.size == 0:
                continue
            # idx_cpu is sorted ascending from np.where
            Sb = int(lengths[idx_cpu].max())
            if Sb <= 0:
                # all empty -> create a dummy 1-length NaN stack
                stack = np.full((idx_cpu.size, 1, 2), np.nan, dtype=np.float32)
            else:
                stack = np.full((idx_cpu.size, Sb, 2), np.nan, dtype=np.float32)
                for j, ui in enumerate(idx_cpu.tolist()):
                    pts = traj_pts_list[ui]
                    L = int(pts.shape[0])
                    if L <= 0:
                        continue
                    stack[j, :L, :] = pts

            idx_t = torch.as_tensor(idx_cpu, dtype=torch.long, device=dev)  # sorted
            stack_t = torch.as_tensor(stack, dtype=torch.float32, device=dev)
            bucket_infos.append((idx_t, stack_t))

    # -------------------------
    # assemble phi_all without index_copy_ (MPS-safe):
    # for a pivot batch piv_t (B,2),
    #   compute phi per bucket (B,Ub), then concat -> (B,U)
    #   then sort columns by all_idx
    # We precompute all_idx + perm once (since bucket idx are fixed).
    # -------------------------
    all_idx = torch.cat([idx_t for idx_t, _ in bucket_infos], dim=0)  # (U,)
    perm = torch.argsort(all_idx)                                     # (U,)

    # Optional debug safety (can be removed for speed):
    # if device != "mps":  # even on mps it's fine, just keep commented if you want
    #     if not torch.equal(all_idx[perm], torch.arange(U, device=dev)):
    #         raise RuntimeError("Bucket indices do not cover [0..U-1] uniquely.")

    def _phi_all_for_pivots(piv_t: torch.Tensor) -> torch.Tensor:
        # piv_t: (B,2) -> (B,U)
        phis = []
        for idx_t, stack_t in bucket_infos:
            phis.append(_phi_bucket(stack_t, piv_t))  # (B,Ub)
        phi_cat = torch.cat(phis, dim=1)              # (B,U) but columns follow all_idx order
        phi_all = phi_cat[:, perm]                    # reorder to 0..U-1 order
        return phi_all

    # -------------------------
    # greedy loop
    # -------------------------
    for _ in range(K):
        # sample candidate indices from remaining (CPU)
        cand_idx = rng.choice(remaining, size=min(int(pool_cand), remaining.shape[0]), replace=False)
        cand_idx_t = torch.as_tensor(cand_idx, dtype=torch.long, device=dev)
        piv_all_t = points_pool_t.index_select(0, cand_idx_t)  # (Ncand,2)

        # Current Hardness score: mean lower bound on the sampled hard pairs.
        cur_obj = cur_t.mean()

        best_gain = None
        best_ci = None

        Ncand = int(piv_all_t.shape[0])
        bs = int(cand_eval_bs)

        for s in range(0, Ncand, bs):
            e = min(Ncand, s + bs)
            piv_t = piv_all_t[s:e]  # (B,2)
            B = int(piv_t.shape[0])

            phi_all = _phi_all_for_pivots(piv_t)  # (B,U)

            # Candidate contribution on each hard pair: |phi_p(a)-phi_p(b)|.
            diff = torch.abs(phi_all[:, inv_a_t] - phi_all[:, inv_b_t])  # (B,num_pairs)
            # The lower bound is updated by max, following
            # LB_{P U {p}}(a,b) = max(LB_P(a,b), |phi_p(a)-phi_p(b)|).
            new = torch.maximum(cur_t.unsqueeze(0), diff)                # (B,num_pairs)
            gains = new.mean(dim=1) - cur_obj                             # (B,)

            gmax, arg = torch.max(gains, dim=0)
            gmax_val = float(gmax.item())
            if (best_gain is None) or (gmax_val > best_gain):
                best_gain = gmax_val
                best_ci = int(cand_idx[s + int(arg.item())])

        # Commit the best hard-aware pivot and carry the updated lower bound to
        # the next greedy round.
        selected.append(best_ci)

        piv_best = points_pool_t[best_ci].unsqueeze(0)  # (1,2)
        phi_best = _phi_all_for_pivots(piv_best)[0]     # (U,)

        diff_best = torch.abs(phi_best[inv_a_t] - phi_best[inv_b_t])     # (num_pairs,)
        cur_t = torch.maximum(cur_t, diff_best)

        remaining = remaining[remaining != best_ci]

    return points_pool[np.asarray(selected, dtype=np.int64)]


def select_points_hard_greedy_fast_gpu(
    traj_list: List,
    D_train: Array,
    points_pool: Array,
    K: int,
    seed: int,
    pool_cand: int = 512,
    anchors: int = 512,
    hard_k: int = 50,
    pairs_per_anchor: int = 5,
    traj_subsample: int = 64,
    stack_mode: str = "bucket",   # "pad" / "bucket"
    bucket_edges: tuple = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
    cand_eval_bs: int = 64,       # 每次在 GPU 上并行评估多少个候选点（可调：32/64/128）
    device: str = "cuda:0",
    eps: float = 1e-12,
) -> Array:
    """
    GPU-accelerated hard greedy pivot selection.

    - Uses torch on CUDA to compute phi and candidate gains in batches.
    - Keeps only pad/bucket modes (NO batch mode).

    Requirements:
      - torch with CUDA
      - traj_to_np, _sample_hard_pairs_from_D, select_points_random exist as in your codebase.
    """
    import numpy as np
    import torch

    assert stack_mode in {"pad", "bucket"}, "stack_mode must be 'pad' or 'bucket'"

    rng = np.random.default_rng(seed)
    C = points_pool.shape[0]
    K = min(int(K), C)

    # -------------------------
    # sample hard pairs
    # -------------------------
    a, b = _sample_hard_pairs_from_D(D_train, anchors, hard_k, pairs_per_anchor, seed=seed)
    if a.size == 0:
        return select_points_random(points_pool, K, seed=seed)

    # unique ids to reduce work
    uniq_ids, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    inv_a = inv[: a.shape[0]]
    inv_b = inv[a.shape[0] :]

    U = uniq_ids.shape[0]
    num_pairs = a.shape[0]

    # -------------------------
    # torch constants
    # -------------------------
    dev = function.resolve_torch_device(device)
    # points_pool -> torch on GPU (float32)
    points_pool_t = torch.as_tensor(points_pool, dtype=torch.float32, device=dev)

    inv_a_t = torch.as_tensor(inv_a, dtype=torch.long, device=dev)
    inv_b_t = torch.as_tensor(inv_b, dtype=torch.long, device=dev)

    # cur over pairs on GPU
    cur_t = torch.zeros((num_pairs,), dtype=torch.float32, device=dev)

    # remaining indices on CPU (cheaper to sample/remove)
    remaining = np.arange(C, dtype=np.int64)
    selected = []

    # -------------------------
    # build representation for uniq trajectories
    # -------------------------
    if traj_subsample and int(traj_subsample) > 0:
        S_fix = int(traj_subsample)
    else:
        S_fix = None  # no subsample

    # build pts list on CPU first (variable length), then we will stack per mode
    traj_pts_list = []
    lengths = np.empty((U,), dtype=np.int32)

    for i, tid in enumerate(uniq_ids.tolist()):
        pts = traj_to_np(traj_list[int(tid)]).astype(np.float32)  # (L,2)
        L = pts.shape[0]
        if L <= 0:
            pts = np.empty((0, 2), dtype=np.float32)
        else:
            if S_fix is not None and L > S_fix:
                idx = rng.choice(L, size=S_fix, replace=False)
                pts = pts[idx]
                L = S_fix
        traj_pts_list.append(pts)
        lengths[i] = pts.shape[0]

    # helper: compute phi for a bucket stack and candidate pivots batch
    # stack: (Ub, Sb, 2) on GPU
    # piv:   (B, 2) on GPU
    # returns: (B, Ub)
    def _phi_bucket(stack_t: torch.Tensor, piv_t: torch.Tensor) -> torch.Tensor:
        """
        stack_t: (Ub, Sb, 2)  NaN padding
        piv_t:   (B, 2)
        return:  (B, Ub)
        """
        # diff: (B,Ub,Sb,2)
        diff = stack_t.unsqueeze(0) - piv_t[:, None, None, :]
        d2 = (diff * diff).sum(dim=-1)          # (B,Ub,Sb)

        # 手动 nanmin：
        # 1) 把 NaN 替换成 +inf
        inf = torch.tensor(float("inf"), device=d2.device, dtype=d2.dtype)
        d2_safe = torch.where(torch.isnan(d2), inf, d2)

        # 2) 正常 min
        m, _ = torch.min(d2_safe, dim=-1)       # (B,Ub)

        # 3) 如果一整条轨迹全是 NaN，min 会是 inf，正好符合语义
        return torch.sqrt(m + eps)

    # prepare stacks
    if stack_mode == "pad":
        # global padded stack (U,S,2)
        if S_fix is None:
            # no subsample => pad to max length among uniq (可能很大，不推荐在 Geolife 长尾下用)
            S = int(lengths.max()) if lengths.max() > 0 else 1
        else:
            S = S_fix

        traj_stack = np.full((U, S, 2), np.nan, dtype=np.float32)
        for ui in range(U):
            pts = traj_pts_list[ui]
            L = pts.shape[0]
            if L <= 0:
                continue
            traj_stack[ui, :L, :] = pts

        traj_stack_t = torch.as_tensor(traj_stack, dtype=torch.float32, device=dev)

        # one bucket that is the full U
        bucket_infos = [(torch.arange(U, device=dev), traj_stack_t)]

    else:
        # bucket mode: build per-bucket stacks (Ub,Sb,2)
        edges = np.array(bucket_edges, dtype=np.int32)
        bucket_id = np.searchsorted(edges, lengths, side="right")  # 0..len(edges)

        bucket_infos = []
        for bid in np.unique(bucket_id):
            idx_cpu = np.where(bucket_id == bid)[0]
            if idx_cpu.size == 0:
                continue
            Sb = int(lengths[idx_cpu].max())
            if Sb <= 0:
                # all empty
                idx_t = torch.as_tensor(idx_cpu, dtype=torch.long, device=dev)
                # store None stack by using empty with NaN; phi will turn to inf
                stack_t = torch.full((idx_cpu.size, 1, 2), float("nan"), dtype=torch.float32, device=dev)
                bucket_infos.append((idx_t, stack_t))
                continue

            stack = np.full((idx_cpu.size, Sb, 2), np.nan, dtype=np.float32)
            for j, ui in enumerate(idx_cpu.tolist()):
                pts = traj_pts_list[ui]
                L = pts.shape[0]
                if L <= 0:
                    continue
                stack[j, :L, :] = pts

            idx_t = torch.as_tensor(idx_cpu, dtype=torch.long, device=dev)
            stack_t = torch.as_tensor(stack, dtype=torch.float32, device=dev)
            bucket_infos.append((idx_t, stack_t))

    # -------------------------
    # greedy loop
    # -------------------------
    for _ in range(K):
        # sample candidates from remaining (CPU)
        cand_idx = rng.choice(remaining, size=min(int(pool_cand), remaining.shape[0]), replace=False)
        cand_idx_t = torch.as_tensor(cand_idx, dtype=torch.long, device=dev)
        piv_all_t = points_pool_t.index_select(0, cand_idx_t)  # (Ncand,2)

        cur_obj = cur_t.mean()
        best_gain = None
        best_ci = None

        # evaluate candidates in chunks on GPU
        Ncand = piv_all_t.shape[0]
        for s in range(0, Ncand, int(cand_eval_bs)):
            e = min(Ncand, s + int(cand_eval_bs))
            piv_t = piv_all_t[s:e]  # (B,2)
            B = piv_t.shape[0]

            # assemble phi_all for this chunk: (B,U)
            phi_all = torch.empty((B, U), dtype=torch.float32, device=dev)

            for idx_t, stack_t in bucket_infos:
                phi_b = _phi_bucket(stack_t, piv_t)          # (B,Ub)
                phi_all.index_copy_(1, idx_t, phi_b)         # fill columns for those traj ids

            diff = torch.abs(phi_all[:, inv_a_t] - phi_all[:, inv_b_t])   # (B,num_pairs)
            new = torch.maximum(cur_t.unsqueeze(0), diff)                 # (B,num_pairs)
            gains = new.mean(dim=1) - cur_obj                              # (B,)

            # find best within this chunk
            gmax, arg = torch.max(gains, dim=0)
            gmax_val = float(gmax.item())
            if (best_gain is None) or (gmax_val > best_gain):
                best_gain = gmax_val
                best_ci = int(cand_idx[s + int(arg.item())])

        # commit best
        selected.append(best_ci)

        # update cur using best pivot (compute phi for 1 pivot but still via chunk code)
        piv_best_t = points_pool_t[best_ci].unsqueeze(0)  # (1,2)
        phi_best = torch.empty((1, U), dtype=torch.float32, device=dev)
        for idx_t, stack_t in bucket_infos:
            phi_b = _phi_bucket(stack_t, piv_best_t)      # (1,Ub)
            phi_best.index_copy_(1, idx_t, phi_b)
        diff_best = torch.abs(phi_best[0, inv_a_t] - phi_best[0, inv_b_t])  # (num_pairs,)
        cur_t = torch.maximum(cur_t, diff_best)

        # remove from remaining (CPU)
        remaining = remaining[remaining != best_ci]

    # return as numpy
    return points_pool[np.asarray(selected, dtype=np.int64)]


def select_points_hard_greedy_fast(
    traj_list: List,
    D_train: Array,
    points_pool: Array,
    K: int,
    seed: int,
    pool_cand: int = 512,
    anchors: int = 512,
    hard_k: int = 50,
    pairs_per_anchor: int = 5,
    traj_subsample: int = 64,
    stack_mode: str = "bucket",     # "pad" / "batch" / "bucket"
    batch_size: int = 256,          # stack_mode="batch" 时生效
    bucket_edges: tuple = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
) -> Array:
    """
    Hard-region-aware greedy selection of pivot points.

    stack_mode:
      - "pad": 预先构造 (U,S,2) padded stack，S 通常 = traj_subsample（快，但若 traj_subsample=0 且长尾很长会爆）
      - "batch": 不构造 padded stack；phi_all(p) 逐条轨迹按 batch 计算（稳但更慢）
      - "bucket": 按轨迹长度分桶；桶内 padding + 向量化（长尾稳 + 速度接近 pad，推荐）

    traj_subsample:
      - >0：对每条轨迹随机抽样到 S=traj_subsample（推荐 64/128）
      - =0：不抽样，"bucket" 会按真实长度分桶，不会全局 pad 到 max_len
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    C = points_pool.shape[0]
    K = min(int(K), C)

    # -------------------------
    # sample hard pairs
    # -------------------------
    a, b = _sample_hard_pairs_from_D(D_train, anchors, hard_k, pairs_per_anchor, seed=seed)
    if a.size == 0:
        return select_points_random(points_pool, K, seed=seed)

    uniq_ids, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    inv_a = inv[: a.shape[0]]
    inv_b = inv[a.shape[0] :]

    U = uniq_ids.shape[0]

    if stack_mode not in {"pad", "batch", "bucket"}:
        raise ValueError(f"stack_mode must be 'pad' / 'batch' / 'bucket', got {stack_mode}")

    # -------------------------
    # helpers
    # -------------------------
    def _phi_from_stack(traj_stack: np.ndarray, p: np.ndarray) -> np.ndarray:
        """traj_stack: (N,S,2) with NaN padding -> phi: (N,)"""
        p = np.asarray(p, dtype=np.float32).reshape(1, 1, 2)
        diff = traj_stack - p
        d2 = np.sum(diff * diff, axis=-1)         # (N,S)
        m = np.nanmin(d2, axis=1)                 # (N,)
        m = np.where(np.isfinite(m), m, np.inf).astype(np.float32)
        return np.sqrt(m + 1e-12).astype(np.float32)

    # ==========================================================
    # Mode 1: "pad" (global padded stack)
    # ==========================================================
    if stack_mode == "pad":
        if traj_subsample and int(traj_subsample) > 0:
            S = int(traj_subsample)
        else:
            S = 0
            for tid in uniq_ids.tolist():
                pts = traj_to_np(traj_list[int(tid)])
                S = max(S, pts.shape[0])
            if S <= 0:
                return select_points_random(points_pool, K, seed=seed)

        traj_stack = np.full((U, S, 2), np.nan, dtype=np.float32)
        for ui, tid in enumerate(uniq_ids.tolist()):
            pts = traj_to_np(traj_list[int(tid)]).astype(np.float32)
            L = pts.shape[0]
            if L <= 0:
                continue
            if traj_subsample and int(traj_subsample) > 0 and L > S:
                idx = rng.choice(L, size=S, replace=False)
                pts = pts[idx]
                L = S
            traj_stack[ui, :L, :] = pts

        def phi_all(p: np.ndarray) -> np.ndarray:
            return _phi_from_stack(traj_stack, p)

    # ==========================================================
    # Mode 2: "batch" (no padding; loop inside each batch)
    # ==========================================================
    elif stack_mode == "batch":
        traj_pts_list = []
        S = int(traj_subsample) if traj_subsample and int(traj_subsample) > 0 else None

        for tid in uniq_ids.tolist():
            pts = traj_to_np(traj_list[int(tid)]).astype(np.float32)
            L = pts.shape[0]
            if L <= 0:
                traj_pts_list.append(np.empty((0, 2), dtype=np.float32))
                continue
            if S is not None and L > S:
                idx = rng.choice(L, size=S, replace=False)
                pts = pts[idx]
            traj_pts_list.append(pts)

        def phi_all(p: np.ndarray) -> np.ndarray:
            p = np.asarray(p, dtype=np.float32).reshape(1, 2)
            out = np.empty((U,), dtype=np.float32)
            for s in range(0, U, int(batch_size)):
                e = min(U, s + int(batch_size))
                for i in range(s, e):
                    pts = traj_pts_list[i]
                    if pts.shape[0] == 0:
                        out[i] = np.inf
                        continue
                    diff = pts - p
                    d2 = np.sum(diff * diff, axis=1)
                    out[i] = float(np.sqrt(np.min(d2) + 1e-12))
            return out

    # ==========================================================
    # Mode 3: "bucket" (length-bucket + vectorized per bucket)
    # ==========================================================
    else:
        # 1) build pts list (optionally subsample)
        traj_pts_list = []
        if traj_subsample and int(traj_subsample) > 0:
            S_fix = int(traj_subsample)
        else:
            S_fix = None

        lengths = np.empty((U,), dtype=np.int32)
        for i, tid in enumerate(uniq_ids.tolist()):
            pts = traj_to_np(traj_list[int(tid)]).astype(np.float32)
            L = pts.shape[0]
            if L <= 0:
                pts = np.empty((0, 2), dtype=np.float32)
            else:
                if S_fix is not None and L > S_fix:
                    idx = rng.choice(L, size=S_fix, replace=False)
                    pts = pts[idx]
                    L = S_fix
            traj_pts_list.append(pts)
            lengths[i] = pts.shape[0]

        # 2) make bucket id for each traj by length
        # bucket_edges: (16,32,64,...) -> bins: [0..16],(16..32],...,(last..inf]
        edges = np.array(bucket_edges, dtype=np.int32)
        bucket_id = np.searchsorted(edges, lengths, side="right")  # 0..len(edges)

        # 3) build per-bucket stacks (each bucket has its own max len)
        buckets = {}
        # buckets[bid] = dict(idx=array(Ub), stack=(Ub,Sb,2))
        for bid in np.unique(bucket_id):
            idx = np.where(bucket_id == bid)[0]
            if idx.size == 0:
                continue
            # bucket max length (avoid 0 -> make Sb=1 so stack exists)
            Sb = int(np.max(lengths[idx]))
            if Sb <= 0:
                # all empty in this bucket
                buckets[int(bid)] = {"idx": idx, "stack": None}
                continue

            stack = np.full((idx.size, Sb, 2), np.nan, dtype=np.float32)
            for j, ui in enumerate(idx.tolist()):
                pts = traj_pts_list[ui]
                Lj = pts.shape[0]
                if Lj <= 0:
                    continue
                stack[j, :Lj, :] = pts
            buckets[int(bid)] = {"idx": idx, "stack": stack}

        def phi_all(p: np.ndarray) -> np.ndarray:
            out = np.empty((U,), dtype=np.float32)
            for bid, info in buckets.items():
                idx = info["idx"]
                stack = info["stack"]
                if stack is None:
                    out[idx] = np.inf
                else:
                    out[idx] = _phi_from_stack(stack, p)
            return out

    # -------------------------
    # greedy selection
    # -------------------------
    cur = np.zeros((a.shape[0],), dtype=np.float32)
    selected = []
    remaining = np.arange(C, dtype=np.int64)

    for _ in range(K):
        cand_idx = rng.choice(remaining, size=min(int(pool_cand), remaining.shape[0]), replace=False)
        cur_obj = float(cur.mean())
        best, best_gain = None, -1e18

        for ci in cand_idx:
            p = points_pool[int(ci)]
            phi = phi_all(p)  # (U,)
            diff = np.abs(phi[inv_a] - phi[inv_b])
            new = np.maximum(cur, diff)
            gain = float(new.mean() - cur_obj)
            if gain > best_gain:
                best_gain = gain
                best = int(ci)

        selected.append(best)
        phi = phi_all(points_pool[best])
        cur = np.maximum(cur, np.abs(phi[inv_a] - phi[inv_b]))
        remaining = remaining[remaining != best]

    return points_pool[np.asarray(selected, dtype=np.int64)]


def select_points_hard_greedy(
    traj_list: List,
    D_train: Array,
    points_pool: Array,
    K: int,
    seed: int,
    pool_cand: int = 512,
    anchors: int = 512,
    hard_k: int = 50,
    pairs_per_anchor: int = 5,
    traj_subsample: int = 0,
) -> Array:
    """Hard-region-aware greedy selection of pivot *points*.

    Objective: maximize mean of current proxy lower-bound on hard pairs.
        proxy_LB_p(a,b) = |phi_p(a)-phi_p(b)|,
        phi_p(T) = min_{x in T} ||x-p||.
        LB_P = max_{p in P} proxy_LB_p.

    Note: This is a *selection-time proxy*. You can still evaluate with your
    DTW/Fréchet pivot-LB downstream.
    """
    rng = np.random.default_rng(seed)
    C = points_pool.shape[0]
    K = min(int(K), C)

    # prepare trajectory point arrays (optionally subsampled)
    traj_pts_list = []
    for t in traj_list:
        pts = traj_to_np(t)
        pts = _subsample_points(pts, traj_subsample, rng)
        traj_pts_list.append(pts)

    a, b = _sample_hard_pairs_from_D(D_train, anchors, hard_k, pairs_per_anchor, seed=seed)
    if a.size == 0:
        # fall back to random points if sampling failed
        return select_points_random(points_pool, K, seed=seed)

    # unique ids to avoid recomputing phi twice
    uniq_ids, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    inv_a = inv[: a.shape[0]]
    inv_b = inv[a.shape[0] :]

    cur = np.zeros((a.shape[0],), dtype=np.float32)
    selected = []
    remaining = np.arange(C, dtype=np.int64)

    for _ in range(K):
        cand_idx = rng.choice(remaining, size=min(int(pool_cand), remaining.shape[0]), replace=False)
        cur_obj = float(cur.mean())
        best, best_gain = None, -1e18

        for ci in cand_idx:
            p = points_pool[int(ci)]
            phi = _phi_min_dist_for_ids(traj_pts_list, uniq_ids, p)  # (U,)
            diff = np.abs(phi[inv_a] - phi[inv_b])
            new = np.maximum(cur, diff)
            gain = float(new.mean() - cur_obj)
            if gain > best_gain:
                best_gain = gain
                best = int(ci)

        selected.append(best)
        p = points_pool[best]
        phi = _phi_min_dist_for_ids(traj_pts_list, uniq_ids, p)
        diff = np.abs(phi[inv_a] - phi[inv_b])
        cur = np.maximum(cur, diff)
        remaining = remaining[remaining != best]

    return points_pool[np.asarray(selected, dtype=np.int64)]


def _ndcg_at_k_from_scores(true_dist: Array, pred_score: Array, k: int) -> float:
    """NDCG@k where smaller true_dist is more relevant and smaller pred_score is ranked first."""
    order = np.argsort(pred_score, kind="stable")
    # relevance by true rank (monotone)
    true_order = np.argsort(true_dist, kind="stable")
    rel = np.empty_like(true_order, dtype=np.float32)
    # highest relevance for smallest distance
    rel[true_order] = (np.arange(true_order.size, 0, -1) / true_order.size).astype(np.float32)
    rel_sorted = rel[order][:k]

    gains = (2.0 ** rel_sorted - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, rel_sorted.size + 2))
    dcg = float((gains * discounts).sum())

    ideal = np.sort(rel)[::-1][:k]
    idcg = float(((2.0 ** ideal - 1.0) * discounts[: ideal.size]).sum()) + 1e-12
    return dcg / idcg


def select_points_ndcg_greedy(
    traj_list: List,
    D_train: Array,
    points_pool: Array,
    K: int,
    seed: int,
    pool_cand: int = 256,
    anchors: int = 256,
    eval_k: int = 50,
    ndcg_at: int = 10,
    neg_num: int = 50,
    traj_subsample: int = 0,
) -> Array:
    """NDCG-aware greedy selection of pivot *points*.

    We evaluate each candidate pivot by the improvement in mean NDCG@ndcg_at
    on a small per-anchor candidate set (true top-eval_k neighbors + neg_num random negatives).

    Proxy score uses L_infty on phi_p(T)=min||x-p||:
        score_P(a,c) = max_{p in P} |phi_p(a)-phi_p(c)|.
    """
    rng = np.random.default_rng(seed)
    N = D_train.shape[0]
    C = points_pool.shape[0]
    K = min(int(K), C)

    traj_pts_list = []
    for t in traj_list:
        pts = traj_to_np(t)
        pts = _subsample_points(pts, traj_subsample, rng)
        traj_pts_list.append(pts)

    A = rng.choice(N, size=min(int(anchors), N), replace=False)

    # build per-anchor candidate sets
    cand_ids = []
    cand_true_dist = []
    for a in A:
        row = D_train[a].copy()
        row[a] = np.inf
        kk = min(int(eval_k), N - 1)
        top = np.argpartition(row, kk)[:kk]
        # add random negatives (avoid duplicates)
        neg = []
        if int(neg_num) > 0:
            while len(neg) < int(neg_num):
                j = int(rng.integers(0, N))
                if j != a and j not in top:
                    neg.append(j)
        ids = np.concatenate([top, np.asarray(neg, dtype=np.int64)])
        cand_ids.append(ids)
        cand_true_dist.append(row[ids].astype(np.float32))
    # pad to rectangular (A, M)
    M = max(x.size for x in cand_ids)
    cand_mat = np.full((A.size, M), -1, dtype=np.int64)
    true_mat = np.full((A.size, M), np.inf, dtype=np.float32)
    for i, (ids, td) in enumerate(zip(cand_ids, cand_true_dist)):
        cand_mat[i, : ids.size] = ids
        true_mat[i, : ids.size] = td

    # current scores with selected pivots
    cur_score = np.zeros((A.size, M), dtype=np.float32)
    selected = []
    remaining = np.arange(C, dtype=np.int64)

    def mean_ndcg(score_mat: Array) -> float:
        s = 0.0
        for i in range(A.size):
            valid = cand_mat[i] >= 0
            s += _ndcg_at_k_from_scores(true_mat[i, valid], score_mat[i, valid], k=min(int(ndcg_at), valid.sum()))
        return s / max(1, A.size)

    cur_ndcg = mean_ndcg(cur_score)

    # precompute unique trajectory ids needed each iteration
    uniq_ids = np.unique(np.concatenate([A, cand_mat[cand_mat >= 0]]))
    # mapping from traj id -> row index in uniq_ids
    id2pos = {int(tid): i for i, tid in enumerate(uniq_ids.tolist())}
    A_pos = np.asarray([id2pos[int(t)] for t in A], dtype=np.int64)
    cand_pos = np.full_like(cand_mat, -1, dtype=np.int64)
    for i in range(cand_mat.shape[0]):
        for j in range(cand_mat.shape[1]):
            tid = int(cand_mat[i, j])
            if tid >= 0:
                cand_pos[i, j] = id2pos[tid]

    for _ in range(K):
        cand_idx = rng.choice(remaining, size=min(int(pool_cand), remaining.shape[0]), replace=False)
        best, best_gain = None, -1e18

        for ci in cand_idx:
            p = points_pool[int(ci)]
            phi = _phi_min_dist_for_ids(traj_pts_list, uniq_ids, p)

            # update score matrix cheaply: score = max(old, |phi(a)-phi(c)|)
            diff = np.abs(phi[cand_pos] - phi[A_pos][:, None])
            new_score = np.maximum(cur_score, diff)
            nd = mean_ndcg(new_score)
            gain = nd - cur_ndcg
            if gain > best_gain:
                best_gain = gain
                best = int(ci)

        selected.append(best)
        p = points_pool[best]
        phi = _phi_min_dist_for_ids(traj_pts_list, uniq_ids, p)
        diff = np.abs(phi[cand_pos] - phi[A_pos][:, None])
        cur_score = np.maximum(cur_score, diff)
        cur_ndcg = mean_ndcg(cur_score)
        remaining = remaining[remaining != best]

    return points_pool[np.asarray(selected, dtype=np.int64)]


def select_points_ndcg_greedy_gpu(
    traj_list: List,
    D_train: Array,
    points_pool: Array,
    K: int,
    seed: int,
    pool_cand: int = 256,
    anchors: int = 256,
    eval_k: int = 50,
    ndcg_at: int = 10,
    neg_num: int = 50,
    traj_subsample: int = 64,
    stack_mode: str = "bucket",
    bucket_edges: tuple = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
    cand_eval_bs: int = 64,
    device: str = "cuda:0",
    eps: float = 1e-12,
) -> Array:
    """GPU-accelerated NDCG-aware greedy selection of pivot points.

    Key optimizations:
      - Stack trajectories with bucketing for GPU batch processing
      - Batch evaluate candidates on GPU
      - MPS-compatible operations

    Args:
        traj_list: list of trajectories
        D_train: (N,N) true distance matrix
        points_pool: (C,2) candidate points
        K: number of pivots to select
        seed: random seed
        pool_cand: candidates to evaluate per iteration
        anchors: number of anchor trajectories
        eval_k: top-k neighbors per anchor
        ndcg_at: NDCG@k parameter
        neg_num: negative samples per anchor
        traj_subsample: trajectory point subsampling
        stack_mode: "pad" / "bucket"
        bucket_edges: bucket boundaries
        cand_eval_bs: candidate batch size on GPU
        device: "cuda:0" / "mps" / "cpu"
        eps: numerical stability

    Returns:
        pivots: (K, 2) selected pivot points
    """
    import torch

    if stack_mode not in {"pad", "bucket"}:
        raise ValueError(f"stack_mode must be 'pad' or 'bucket', got {stack_mode}")

    # Device resolution
    dev = function.resolve_torch_device(device)

    rng = np.random.default_rng(seed)
    N = D_train.shape[0]
    C = points_pool.shape[0]
    K = min(int(K), C)

    # Sample anchor trajectories
    A = rng.choice(N, size=min(int(anchors), N), replace=False)

    # Build per-anchor candidate sets
    cand_ids = []
    cand_true_dist = []
    for a in A:
        row = D_train[a].copy()
        row[a] = np.inf
        kk = min(int(eval_k), N - 1)
        top = np.argpartition(row, kk)[:kk]
        neg = []
        if int(neg_num) > 0:
            while len(neg) < int(neg_num):
                j = int(rng.integers(0, N))
                if j != a and j not in top:
                    neg.append(j)
        ids = np.concatenate([top, np.asarray(neg, dtype=np.int64)])
        cand_ids.append(ids)
        cand_true_dist.append(row[ids].astype(np.float32))

    # Pad to rectangular (len(A), M)
    M = max(x.size for x in cand_ids)
    cand_mat = np.full((A.size, M), -1, dtype=np.int64)
    true_mat = np.full((A.size, M), np.inf, dtype=np.float32)
    for i, (ids, td) in enumerate(zip(cand_ids, cand_true_dist)):
        cand_mat[i, : ids.size] = ids
        true_mat[i, : ids.size] = td

    # Unique trajectory IDs needed
    uniq_ids = np.unique(np.concatenate([A, cand_mat[cand_mat >= 0]]))
    id2pos = {int(tid): i for i, tid in enumerate(uniq_ids.tolist())}
    A_pos = np.asarray([id2pos[int(t)] for t in A], dtype=np.int64)
    cand_pos = np.full_like(cand_mat, -1, dtype=np.int64)
    for i in range(cand_mat.shape[0]):
        for j in range(cand_mat.shape[1]):
            tid = int(cand_mat[i, j])
            if tid >= 0:
                cand_pos[i, j] = id2pos[tid]

    U = int(uniq_ids.shape[0])

    # Prepare trajectory points
    S_fix = int(traj_subsample) if traj_subsample and int(traj_subsample) > 0 else None
    traj_pts_list = []
    lengths = np.empty((U,), dtype=np.int32)

    for i, tid in enumerate(uniq_ids.tolist()):
        pts = traj_to_np(traj_list[int(tid)]).astype(np.float32)
        L = int(pts.shape[0])
        if L <= 0:
            pts = np.empty((0, 2), dtype=np.float32)
            L = 0
        else:
            if S_fix is not None and L > S_fix:
                idx = rng.choice(L, size=S_fix, replace=False)
                pts = pts[idx]
                L = S_fix
        traj_pts_list.append(pts)
        lengths[i] = L

    # Transfer to GPU
    points_pool_t = torch.as_tensor(points_pool, dtype=torch.float32, device=dev)
    A_pos_t = torch.as_tensor(A_pos, dtype=torch.long, device=dev)
    cand_pos_t = torch.as_tensor(cand_pos, dtype=torch.long, device=dev)
    true_mat_t = torch.as_tensor(true_mat, dtype=torch.float32, device=dev)
    valid_mask_t = torch.as_tensor(cand_mat >= 0, dtype=torch.bool, device=dev)

    # MPS-friendly nanmin
    def _min_ignore_nan(x: torch.Tensor, dim: int):
        inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        x_safe = torch.where(torch.isnan(x), inf, x)
        return torch.min(x_safe, dim=dim).values

    def _phi_bucket(stack_t: torch.Tensor, piv_t: torch.Tensor) -> torch.Tensor:
        diff = stack_t.unsqueeze(0) - piv_t[:, None, None, :]
        d2 = (diff * diff).sum(dim=-1)
        m = _min_ignore_nan(d2, dim=-1)
        return torch.sqrt(m + eps)

    # Build bucket infos
    bucket_infos = []
    if stack_mode == "pad":
        if S_fix is None:
            S = int(lengths.max()) if int(lengths.max()) > 0 else 1
        else:
            S = int(S_fix)
        stack = np.full((U, S, 2), np.nan, dtype=np.float32)
        for ui in range(U):
            pts = traj_pts_list[ui]
            L = int(pts.shape[0])
            if L > 0:
                stack[ui, :L, :] = pts
        idx_t = torch.arange(U, dtype=torch.long, device=dev)
        stack_t = torch.as_tensor(stack, dtype=torch.float32, device=dev)
        bucket_infos.append((idx_t, stack_t))
    else:
        edges = np.array(bucket_edges, dtype=np.int32)
        bucket_id = np.searchsorted(edges, lengths, side="right")
        for bid in np.unique(bucket_id):
            idx_cpu = np.where(bucket_id == bid)[0]
            if idx_cpu.size == 0:
                continue
            Sb = int(lengths[idx_cpu].max())
            if Sb <= 0:
                stack = np.full((idx_cpu.size, 1, 2), np.nan, dtype=np.float32)
            else:
                stack = np.full((idx_cpu.size, Sb, 2), np.nan, dtype=np.float32)
                for j, ui in enumerate(idx_cpu.tolist()):
                    pts = traj_pts_list[ui]
                    L = int(pts.shape[0])
                    if L > 0:
                        stack[j, :L, :] = pts
            idx_t = torch.as_tensor(idx_cpu, dtype=torch.long, device=dev)
            stack_t = torch.as_tensor(stack, dtype=torch.float32, device=dev)
            bucket_infos.append((idx_t, stack_t))

    all_idx = torch.cat([idx_t for idx_t, _ in bucket_infos], dim=0)
    perm = torch.argsort(all_idx)

    def _phi_all_for_pivots(piv_t: torch.Tensor) -> torch.Tensor:
        phis = []
        for idx_t, stack_t in bucket_infos:
            phis.append(_phi_bucket(stack_t, piv_t))
        phi_cat = torch.cat(phis, dim=1)
        return phi_cat[:, perm]

    # NDCG helper - simplified version for GPU
    def _mean_ndcg_gpu(cur_score: torch.Tensor) -> float:
        """Compute mean NDCG@k on GPU."""
        A_size = cur_score.shape[0]
        total_ndcg = 0.0
        
        for i in range(A_size):
            valid = valid_mask_t[i]
            if valid.sum() == 0:
                continue
            
            td = true_mat_t[i, valid]
            ps = cur_score[i, valid]
            n = int(td.shape[0])
            k_use = min(ndcg_at, n)
            
            # Sort by pred_score (ascending)
            order = torch.argsort(ps)
            true_order = torch.argsort(td)
            rel = torch.zeros_like(td)
            rel[true_order] = torch.linspace(float(n), 1.0, n, device=dev, dtype=torch.float32) / n
            rel_sorted = rel[order][:k_use]
            
            gains = (2.0 ** rel_sorted - 1.0)
            discounts = 1.0 / torch.log2(torch.arange(2, k_use + 2, device=dev, dtype=torch.float32))
            dcg = (gains * discounts).sum()
            
            ideal = torch.sort(rel, descending=True).values[:k_use]
            idcg = ((2.0 ** ideal - 1.0) * discounts[:ideal.shape[0]]).sum() + 1e-12
            total_ndcg += float((dcg / idcg).item())
        
        return total_ndcg / max(1, A_size)

    # Current score matrix
    cur_score_t = torch.zeros((A.size, M), dtype=torch.float32, device=dev)
    cur_ndcg = _mean_ndcg_gpu(cur_score_t)

    remaining = np.arange(C, dtype=np.int64)
    selected = []

    for _ in range(K):
        cand_idx = rng.choice(remaining, size=min(int(pool_cand), remaining.shape[0]), replace=False)
        cand_idx_t = torch.as_tensor(cand_idx, dtype=torch.long, device=dev)
        piv_all_t = points_pool_t.index_select(0, cand_idx_t)

        best_gain = None
        best_ci = None

        Ncand = int(piv_all_t.shape[0])
        bs = int(cand_eval_bs)

        for s in range(0, Ncand, bs):
            e = min(Ncand, s + bs)
            piv_t = piv_all_t[s:e]
            B = int(piv_t.shape[0])

            phi_all = _phi_all_for_pivots(piv_t)  # (B, U)

            for b in range(B):
                phi = phi_all[b]  # (U,)
                # Compute score update
                diff = torch.abs(phi[cand_pos_t] - phi[A_pos_t][:, None])
                new_score = torch.maximum(cur_score_t, diff)
                
                nd = _mean_ndcg_gpu(new_score)
                gain = nd - cur_ndcg
                
                if (best_gain is None) or (gain > best_gain):
                    best_gain = gain
                    best_ci = int(cand_idx[s + b])

        selected.append(best_ci)
        
        # Update current score
        piv_best = points_pool_t[best_ci].unsqueeze(0)
        phi_best = _phi_all_for_pivots(piv_best)[0]
        diff_best = torch.abs(phi_best[cand_pos_t] - phi_best[A_pos_t][:, None])
        cur_score_t = torch.maximum(cur_score_t, diff_best)
        cur_ndcg = _mean_ndcg_gpu(cur_score_t)
        
        remaining = remaining[remaining != best_ci]

    return points_pool[np.asarray(selected, dtype=np.int64)]


def _sqdist_to_centers(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    points: (C,2), centers: (B,2) -> (C,B) squared distances
    """
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    px = (points * points).sum(axis=1, keepdims=True)      # (C,1)
    cy = (centers * centers).sum(axis=1, keepdims=True).T  # (1,B)
    d2 = px + cy - 2.0 * (points @ centers.T)              # (C,B)
    np.maximum(d2, 0.0, out=d2)
    return d2

def select_points_facility_location(
    points_pool: np.ndarray,
    K: int,
    sigma: Optional[float] = None,
    cand_pool: int = 1024,
    seed: int = 0,
) -> np.ndarray:
    """
    Greedy facility location on point pool.
    points_pool: (C,2)
    Returns pivots: (K,2)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(points_pool, dtype=np.float32)
    C = X.shape[0]
    K = min(K, C)

    # Heuristic sigma: median distance between random pairs
    if sigma is None:
        m = min(2000, C)
        idx1 = rng.choice(C, size=m, replace=False)
        idx2 = rng.choice(C, size=m, replace=False)
        d2 = ((X[idx1] - X[idx2]) ** 2).sum(axis=1)
        sigma = float(np.sqrt(np.median(d2) + 1e-12))
        if sigma <= 1e-6:
            sigma = 1.0

    inv2sig2 = 1.0 / (2.0 * sigma * sigma)

    selected = []
    remaining = np.arange(C, dtype=np.int64)

    # maintain best similarity for each point to current selected set
    best_sim = np.zeros(C, dtype=np.float32)  # max_j s(i, sel_j)

    for it in range(K):
        # candidate subset for speed
        if remaining.shape[0] > cand_pool:
            cand = rng.choice(remaining, size=cand_pool, replace=False)
        else:
            cand = remaining

        # compute marginal gains for candidates
        best_gain = -1e18
        best_idx = None

        # evaluate each candidate
        # gain(p) = sum_i max(best_sim[i], sim(i,p)) - sum_i best_sim[i]
        cur_sum = float(best_sim.sum())

        # compute sim(i,p) in a vectorized way per candidate
        for ci in cand:
            p = X[ci:ci+1]                 # (1,2)
            d2 = _sqdist_to_centers(X, p)[:, 0]  # (C,)
            sim = np.exp(-d2 * inv2sig2).astype(np.float32)
            new_sum = float(np.maximum(best_sim, sim).sum())
            gain = new_sum - cur_sum
            if gain > best_gain:
                best_gain = gain
                best_idx = int(ci)

        # commit
        selected.append(best_idx)
        p = X[best_idx:best_idx+1]
        d2 = _sqdist_to_centers(X, p)[:, 0]
        sim = np.exp(-d2 * inv2sig2).astype(np.float32)
        best_sim = np.maximum(best_sim, sim)

        remaining = remaining[remaining != best_idx]

    return X[np.array(selected, dtype=np.int64)]

def _dist2_point_to_points(X: np.ndarray, p: np.ndarray) -> np.ndarray:
    d = X - p[None, :]
    return (d * d).sum(axis=1)

def _kmedoids_pp_init(X: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """
    k-medoids++ like init using squared distances.
    Returns medoid indices (K,)
    """
    rng = np.random.default_rng(seed)
    C = X.shape[0]
    K = min(K, C)

    medoids = [int(rng.integers(0, C))]
    min_d2 = _dist2_point_to_points(X, X[medoids[0]])

    for _ in range(1, K):
        # sample next proportional to min_d2
        probs = min_d2 / (min_d2.sum() + 1e-12)
        nxt = int(rng.choice(C, p=probs))
        medoids.append(nxt)
        min_d2 = np.minimum(min_d2, _dist2_point_to_points(X, X[nxt]))
    return np.array(medoids, dtype=np.int64)

def select_points_kmedoids(
    points_pool: np.ndarray,
    K: int,
    max_iter: int = 20,
    seed: int = 0,
    update_sample: int = 2000,  # per cluster medoid update sample cap (speed)
) -> np.ndarray:
    """
    Practical k-medoids on points_pool using:
      - k-medoids++ init
      - iterative assignment + medoid update by within-cluster 1-median (approx via sampling)

    Returns pivots: (K,2)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(points_pool, dtype=np.float32)
    C = X.shape[0]
    K = min(K, C)

    med = _kmedoids_pp_init(X, K, seed=seed)  # indices
    med_pts = X[med]                          # (K,2)

    # precompute X norms for fast distance calc
    X_sq = (X * X).sum(axis=1)  # (C,)

    def dist2_to_medoids(med_pts_: np.ndarray) -> np.ndarray:
        med_sq = (med_pts_ * med_pts_).sum(axis=1)  # (K,)
        d2 = X_sq[:, None] + med_sq[None, :] - 2.0 * (X @ med_pts_.T)
        np.maximum(d2, 0.0, out=d2)
        return d2

    prev_assign = None

    for it in range(max_iter):
        d2 = dist2_to_medoids(med_pts)
        assign = np.argmin(d2, axis=1)  # (C,)

        if prev_assign is not None and np.array_equal(assign, prev_assign):
            break
        prev_assign = assign

        # update medoid of each cluster
        new_med = med.copy()
        for k in range(K):
            idx = np.where(assign == k)[0]
            if idx.size == 0:
                # empty cluster: re-seed to farthest point
                far = int(np.argmax(np.min(d2, axis=1)))
                new_med[k] = far
                continue

            # sample candidates within cluster to speed up
            if idx.size > update_sample:
                cand = rng.choice(idx, size=update_sample, replace=False)
            else:
                cand = idx

            # choose point minimizing sum of distances to others in cluster (approx)
            # compute pairwise distances from cand to all idx (block via formula)
            X_idx = X[idx]                 # (n,2)
            X_idx_sq = X_sq[idx]           # (n,)
            X_cand = X[cand]               # (m,2)
            cand_sq = (X_cand * X_cand).sum(axis=1)  # (m,)

            # dist2(m,n) = ||cand||^2 + ||x||^2 - 2 cand·x
            d2_cn = cand_sq[:, None] + X_idx_sq[None, :] - 2.0 * (X_cand @ X_idx.T)
            np.maximum(d2_cn, 0.0, out=d2_cn)
            # objective: sum sqrt(dist2) (L2)
            obj = np.sqrt(d2_cn).sum(axis=1)
            best = int(cand[int(np.argmin(obj))])
            new_med[k] = best

        med = new_med
        med_pts = X[med]

    return X[med]


def select_points_facility_location_gpu(
    points_pool: np.ndarray,
    K: int,
    sigma: Optional[float] = None,
    cand_pool: int = 1024,
    seed: int = 0,
    cand_eval_bs: int = 64,
    device: str = "cuda:0",
) -> np.ndarray:
    """GPU-accelerated greedy facility location on point pool.

    Key optimizations:
      - Batch compute similarity on GPU
      - Vectorized gain computation

    Args:
        points_pool: (C, 2) candidate points
        K: number of pivots to select
        sigma: Gaussian kernel bandwidth (auto if None)
        cand_pool: candidate pool size per iteration
        seed: random seed
        cand_eval_bs: batch size for candidate evaluation
        device: "cuda:0" / "mps" / "cpu"

    Returns:
        pivots: (K, 2) selected pivot points
    """
    import torch

    # Device resolution
    dev = function.resolve_torch_device(device)

    rng = np.random.default_rng(seed)
    X = np.asarray(points_pool, dtype=np.float32)
    C = X.shape[0]
    K = min(K, C)

    # Heuristic sigma: median distance between random pairs
    if sigma is None:
        m = min(2000, C)
        idx1 = rng.choice(C, size=m, replace=False)
        idx2 = rng.choice(C, size=m, replace=False)
        d2 = ((X[idx1] - X[idx2]) ** 2).sum(axis=1)
        sigma = float(np.sqrt(np.median(d2) + 1e-12))
        if sigma <= 1e-6:
            sigma = 1.0

    inv2sig2 = 1.0 / (2.0 * sigma * sigma)

    # Transfer to GPU
    X_t = torch.as_tensor(X, dtype=torch.float32, device=dev)  # (C, 2)
    X_sq_t = (X_t * X_t).sum(dim=1)  # (C,)

    selected = []
    remaining = np.arange(C, dtype=np.int64)

    # Maintain best similarity for each point to current selected set
    best_sim_t = torch.zeros(C, dtype=torch.float32, device=dev)

    def _compute_sim_batch(piv_t: torch.Tensor) -> torch.Tensor:
        """Compute similarity from all points to batch of pivots.
        piv_t: (B, 2) -> sim: (B, C)
        """
        piv_sq = (piv_t * piv_t).sum(dim=1)  # (B,)
        # d2(i, j) = ||X[i]||^2 + ||piv[j]||^2 - 2 X[i] · piv[j]
        d2 = X_sq_t.unsqueeze(0) + piv_sq.unsqueeze(1) - 2.0 * (piv_t @ X_t.T)  # (B, C)
        d2 = torch.clamp(d2, min=0.0)
        return torch.exp(-d2 * inv2sig2)

    for _ in range(K):
        # Candidate subset for speed
        if remaining.shape[0] > cand_pool:
            cand = rng.choice(remaining, size=cand_pool, replace=False)
        else:
            cand = remaining

        cur_sum = best_sim_t.sum()

        best_gain = None
        best_idx = None

        cand_t = torch.as_tensor(cand, dtype=torch.long, device=dev)
        piv_all_t = X_t.index_select(0, cand_t)  # (Ncand, 2)

        Ncand = piv_all_t.shape[0]
        bs = int(cand_eval_bs)

        for s in range(0, Ncand, bs):
            e = min(Ncand, s + bs)
            piv_t = piv_all_t[s:e]  # (B, 2)

            sim = _compute_sim_batch(piv_t)  # (B, C)
            new_best = torch.maximum(best_sim_t.unsqueeze(0), sim)  # (B, C)
            new_sums = new_best.sum(dim=1)  # (B,)
            gains = new_sums - cur_sum  # (B,)

            gmax, arg = torch.max(gains, dim=0)
            gmax_val = float(gmax.item())
            if (best_gain is None) or (gmax_val > best_gain):
                best_gain = gmax_val
                best_idx = int(cand[s + int(arg.item())])

        # Commit
        selected.append(best_idx)
        piv_best = X_t[best_idx].unsqueeze(0)  # (1, 2)
        sim_best = _compute_sim_batch(piv_best)[0]  # (C,)
        best_sim_t = torch.maximum(best_sim_t, sim_best)

        remaining = remaining[remaining != best_idx]

    return X[np.array(selected, dtype=np.int64)]


def select_points_kmedoids_gpu(
    points_pool: np.ndarray,
    K: int,
    max_iter: int = 20,
    seed: int = 0,
    update_sample: int = 2000,
    device: str = "cuda:0",
) -> np.ndarray:
    """GPU-accelerated k-medoids on points_pool.

    Key optimizations:
      - Distance computations on GPU
      - Vectorized medoid updates

    Args:
        points_pool: (C, 2) candidate points
        K: number of clusters/medoids
        max_iter: maximum iterations
        seed: random seed
        update_sample: sample cap per cluster for medoid update
        device: "cuda:0" / "mps" / "cpu"

    Returns:
        pivots: (K, 2) selected medoid points
    """
    import torch

    # Device resolution
    dev = function.resolve_torch_device(device)

    rng = np.random.default_rng(seed)
    X = np.asarray(points_pool, dtype=np.float32)
    C = X.shape[0]
    K = min(K, C)

    # Transfer to GPU
    X_t = torch.as_tensor(X, dtype=torch.float32, device=dev)  # (C, 2)
    X_sq_t = (X_t * X_t).sum(dim=1)  # (C,)

    def dist2_to_points(pts_t: torch.Tensor) -> torch.Tensor:
        """Compute squared distances from all X to pts.
        pts_t: (M, 2) -> (C, M)
        """
        pts_sq = (pts_t * pts_t).sum(dim=1)  # (M,)
        d2 = X_sq_t.unsqueeze(1) + pts_sq.unsqueeze(0) - 2.0 * (X_t @ pts_t.T)
        return torch.clamp(d2, min=0.0)

    # k-medoids++ initialization on GPU
    med_idx = [int(rng.integers(0, C))]
    min_d2_t = dist2_to_points(X_t[med_idx[0]].unsqueeze(0))[:, 0]

    for _ in range(1, K):
        # Sample next proportional to min_d2
        probs = min_d2_t / (min_d2_t.sum() + 1e-12)
        probs_np = probs.cpu().numpy()
        nxt = int(rng.choice(C, p=probs_np))
        med_idx.append(nxt)
        d2_nxt = dist2_to_points(X_t[nxt].unsqueeze(0))[:, 0]
        min_d2_t = torch.minimum(min_d2_t, d2_nxt)

    med_idx = np.array(med_idx, dtype=np.int64)
    med_t = torch.as_tensor(med_idx, dtype=torch.long, device=dev)
    med_pts_t = X_t[med_t]  # (K, 2)

    prev_assign = None

    for _ in range(max_iter):
        d2 = dist2_to_points(med_pts_t)  # (C, K)
        assign = torch.argmin(d2, dim=1)  # (C,)
        assign_np = assign.cpu().numpy()

        if prev_assign is not None and np.array_equal(assign_np, prev_assign):
            break
        prev_assign = assign_np

        # Update medoid of each cluster
        new_med_idx = med_idx.copy()
        for k in range(K):
            mask = (assign == k)
            idx_k = torch.where(mask)[0]

            if idx_k.shape[0] == 0:
                # Empty cluster: re-seed to farthest point
                min_d2_all = torch.min(d2, dim=1).values
                far = int(torch.argmax(min_d2_all).item())
                new_med_idx[k] = far
                continue

            idx_k_np = idx_k.cpu().numpy()

            # Sample candidates within cluster to speed up
            if idx_k_np.size > update_sample:
                cand_np = rng.choice(idx_k_np, size=update_sample, replace=False)
            else:
                cand_np = idx_k_np

            # Choose point minimizing sum of distances to others in cluster
            cand_t = torch.as_tensor(cand_np, dtype=torch.long, device=dev)
            X_cand_t = X_t[cand_t]  # (m, 2)
            X_idx_t = X_t[idx_k]    # (n, 2)

            cand_sq = (X_cand_t * X_cand_t).sum(dim=1)  # (m,)
            idx_sq = (X_idx_t * X_idx_t).sum(dim=1)     # (n,)

            # d2(m, n) = ||cand||^2 + ||x||^2 - 2 cand·x
            d2_cn = cand_sq.unsqueeze(1) + idx_sq.unsqueeze(0) - 2.0 * (X_cand_t @ X_idx_t.T)
            d2_cn = torch.clamp(d2_cn, min=0.0)

            # Objective: sum sqrt(dist2) (L2)
            obj = torch.sqrt(d2_cn).sum(dim=1)  # (m,)
            best_local = int(torch.argmin(obj).item())
            new_med_idx[k] = int(cand_np[best_local])

        med_idx = new_med_idx
        med_t = torch.as_tensor(med_idx, dtype=torch.long, device=dev)
        med_pts_t = X_t[med_t]

    return X[med_idx]


# -------------------------
# Trajectory-index based selection on D (for ablations)
# -------------------------
def select_traj_random(D: Array, K: int, seed: int = 0) -> Array:
    N = D.shape[0]
    rng = np.random.default_rng(seed)
    K = min(K, N)
    return rng.choice(N, size=K, replace=False).astype(np.int64)

def select_traj_fps(D: Array, K: int, seed: int = 0) -> Array:
    """FPS on trajectories with pairwise distance matrix D."""
    N = D.shape[0]
    rng = np.random.default_rng(seed)
    K = min(K, N)
    p0 = int(rng.integers(0, N))
    piv = [p0]
    min_dist = D[:, p0].copy()
    for _ in range(1, K):
        p = int(np.argmax(min_dist))
        piv.append(p)
        min_dist = np.minimum(min_dist, D[:, p])
    return np.array(piv, dtype=np.int64)

def _sample_pairs(N: int, num_pairs: int, seed: int = 0) -> Tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, N, size=num_pairs, endpoint=False)
    b = rng.integers(0, N, size=num_pairs, endpoint=False)
    mask = a != b
    a, b = a[mask], b[mask]
    if a.shape[0] < num_pairs:
        extra = num_pairs - a.shape[0]
        a2, b2 = _sample_pairs(N, extra, seed + 1)
        a = np.concatenate([a, a2])
        b = np.concatenate([b, b2])
    return a[:num_pairs], b[:num_pairs]

def select_traj_tightness_greedy(
    D: Array,
    K: int,
    cand_pool: int = 512,
    num_pairs: int = 20000,
    seed: int = 0,
    eps: float = 1e-12,
) -> Array:
    """Greedy selection maximizing mean (LB/D) on sampled pairs.

    LB_P(i,j) = max_{p in P} |D[i,p]-D[j,p]|.
    """
    N = D.shape[0]
    rng = np.random.default_rng(seed)
    K = min(K, N)

    a, b = _sample_pairs(N, num_pairs=num_pairs, seed=seed)
    dab = D[a, b] + eps
    lb = np.zeros_like(dab, dtype=np.float64)

    remaining = np.arange(N, dtype=np.int64)
    pivots: List[int] = []

    for _ in range(K):
        if remaining.shape[0] > cand_pool:
            cand = rng.choice(remaining, size=cand_pool, replace=False)
        else:
            cand = remaining

        cur_obj = float((lb / dab).mean())
        best_p, best_gain = None, -1e18

        # candidate loop (cand_pool is small)
        for p in cand:
            diff = np.abs(D[a, p] - D[b, p])
            new_lb = np.maximum(lb, diff)
            obj = float((new_lb / dab).mean())
            gain = obj - cur_obj
            if gain > best_gain:
                best_gain = gain
                best_p = int(p)

        pivots.append(best_p)
        lb = np.maximum(lb, np.abs(D[a, best_p] - D[b, best_p]))
        remaining = remaining[remaining != best_p]

    return np.array(pivots, dtype=np.int64)

def extract_points_from_traj_indices(
    traj_list: List,
    traj_idx: Array,
    rule: Literal["mid", "start", "end", "max_dev"] = "mid",
) -> Array:
    """Map selected trajectory indices to pivot points via a deterministic rule."""
    out = np.empty((len(traj_idx), 2), dtype=np.float32)
    for k, tid in enumerate(traj_idx):
        t = traj_to_np(traj_list[int(tid)])
        L = t.shape[0]
        if L == 0:
            raise ValueError("Empty trajectory encountered.")
        if rule == "start":
            out[k] = t[0]
        elif rule == "end":
            out[k] = t[-1]
        elif rule == "mid":
            out[k] = t[L // 2]
        elif rule == "max_dev":
            s = t[0]; e = t[-1]
            se = e - s
            denom = float(np.dot(se, se)) + 1e-12
            v = t - s
            proj = (v @ se) / denom
            foot = s + proj[:, None] * se[None, :]
            dev2 = ((t - foot) ** 2).sum(axis=1)
            out[k] = t[int(np.argmax(dev2))]
        else:
            raise ValueError(f"Unknown rule: {rule}")
    return out

# -------------------------
# Unified entry
# -------------------------
@dataclass
class PivotSelectConfig:
    strategy: str = "point-fps"  # point-random|point-fps|point-hard|point-ndcg|traj-random|traj-fps|traj-tightness
    K: int = 32
    seed: int = 0

    # point-pool params (point-*)
    pool_per_traj: int = 16
    pool_include_start_end: bool = True

    # point-hard / point-ndcg params (require D_train)
    select_traj_subsample: int = 0  # subsample points per traj when computing proxy phi
    hard_anchors: int = 512
    hard_k: int = 50
    hard_pairs_per_anchor: int = 5
    hard_cand_pool: int = 512

    ndcg_anchors: int = 256
    ndcg_eval_k: int = 50
    ndcg_at: int = 10
    ndcg_neg: int = 50
    ndcg_cand_pool: int = 256

    # traj-based params (traj-*)
    traj_point_rule: str = "mid"
    tight_cand_pool: int = 512
    tight_num_pairs: int = 20000
    device: str = "cuda:0"  # for future use
def select_pivot_points(
    traj_list: List,
    cfg: PivotSelectConfig,
    D_train: Optional[Array] = None,
) -> Array:
    """Select K pivot points according to cfg.strategy.

    Args:
        traj_list: list of trajectories (training set recommended)
        cfg: selection config
        D_train: optional N×N distance matrix for traj-based strategies

    Returns:
        pivots: (K,2) float32
    """
    strat = cfg.strategy.lower()
    if strat.startswith("point-"):
        # Paper-to-code mapping for point-based pivot selection:
        # - point-random    -> Random
        # - point-fps       -> FPS (Farthest Point Sampling)
        # - point-facility  -> Facility
        # - point-kmedoids  -> KMedoids
        # - point-tightness -> Tightness
        # - point-hard      -> Hardness
        pool = build_point_pool(
            traj_list,
            per_traj=int(cfg.pool_per_traj),
            seed=int(cfg.seed),
            include_start_end=bool(cfg.pool_include_start_end),
        )
        if strat == "point-random":
            res = select_points_random(pool, cfg.K, seed=cfg.seed)
            
            return res
        if strat == "point-fps":
            res = select_points_fps_gpu(pool, cfg.K, seed=cfg.seed, device=cfg.device)
            return res
        if strat == "point-hard":
            if D_train is None:
                raise ValueError("point-hard requires D_train (train distance matrix).")

            # NOTE: this is the current main Hardness implementation used by
            # LB-TrajRep. The function name keeps the original "mac" history,
            # but the logic is now device-resolved internally and can run on
            # CUDA / MPS / CPU through resolve_torch_device().
            res1 = select_points_hard_greedy_fast_mac(
                traj_list=traj_list,
                D_train=D_train,
                points_pool=pool,
                K=cfg.K,
                seed=cfg.seed,
                pool_cand=int(cfg.hard_cand_pool),
                anchors=int(cfg.hard_anchors),
                hard_k=int(cfg.hard_k),
                pairs_per_anchor=int(cfg.hard_pairs_per_anchor),
                traj_subsample=int(cfg.select_traj_subsample),
                cand_eval_bs=128,
            )
            return res1

        if strat == "point-ndcg":
            if D_train is None:
                raise ValueError("point-ndcg requires D_train (train distance matrix).")
            res = select_points_ndcg_greedy_gpu(
                traj_list=traj_list,
                D_train=D_train,
                points_pool=pool,
                K=cfg.K,
                seed=cfg.seed,
                pool_cand=int(cfg.ndcg_cand_pool),
                anchors=int(cfg.ndcg_anchors),
                eval_k=int(cfg.ndcg_eval_k),
                ndcg_at=int(cfg.ndcg_at),
                neg_num=int(cfg.ndcg_neg),
                traj_subsample=int(cfg.select_traj_subsample),
                device=cfg.device,
            )
            
            return res
        if strat == "point-tightness":
            res = select_points_tightness_greedy_gpu(
                traj_list=traj_list,
                D_train=D_train,
                points_pool=pool,
                K=cfg.K,
                cand_pool=int(cfg.tight_cand_pool),
                num_pairs=int(cfg.tight_num_pairs),
                traj_point_subsample=int(cfg.select_traj_subsample),
                seed=cfg.seed,
                device=cfg.device
            )
            
            return res
            
        if strat == "point-facility":
            res = select_points_facility_location_gpu(
                points_pool=pool,
                K=cfg.K,
                sigma=None,
                cand_pool=1024,
                seed=cfg.seed,
                device=cfg.device
            )
            return res
        if strat == "point-kmedoids":
            res = select_points_kmedoids_gpu(
                points_pool=pool,
                K=cfg.K,
                max_iter=20,
                seed=cfg.seed,
                update_sample=2000,
                device=cfg.device
            ) 
            
            return res
        raise ValueError(f"Unknown point strategy: {cfg.strategy}")

    # traj-based
    if D_train is None:
        raise ValueError(f"Strategy {cfg.strategy} requires D_train (train distance matrix).")

    if strat == "traj-random":
        idx = select_traj_random(D_train, cfg.K, seed=cfg.seed)
    elif strat == "traj-fps":
        idx = select_traj_fps(D_train, cfg.K, seed=cfg.seed)
    elif strat == "traj-tightness":
        idx = select_traj_tightness_greedy(
            D_train, cfg.K, seed=cfg.seed,
            cand_pool=int(cfg.tight_cand_pool),
            num_pairs=int(cfg.tight_num_pairs),
        )
    else:
        raise ValueError(f"Unknown traj strategy: {cfg.strategy}")

    return extract_points_from_traj_indices(traj_list, idx, rule=cfg.traj_point_rule)
