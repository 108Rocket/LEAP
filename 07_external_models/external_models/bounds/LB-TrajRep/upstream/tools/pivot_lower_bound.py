import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass
class PivotConfig:
    """Configuration for pivot selection."""
    num_pivots: int = 128
    sample_method: Literal["random", "fps_start_end"] = "random"
    random_seed: int = 666


def _traj_start_end_xy(traj) -> np.ndarray:
    """Return [lon_s, lat_s, lon_e, lat_e] from a trajectory.

    Assumes traj is a list/array of points, each point is (lon, lat) or [lon, lat].
    """
    s = traj[0]
    e = traj[-1]
    return np.array([s[0], s[1], e[0], e[1]], dtype=np.float32)


def select_pivot_indices(base_traj_list, cfg: PivotConfig) -> np.ndarray:
    """Select pivot indices within the *base* trajectory list.

    Returns:
        pivot_idx: (M,) int64 indices into base_traj_list

    Methods:
      - random: uniform random without replacement
      - fps_start_end: farthest-point sampling on 4D start/end vector
    """
    n = len(base_traj_list)
    m = int(min(cfg.num_pivots, n))
    rng = np.random.default_rng(cfg.random_seed)

    if m <= 0:
        return np.zeros((0,), dtype=np.int64)

    method = (cfg.sample_method or "random").lower()
    if method == "random":
        return rng.choice(n, size=m, replace=False).astype(np.int64)

    if method != "fps_start_end":
        raise ValueError(f"Unknown pivot sample_method: {cfg.sample_method}")

    # FPS in 4D start/end space
    feats = np.stack([_traj_start_end_xy(t) for t in base_traj_list], axis=0)  # (n,4)

    # pick first pivot randomly
    pivots = np.empty((m,), dtype=np.int64)
    pivots[0] = int(rng.integers(0, n))

    # maintain min squared distance to selected pivots
    diff = feats - feats[pivots[0]][None, :]
    min_d2 = (diff * diff).sum(axis=1)

    for k in range(1, m):
        nxt = int(np.argmax(min_d2))
        pivots[k] = nxt
        diff = feats - feats[nxt][None, :]
        d2 = (diff * diff).sum(axis=1)
        min_d2 = np.minimum(min_d2, d2)

    return pivots


def slice_pivot_distance_tables(
    test_qb_dist: np.ndarray,
    base_bb_dist: np.ndarray,
    pivot_base_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice distance tables needed for pivot LB.

    Args:
        test_qb_dist: (Q, B) true distance between query and base
        base_bb_dist: (B, B) true distance between base and base
        pivot_base_indices: (M,) indices of pivots within base

    Returns:
        q_to_piv: (Q, M) distances from queries to pivots
        b_to_piv: (B, M) distances from bases to pivots

    Notes:
        This implements d(q, p_i) and d(b, p_i).
        Requires that pivots are selected from base set.
    """
    pivot_base_indices = np.asarray(pivot_base_indices, dtype=np.int64)
    if test_qb_dist.ndim != 2:
        raise ValueError("test_qb_dist must be 2D (Q,B)")
    if base_bb_dist.ndim != 2:
        raise ValueError("base_bb_dist must be 2D (B,B)")

    Q, B = test_qb_dist.shape
    Bb0, Bb1 = base_bb_dist.shape
    if Bb0 != B or Bb1 != B:
        raise ValueError(f"base_bb_dist shape {base_bb_dist.shape} must match B={B}")

    # Query-to-pivot: slice columns from test_qb_dist
    q_to_piv = test_qb_dist[:, pivot_base_indices]

    # Base-to-pivot: slice columns from base_bb_dist
    b_to_piv = base_bb_dist[:, pivot_base_indices]

    return q_to_piv.astype(np.float32), b_to_piv.astype(np.float32)


def compute_pivot_lb_from_pivot_dists(
    q_to_piv: np.ndarray,
    b_to_piv: np.ndarray,
    chunk_q: int = 128,
) -> np.ndarray:
    """Compute Pivot Lower Bound matrix.

    LB(q,c) = max_i |d(q,p_i) - d(c,p_i)|

    Args:
        q_to_piv: (Q,M)
        b_to_piv: (B,M)
        chunk_q: compute in query chunks to reduce peak memory

    Returns:
        lb: (Q,B) float32
    """
    q_to_piv = np.asarray(q_to_piv, dtype=np.float32)
    b_to_piv = np.asarray(b_to_piv, dtype=np.float32)

    Q, M = q_to_piv.shape
    B, M2 = b_to_piv.shape
    if M != M2:
        raise ValueError(f"pivot dim mismatch: q_to_piv M={M}, b_to_piv M={M2}")

    lb = np.empty((Q, B), dtype=np.float32)

    for qs in range(0, Q, chunk_q):
        qe = min(qs + chunk_q, Q)
        qblk = q_to_piv[qs:qe]  # (bq,M)

        # compute max abs diff along pivot dimension
        # diff: (bq,B,M) -> reduce over M
        # We do it via broadcasting: (bq,1,M) - (1,B,M)
        diff = np.abs(qblk[:, None, :] - b_to_piv[None, :, :])
        lb[qs:qe] = diff.max(axis=2)

    return lb
