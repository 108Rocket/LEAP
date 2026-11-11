"""Geometry normalization and compatible imports for Level 3 (RGR)."""
import numpy as np
import numba as nb
from leap.common.raw_geometry_refinement import directed_hausdorff_squared as direction, candidate_distance_kernel as pair_kernel, hausdorff_candidate_distances as fused_raw_scores


@nb.njit(parallel=True,cache=True)
def normalize_points(raw,mean,std):
    # Callers preserve float64 absolute coordinates until subtraction/scaling.
    out=np.empty(raw.shape,np.float32)
    for i in nb.prange(len(raw)):
        for d in range(2):
            out[i,d]=(np.float64(raw[i,d])-mean[d])/std[d]
    return out


@nb.njit(cache=True)
def normalize_points_serial(raw,mean,std):
    """Avoid a parallel thread-pool barrier for a single incoming trajectory."""
    out=np.empty(raw.shape,np.float32)
    for i in range(len(raw)):
        for d in range(2):
            out[i,d]=(np.float64(raw[i,d])-mean[d])/std[d]
    return out


@nb.njit(cache=True)
def normalize_points_into(raw,mean,std,out):
    """Same normalization, writing into a session-owned pinned host view."""
    for i in range(len(raw)):
        for d in range(2):
            out[i,d]=(np.float64(raw[i,d])-mean[d])/std[d]
