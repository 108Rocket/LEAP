"""Reuse exact seed distances during Hausdorff raw-geometry refinement."""
import torch
import triton
import triton.language as tl

from leap.common.early_refinement import bounded_direction
from leap.common.raw_geometry_refinement import directed_hausdorff_squared, point_direction


@triton.jit
def seed_position(SEEDS, q, d, S: tl.constexpr):
    lane = tl.arange(0, triton.next_power_of_2(S))
    ids = tl.load(SEEDS + q * S + lane, lane < S, other=-1)
    return tl.min(tl.where((lane < S) & (ids == d), lane, S), 0)


@triton.jit
def cached_kernel(P, OFF, T, TOFF, Q, D, SEEDS, RAW, RADIUS, OUT,
                  S: tl.constexpr, POINT: tl.constexpr, BOUNDED: tl.constexpr):
    pair = tl.program_id(0)
    q = tl.load(Q + pair)
    d = tl.load(D + pair)
    at = seed_position(SEEDS, q, d, S)
    if at < S:
        value = tl.load(RAW + q * S + at)
    else:
        if POINT:
            ab = point_direction(P, OFF, q, T, TOFF, d)
            ba = point_direction(T, TOFF, d, P, OFF, q)
            value = tl.sqrt(tl.maximum(ab, ba))
        elif BOUNDED:
            radius = tl.load(RADIUS + q)
            ab, rejected, _ = bounded_direction(P, OFF, q, T, TOFF, d, radius, 32, False)
            ba = tl.full((), 0, P.dtype.element_ty)
            if not rejected:
                ba, rejected, _ = bounded_direction(T, TOFF, d, P, OFF, q, radius, 32, False)
            value = tl.where(rejected, float('inf'), tl.sqrt(tl.maximum(ab, ba)))
        else:
            ab = directed_hausdorff_squared(P, OFF, q, T, TOFF, d)
            ba = directed_hausdorff_squared(T, TOFF, d, P, OFF, q)
            value = tl.sqrt(tl.maximum(ab, ba))
    tl.store(OUT + pair, value)


@torch.no_grad()
def cached_pairs(queries, database, qids, dids, seeds, raw, radii=None, *,
                 point=False, bounded=True):
    """The cache is per search call and contains original distances, not bounds."""
    device = queries.points.device
    if seeds.ndim != 2 or seeds.shape[0] != len(queries.coordinates) or seeds.shape[1] < 1:
        raise ValueError('Expected a nonempty seed matrix with one row per query')
    if raw.shape != seeds.shape or raw.dtype != queries.points.dtype:
        raise ValueError('Cached distances must match the seed matrix and geometry dtype')
    if qids.ndim != 1 or qids.shape != dids.shape:
        raise ValueError('Expected matching one-dimensional pair IDs')
    if any(t.dtype != torch.int64 for t in (qids, dids, seeds)):
        raise ValueError('Pair and seed IDs must be int64')
    tensors = (database.points, qids, dids, seeds, raw)
    if device.type != 'cuda' or any(t.device != device for t in tensors):
        raise ValueError('Geometry, IDs and cached distances must share a CUDA device')
    if bounded and not point:
        if (radii is None or radii.shape != (len(queries.coordinates),)
                or radii.dtype != queries.points.dtype or radii.device != device):
            raise ValueError('Expected one matching radius per query')
    out = torch.empty(dids.shape, device=device, dtype=queries.points.dtype)
    if dids.numel():
        cached_kernel[(dids.numel(),)](
            queries.points, queries.offsets, database.points, database.offsets,
            qids.contiguous(), dids.contiguous(), seeds.contiguous(), raw.contiguous(),
            out if radii is None else radii.contiguous(), out, seeds.shape[1], point, bounded,
            num_warps=4, enable_fp_fusion=False)
    return out
