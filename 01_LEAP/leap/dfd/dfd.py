"""RGR for DFD: original ordered vertices, eq:dfd-target, no resampling.

Eiter--Mannila: C(i,j)=max(d(i,j), min(C(i-1,j),C(i-1,j-1),C(i,j-1))).
Squared distances commute with min/max; take one square root at the end.
Three rolling anti-diagonal buffers are retained per pair. This is FP32, not a strict
FP64-reference reviewer or the point-to-segment Hausdorff early-reject path.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def dfd_kernel(P, OFF, T, TOFF, Q, D, OUT, WORK, BLOCK: tl.constexpr):
    pair = tl.program_id(0)
    q = tl.load(Q + pair)
    d = tl.load(D + pair)
    a0 = tl.load(OFF + q)
    na = tl.load(OFF + q + 1) - a0
    b0 = tl.load(TOFF + d)
    nb = tl.load(TOFF + d + 1) - b0
    j = tl.arange(0, BLOCK)
    bx = tl.load(T + (b0 + j) * 2, j < nb, other=0.)
    by = tl.load(T + (b0 + j) * 2 + 1, j < nb, other=0.)
    work = WORK + pair * 3 * BLOCK
    for bank in range(3):
        tl.store(work + bank * BLOCK + j, float('inf'))
    tl.debug_barrier()
    for diagonal in range(na + nb - 1):
        i = diagonal - j
        valid = (i >= 0) & (i < na) & (j < nb)
        ax = tl.load(P + (a0 + i) * 2, valid, other=0.)
        ay = tl.load(P + (a0 + i) * 2 + 1, valid, other=0.)
        previous_bank = (diagonal + 2) % 3
        previous2_bank = (diagonal + 1) % 3
        previous = tl.load(work + previous_bank * BLOCK + j, volatile=True)
        left = tl.load(work + previous_bank * BLOCK + j - 1,
                       j > 0, other=float('inf'), volatile=True)
        cross = tl.load(work + previous2_bank * BLOCK + j - 1,
                        j > 0, other=float('inf'), volatile=True)
        predecessor = tl.minimum(previous, tl.minimum(left, cross))
        predecessor = tl.where((i == 0) & (j == 0), 0., predecessor)
        cost = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
        current = tl.where(valid, tl.maximum(cost, predecessor), float('inf'))
        tl.store(work + (diagonal % 3) * BLOCK + j, current)
        tl.debug_barrier()
    value = tl.load(work + ((na + nb - 2) % 3) * BLOCK + nb - 1, volatile=True)
    tl.store(OUT + pair, tl.sqrt(value))


@torch.no_grad()
def dfd_pairs(queries, database, qids, dids):
    from leap.common.runtime.context import current
    plan = current()
    if plan is not None:
        return plan.pair_values(queries, database, qids, dids, 0)
    if qids.shape != dids.shape or qids.ndim != 1:
        raise ValueError('Expected one-dimensional matching pair ID arrays')
    if qids.dtype != torch.int64 or dids.dtype != torch.int64:
        raise ValueError('Pair IDs must be int64')
    if any(t.device != queries.points.device for t in (database.points, qids, dids)):
        raise ValueError('Pairs and geometry must share a CUDA device')
    out = torch.empty(dids.shape, device=dids.device, dtype=queries.points.dtype)
    if not dids.numel():
        return out
    maximum = int((database.offsets[1:] - database.offsets[:-1]).max())
    block = triton.next_power_of_2(maximum)
    # Bound temporary rolling buffers; chunk pairs, never truncate source paths.
    chunk = max(1, min(dids.numel(), (32 * 1024 * 1024) // (3 * block)))
    work = torch.empty((chunk, 3, block), device=dids.device, dtype=queries.points.dtype)
    qids, dids = qids.contiguous(), dids.contiguous()
    for start in range(0, dids.numel(), chunk):
        stop = min(start + chunk, dids.numel())
        dfd_kernel[(stop - start,)](queries.points, queries.offsets,
            database.points, database.offsets, qids[start:stop], dids[start:stop],
            out[start:stop], work, block, num_warps=4, enable_fp_fusion=False)
    return out


@torch.no_grad()
def dfd_pairs_bucketed(queries, database, qids, dids):
    """Same FP32 recurrence, with rolling-buffer width local to a length bucket."""
    if qids.shape != dids.shape or qids.ndim != 1 or qids.dtype != torch.int64 or dids.dtype != torch.int64:
        raise ValueError('Expected matching flat int64 pair IDs')
    out = torch.empty(dids.shape, device=dids.device, dtype=queries.points.dtype)
    if not dids.numel():
        return out
    lengths = database.offsets[dids+1] - database.offsets[dids]
    maximum = triton.next_power_of_2(int(lengths.max()))
    lower = 0
    for block in (2**i for i in range(maximum.bit_length())):
        rows = ((lengths > lower) & (lengths <= block)).nonzero().flatten()
        lower = block
        if not len(rows):
            continue
        qs, ds = qids[rows].contiguous(), dids[rows].contiguous()
        values = torch.empty(len(rows), device=dids.device, dtype=queries.points.dtype)
        chunk = max(1, min(len(rows), (32 * 1024 * 1024) // (3 * block)))
        work = torch.empty((chunk, 3, block), device=dids.device, dtype=queries.points.dtype)
        for lo in range(0, len(rows), chunk):
            hi = min(lo + chunk, len(rows))
            dfd_kernel[(hi-lo,)](queries.points, queries.offsets, database.points, database.offsets,
                qs[lo:hi], ds[lo:hi], values[lo:hi], work, block, num_warps=4, enable_fp_fusion=False)
        out[rows] = values
    return out


@triton.jit
def bound_kernel(P, OFF, T, TOFF, ZQ, ZD, Q, D, OUT):
    pair = tl.program_id(0)
    q = tl.load(Q + pair)
    d = tl.load(D + pair)
    j = tl.arange(0, 64)
    # Vertex response is a lower bound on point-set Hausdorff, hence on DFD.
    qa = tl.load(ZQ + q * 128 + j)
    da = tl.load(ZD + d * 128 + j)
    bound = tl.max(tl.abs(qa - da), axis=0)
    q0 = tl.load(OFF + q)
    q1 = tl.load(OFF + q + 1) - 1
    d0 = tl.load(TOFF + d)
    d1 = tl.load(TOFF + d + 1) - 1
    for end in range(2):
        qi = tl.where(end == 0, q0, q1)
        di = tl.where(end == 0, d0, d1)
        dx = tl.load(P + 2 * qi) - tl.load(T + 2 * di)
        dy = tl.load(P + 2 * qi + 1) - tl.load(T + 2 * di + 1)
        # Match tree and dense candidate scores before score/ID tie breaking.
        bound = tl.maximum(bound, libdevice.sqrt_rn(dx * dx + dy * dy))
    tl.store(OUT + pair, bound)


@torch.no_grad()
def dfd_pair_bounds(queries, database, qids, dids):
    out = torch.empty(dids.shape, device=dids.device, dtype=queries.points.dtype)
    if dids.numel():
        bound_kernel[(dids.numel(),)](queries.points, queries.offsets,
            database.points, database.offsets, queries.coordinates, database.coordinates,
            qids.contiguous(), dids.contiguous(), out, num_warps=4, enable_fp_fusion=False)
    return out
