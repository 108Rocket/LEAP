"""Original-vertex DTW with optional exact frontier abandonment and bucketing.

The raw recurrence is eq:dtw-target; pair budgets derive from eq:search-radius.
Nonnegative local costs imply that every future path must cross one of the
last two antidiagonals. Reject only if BOTH frontiers exceed the query radius;
a diagonal DTW step can skip a single antidiagonal. Surviving scores retain
the original FP32 recurrence. FP32 batch refinement uses a block-maintained
frontier; unbounded initialization and native Single execution stay unchanged.
No vertices or candidate IDs are removed here.
"""
import torch
import triton
import triton.language as tl
from leap.dtw.dtw import dtw_kernel


@triton.jit
def bounded_kernel(P, OFF, T, TOFF, Q, D, OUT, WORK, LIMIT, STEPS,
                   BLOCK: tl.constexpr, CHECK: tl.constexpr, RECORD: tl.constexpr):
    pair = tl.program_id(0)
    q = tl.load(Q + pair)
    d = tl.load(D + pair)
    a0 = tl.load(OFF + q)
    na = tl.load(OFF + q + 1) - a0
    b0 = tl.load(TOFF + d)
    nb = tl.load(TOFF + d + 1) - b0
    limit = tl.load(LIMIT + pair)
    j = tl.arange(0, BLOCK)
    bx = tl.load(T + (b0 + j) * 2, j < nb, other=0.)
    by = tl.load(T + (b0 + j) * 2 + 1, j < nb, other=0.)
    work = WORK + pair * 3 * BLOCK
    for bank in range(3):
        tl.store(work + bank * BLOCK + j, float('inf'))
    tl.debug_barrier()
    diagonal = 0
    alive = True
    while (diagonal < na + nb - 1) & alive:
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
        current = tl.where(valid, tl.sqrt(cost) + predecessor, float('inf'))
        tl.store(work + (diagonal % 3) * BLOCK + j, current)
        tl.debug_barrier()
        if (diagonal + 1) % CHECK == 0:
            alive = tl.min(tl.minimum(previous, current), 0) <= limit
        diagonal += 1
    value = tl.load(work + ((na + nb - 2) % 3) * BLOCK + nb - 1, volatile=True)
    tl.store(OUT + pair, tl.where(alive, value, float('inf')))
    if RECORD:
        tl.store(STEPS + pair, diagonal)


def _use_batch_frontier(queries, limits, *, bucket, check, plan):
    from leap.common.runtime import options
    return (options.dtw_batch_frontier and plan is None and limits is not None
            and bucket and check == 8 and queries.points.dtype == torch.float32
            and len(queries.offsets) > 2 and torch.version.hip is None)


@torch.no_grad()
def pairs(queries, database, qids, dids, limits=None, *, bucket=False,
          check=8, record=False, chunk=16384, maximum=None):
    from leap.common.runtime.context import current
    plan = current()
    if plan is not None and not record:
        return plan.pair_values(queries, database, qids, dids,
                                1 if limits is None else 2, limits, check=check)
    if qids.shape != dids.shape or qids.ndim != 1:
        raise ValueError('Expected one-dimensional matching pair ID arrays')
    if qids.dtype != torch.int64 or dids.dtype != torch.int64:
        raise ValueError('Pair IDs must be int64')
    if any(t.device != queries.points.device for t in (database.points, qids, dids)):
        raise ValueError('Pairs and geometry must share a CUDA device')
    if limits is not None and (limits.shape != dids.shape or limits.dtype != queries.points.dtype
                               or limits.device != dids.device):
        raise ValueError('Limits must be matching FP32 pair budgets')
    if check < 1 or chunk < 1:
        raise ValueError('Check interval and chunk size must be positive')
    use_frontier = _use_batch_frontier(queries, limits, bucket=bucket, check=check, plan=plan)
    kernel = bounded_kernel
    if use_frontier:
        from leap.dtw.dtw_frontier import batch_frontier_kernel
        kernel = batch_frontier_kernel
    out = torch.empty(dids.shape, device=dids.device, dtype=queries.points.dtype)
    steps = torch.empty_like(dids, dtype=torch.int32) if record else None
    if not dids.numel():
        return (out, steps) if record else out
    if maximum is None:
        maximum = triton.next_power_of_2(int((database.offsets[1:] - database.offsets[:-1]).max()))
    lengths = database.offsets[dids+1] - database.offsets[dids] if bucket else None
    lower = 0
    for block in ([2**i for i in range(maximum.bit_length())] if bucket else [maximum]):
        selected = ((lengths > lower) & (lengths <= block)).nonzero().flatten() if bucket else None
        lower = block
        size = len(selected) if bucket else len(dids)
        if not size:
            continue
        qs = (qids[selected] if bucket else qids).contiguous()
        ds = (dids[selected] if bucket else dids).contiguous()
        budgets = (limits[selected] if bucket else limits).contiguous() if limits is not None else None
        values = torch.empty(size, device=dids.device, dtype=queries.points.dtype) if bucket else out
        counts = torch.empty(size, device=dids.device, dtype=torch.int32) if record and bucket else steps
        batch = min(chunk, size, (32 * 1024 * 1024) // (3 * block))
        work = torch.empty((batch, 3, block), device=dids.device, dtype=queries.points.dtype)
        for start in range(0, size, batch):
            end = min(start + batch, size)
            args = (queries.points, queries.offsets, database.points, database.offsets,
                    qs[start:end], ds[start:end], values[start:end], work)
            if budgets is None:
                dtw_kernel[(end-start,)](*args, block, num_warps=4, enable_fp_fusion=False)
                if record:
                    counts[start:end] = (queries.offsets[qs[start:end]+1]-queries.offsets[qs[start:end]]
                                        +database.offsets[ds[start:end]+1]-database.offsets[ds[start:end]]-1)
            else:
                kernel[(end-start,)](*args, budgets[start:end],
                    counts[start:end] if record else work, block, check, record,
                    num_warps=1 if use_frontier else 4, enable_fp_fusion=False)
        if bucket:
            out[selected] = values
            if record:
                steps[selected] = counts
    return (out, steps) if record else out
