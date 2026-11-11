"""Threshold-safe block frontiers for FP32 batch DTW refinement.

Maintain the hull of two surviving antidiagonals once per CHECK steps. Every
continuation can advance at most CHECK target columns, so the expanded hull
cannot skip a path whose final cost is within the supplied budget. Single
sessions continue to call the original bounded kernel.
"""
import triton
import triton.language as tl
from leap.dtw.dtw_bounded import bounded_kernel as _native_kernel


@triton.jit
def frontier_kernel(P, OFF, T, TOFF, Q, D, OUT, WORK, LIMIT, STEPS, COUNTERS,
             BLOCK: tl.constexpr, TILE: tl.constexpr, CHECK: tl.constexpr, RECORD: tl.constexpr,
             RECORD_WORK: tl.constexpr):
    pair = tl.program_id(0)
    q, d = tl.load(Q + pair), tl.load(D + pair)
    a0, b0 = tl.load(OFF + q), tl.load(TOFF + d)
    na = (tl.load(OFF + q + 1) - a0).to(tl.int32)
    nb = (tl.load(TOFF + d + 1) - b0).to(tl.int32)
    limit = tl.load(LIMIT + pair)
    work = WORK + pair * 3 * BLOCK
    lane = tl.arange(0, TILE)
    if BLOCK <= 64:
        target_x0 = tl.load(T + (b0 + lane) * 2, lane < nb, other=0.)
        target_y0 = tl.load(T + (b0 + lane) * 2 + 1, lane < nb, other=0.)
        target_x1 = tl.load(T + (b0 + lane + 32) * 2, lane + 32 < nb, other=0.)
        target_y1 = tl.load(T + (b0 + lane + 32) * 2 + 1, lane + 32 < nb, other=0.)
    prev_lo, prev_hi = nb, -1
    # Keep loop-carried history independent of the invariant target length.
    prev2_lo, prev2_hi = 0, -1
    diagonal = 0
    alive = True
    result = tl.full((), float('inf'), P.dtype.element_ty)
    evaluations = tl.full((), 0, tl.int64)
    issued = tl.full((), 0, tl.int64)
    full_lane = tl.arange(0, BLOCK)
    live_lo, live_hi = 0, 0
    while (diagonal < na + nb - 1) & alive:
        # Each DTW step advances at most one target column. This hull safely
        # contains every CHECK-step continuation of either surviving frontier.
        block_lo = live_lo
        block_hi = tl.minimum(nb - 1, live_hi + CHECK)
        for step in range(CHECK):
            if diagonal < na + nb - 1:
                lo = tl.maximum(block_lo, tl.maximum(0, diagonal - na + 1))
                hi = tl.minimum(block_hi, tl.minimum(nb - 1, diagonal))
                for start in range(lo, hi + 1, TILE):
                    j = start + lane
                    i = diagonal - j
                    valid = (j <= hi) & (i >= 0) & (i < na)
                    pb, p2b, cb = (diagonal + 2) % 3, (diagonal + 1) % 3, diagonal % 3
                    up = tl.load(work + pb * BLOCK + j,
                        valid & (j >= prev_lo) & (j <= prev_hi), other=float('inf'), volatile=True)
                    left = tl.load(work + pb * BLOCK + j - 1,
                        valid & (j > 0) & (j - 1 >= prev_lo) & (j - 1 <= prev_hi),
                        other=float('inf'), volatile=True)
                    cross = tl.load(work + p2b * BLOCK + j - 1,
                        valid & (j > 0) & (j - 1 >= prev2_lo) & (j - 1 <= prev2_hi),
                        other=float('inf'), volatile=True)
                    predecessor = tl.minimum(up, tl.minimum(left, cross))
                    predecessor = tl.where((i == 0) & (j == 0), 0., predecessor)
                    reachable = valid & (predecessor <= limit) & (predecessor < float('inf'))
                    ax = tl.load(P + (a0 + i) * 2, reachable, other=0.)
                    ay = tl.load(P + (a0 + i) * 2 + 1, reachable, other=0.)
                    if BLOCK <= 64:
                        index = j % 32
                        bx0 = tl.inline_asm_elementwise('shfl.sync.idx.b32 $0, $1, $2, 31, -1;',
                            constraints='=f,f,r', args=[target_x0,index], dtype=tl.float32, is_pure=True, pack=1)
                        by0 = tl.inline_asm_elementwise('shfl.sync.idx.b32 $0, $1, $2, 31, -1;',
                            constraints='=f,f,r', args=[target_y0,index], dtype=tl.float32, is_pure=True, pack=1)
                        bx1 = tl.inline_asm_elementwise('shfl.sync.idx.b32 $0, $1, $2, 31, -1;',
                            constraints='=f,f,r', args=[target_x1,index], dtype=tl.float32, is_pure=True, pack=1)
                        by1 = tl.inline_asm_elementwise('shfl.sync.idx.b32 $0, $1, $2, 31, -1;',
                            constraints='=f,f,r', args=[target_y1,index], dtype=tl.float32, is_pure=True, pack=1)
                        bx, by = tl.where(j<32,bx0,bx1), tl.where(j<32,by0,by1)
                    else:
                        bx = tl.load(T + (b0 + j) * 2, reachable, other=0.)
                        by = tl.load(T + (b0 + j) * 2 + 1, reachable, other=0.)
                    cost = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
                    value = tl.sqrt(cost) + predecessor
                    tile_value = tl.where(reachable & (value <= limit), value, float('inf'))
                    tl.store(work + cb * BLOCK + j, tile_value, valid)
                    if RECORD_WORK:
                        evaluations += tl.sum(reachable.to(tl.int64), 0)
                        issued += TILE
                tl.debug_barrier()
                prev2_lo, prev2_hi = prev_lo, prev_hi
                prev_lo, prev_hi = lo, hi
                diagonal += 1
        checkpoint_current = tl.load(work + ((diagonal + 2) % 3) * BLOCK + full_lane,
            (full_lane >= prev_lo) & (full_lane <= prev_hi), other=float('inf'), volatile=True)
        checkpoint_previous = tl.load(work + ((diagonal + 1) % 3) * BLOCK + full_lane,
            (full_lane >= prev2_lo) & (full_lane <= prev2_hi), other=float('inf'), volatile=True)
        live = tl.minimum(checkpoint_current, checkpoint_previous) <= limit
        # Infinite budgets must not turn unwritten/infinite cells into live cells.
        live = live & (tl.minimum(checkpoint_current, checkpoint_previous) < float('inf'))
        live_lo = tl.min(tl.where(live, full_lane, nb), 0)
        live_hi = tl.max(tl.where(live, full_lane, -1), 0)
        alive = live_hi >= live_lo
    if diagonal == na + nb - 1:
        result = tl.load(work + ((diagonal + 2) % 3) * BLOCK + nb - 1,
                         (nb - 1 >= prev_lo) & (nb - 1 <= prev_hi),
                         other=float('inf'), volatile=True)
    tl.store(OUT + pair, result)
    if RECORD:
        tl.store(STEPS + pair, diagonal)
    if RECORD_WORK:
        tl.store(COUNTERS + pair * 3, evaluations)
        tl.store(COUNTERS + pair * 3 + 1, issued)
        tl.store(COUNTERS + pair * 3 + 2, diagonal)


@triton.jit
def batch_frontier_kernel(P, OFF, T, TOFF, Q, D, OUT, WORK, LIMIT, STEPS,
             BLOCK: tl.constexpr, CHECK: tl.constexpr, RECORD: tl.constexpr,
             COUNTERS=None, RECORD_WORK: tl.constexpr=False):
    pair = tl.program_id(0)
    q, d = tl.load(Q + pair), tl.load(D + pair)
    na = tl.load(OFF + q + 1) - tl.load(OFF + q)
    if na >= 64:
        frontier_kernel(P, OFF, T, TOFF, Q, D, OUT, WORK, LIMIT, STEPS, COUNTERS,
                 BLOCK, 32, CHECK, RECORD, RECORD_WORK)
    else:
        _native_kernel(P, OFF, T, TOFF, Q, D, OUT, WORK, LIMIT, STEPS, BLOCK, CHECK, RECORD)
        if RECORD_WORK:
            nb = tl.load(TOFF + d + 1) - tl.load(TOFF + d)
            steps = tl.load(STEPS + pair).to(tl.int64)
            a = tl.maximum(steps, 0)
            b = tl.maximum(steps - na, 0)
            c = tl.maximum(steps - nb, 0)
            e = tl.maximum(steps - na - nb, 0)
            cells = (a * (a + 1) - b * (b + 1) - c * (c + 1) + e * (e + 1)) // 2
            tl.store(COUNTERS + pair * 3, cells)
            tl.store(COUNTERS + pair * 3 + 1, steps * BLOCK)
            tl.store(COUNTERS + pair * 3 + 2, steps)
