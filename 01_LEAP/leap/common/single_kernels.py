"""Small-request kernels reuse the exact existing distance recurrences."""
import triton
import triton.language as tl
from leap.dfd.dfd import dfd_kernel
from leap.dfd.dfd_geometry import direction, witness_direction
from leap.dfd.dfd_refinement import cached_kernel
from leap.dtw.dtw import dtw_kernel
from leap.dtw.dtw_bounded import bounded_kernel
from leap.dtw.dtw_signal import signal_kernel
from leap.dtw.dtw_parallel import check_leaf_tiles


@triton.jit
def pair_bucket(P, OFF, T, TOFF, Q, D, OUT, FLAGS, WORK, LIMIT, INITIAL, RAW,
                BLOCK: tl.constexpr, LOWER: tl.constexpr, KIND: tl.constexpr,
                INITIAL_K: tl.constexpr, CHECK: tl.constexpr):
    pair = tl.program_id(0)
    q = tl.load(Q + pair); d = tl.load(D + pair)
    nb = tl.load(TOFF + d + 1) - tl.load(TOFF + d)
    width = nb
    if KIND == 3:
        na = tl.load(OFF + q + 1) - tl.load(OFF + q)
        width = tl.maximum(na, nb)
    if (width > LOWER) & (width <= BLOCK):
        if KIND == 0:
            dfd_kernel(P, OFF, T, TOFF, Q, D, OUT, WORK, BLOCK)
        elif KIND == 1:
            dtw_kernel(P, OFF, T, TOFF, Q, D, OUT, WORK, BLOCK)
        elif KIND == 2:
            bounded_kernel(P, OFF, T, TOFF, Q, D, OUT, WORK, LIMIT, FLAGS,
                           BLOCK, CHECK, False)
        else:
            cached_kernel(P, OFF, T, TOFF, Q, D, INITIAL, RAW, LIMIT, OUT,
                          FLAGS, WORK, BLOCK, INITIAL_K, 64)


@triton.jit
def signal_bucket(A, AO, B, BO, Q, D, OUT, WORK, M: tl.constexpr,
                  BLOCK: tl.constexpr, LOWER: tl.constexpr):
    pair = tl.program_id(0); d = tl.load(D + pair)
    n = tl.load(BO + d + 1) - tl.load(BO + d)
    if (n > LOWER) & (n <= BLOCK):
        signal_kernel(A, AO, B, BO, Q, D, OUT, WORK, WORK, M, BLOCK, False)


@triton.jit
def dfd_rank_bucket(P, OFF, T, TOFF, IDS, LOW, WQ, WD, KEYS,
                    BLOCK: tl.constexpr, LOWER: tl.constexpr):
    pair = tl.program_id(0); d = tl.load(IDS + pair)
    na = tl.load(OFF + 1) - tl.load(OFF)
    nb = tl.load(TOFF + d + 1) - tl.load(TOFF + d)
    size = tl.maximum(na, nb)
    if (size > LOWER) & (size <= BLOCK):
        a = witness_direction(P, OFF, 0, T, TOFF, d, 8)
        b = witness_direction(T, TOFF, d, P, OFF, 0, 8)
        witness = tl.sqrt(tl.maximum(a, b))
        a = direction(P, OFF, 0, T, TOFF, d, WQ, 8, BLOCK, True)
        b = direction(T, TOFF, d, P, OFF, 0, WD, 8, BLOCK, True)
        ordered = tl.sqrt(tl.maximum(a, b))
        score = tl.maximum(tl.maximum(tl.load(LOW + pair), witness), ordered)
        key = (score.to(tl.int32, bitcast=True).to(tl.int64) << 32) | d
        tl.store(KEYS + pair, key)


@triton.jit
def leaf_bucket(P, OFF, QP, QOFF, MEM, BOX, QBOX, RADIUS, SLACK, SELF, SEEDS,
                LIDS, QIDS, KEEP, EXAMINED, LEAF_WIDTH, B: tl.constexpr,
                N: tl.constexpr, LOWER: tl.constexpr, S: tl.constexpr,
                TILE: tl.constexpr, SHORT_CIRCUIT: tl.constexpr=False):
    task = tl.program_id(0); leaf = tl.load(LIDS + task)
    row = tl.load(QIDS + task)
    width = tl.maximum(tl.load(LEAF_WIDTH + leaf),
                       tl.load(QOFF + row + 1) - tl.load(QOFF + row))
    if (width > LOWER) & (width <= N):
        check_leaf_tiles(P, OFF, QP, QOFF, MEM, BOX, QBOX, RADIUS, SLACK,
                         SELF, SEEDS, LIDS, QIDS, KEEP, EXAMINED,
                         B, N, S, TILE, 64, False, 0, SHORT_CIRCUIT)


@triton.jit
def endpoints_kernel(P, OFF, OUT):
    lo = tl.load(OFF); hi = tl.load(OFF + 1) - 1
    j = tl.arange(0, 4)
    pos = tl.where(j < 2, lo, hi) * 2 + j % 2
    tl.store(OUT + j, tl.load(P + pos))


@triton.jit
def decode_sort(KEYS, OUT, WIDTH: tl.constexpr, K: tl.constexpr):
    i = tl.arange(0, WIDTH)
    values = tl.sort(tl.load(KEYS + i), descending=False)
    tl.store(OUT + i, values & 0xffffffff, i < K)
