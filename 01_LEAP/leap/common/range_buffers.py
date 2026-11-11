"""Bounded first-pass candidate storage shared by HD and DFD range search."""
import torch
import triton
import triton.language as tl


STAGE_CAPACITY = 2048


@triton.jit
def compact_candidates(STAGE, COUNTS, OFF, IDS, ROWS, CAP: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    n = tl.load(COUNTS + row)
    valid = (col < n) & (col < CAP)
    value = tl.load(STAGE + row * CAP + col, valid, other=0)
    base = tl.load(OFF + row)
    tl.store(IDS + base + col, value, valid)
    tl.store(ROWS + base + col, row, valid)


def enumerate_candidates(kernel, args, n, device, groups, block, top_k, max_cells,
                         max_pairs=None):
    counts = torch.empty(n, device=device, dtype=torch.int32)
    visits = torch.empty((n, 2), device=device, dtype=torch.int32)
    capacity = STAGE_CAPACITY
    stage = torch.empty((n, capacity), device=device, dtype=torch.int64)
    kernel[(n,)](*args, counts, stage, counts, counts, visits, groups, block, False,
                  STAGE_CAP=capacity, num_warps=4, enable_fp_fusion=False)
    maximum, total, minimum = torch.stack((counts.max(), counts.sum(), counts.min())).tolist()
    if minimum < top_k:
        raise RuntimeError('Range bound excluded initial neighbors; check geometry and precision')
    if n * maximum > max_cells or (max_pairs is not None and total > max_pairs):
        raise MemoryError('Range batch exceeds bounded reduction; split query batch')
    offsets = counts.long().cumsum(0) - counts
    ids = torch.empty(total, device=device, dtype=torch.int64)
    rows = torch.empty_like(ids)
    # Overflow repeats the complete traversal with exact allocation, never truncates.
    if maximum > capacity:
        written = torch.empty_like(counts)
        kernel[(n,)](*args, offsets, ids, rows, written, visits, groups, block, True,
                      num_warps=4, enable_fp_fusion=False)
        if not torch.equal(written, counts):
            raise RuntimeError('Range changed between count and overflow write passes')
    elif total:
        compact_candidates[(n, triton.cdiv(maximum, 128))](
            stage, counts, offsets, ids, rows, capacity, 128)
    return counts, visits, ids, rows, offsets, maximum


@triton.jit
def scatter_keys(VALUES, IDS, ROWS, OFF, KEYS, TOTAL: tl.constexpr,
                 WIDTH: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = i < TOTAL
    row = tl.load(ROWS + i, valid, other=0)
    base = tl.load(OFF + row, valid, other=0)
    d = tl.load(IDS + i, valid, other=0)
    v = tl.load(VALUES + i, valid, other=0.)
    key = (v.to(tl.int32, bitcast=True).to(tl.int64) << 32) | d.to(tl.int64)
    tl.store(KEYS + row * WIDTH + i - base, key, valid)


def top_keys(values, ids, rows, offsets, n, maximum, top_k):
    keys = torch.full((n, maximum), 9187343244130779135, device=ids.device, dtype=torch.int64)
    if len(ids):
        scatter_keys[(triton.cdiv(len(ids), 256),)](
            values, ids, rows, offsets, keys, len(ids), maximum, 256, num_warps=4)
    return keys.topk(top_k, largest=False).values
