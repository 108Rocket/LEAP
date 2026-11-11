"""Shared sorting and scatter for separate DFD filter and target-DP buckets."""
import torch
import triton
import triton.language as tl
from leap.dfd.dfd_refinement import ordered_fast, original_value

MIN_WIDTH = 64
WORKSPACE_VALUES = 32 * 1024 * 1024


@triton.jit
def cached_kernel(P, OFF, T, TOFF, Q, D, INITIAL, RAW, RADIUS, O, FLAGS, WORK,
                  BLOCK: tl.constexpr, K: tl.constexpr, CACHE: tl.constexpr,
                  FILTER_BLOCK: tl.constexpr):
    pair = tl.program_id(0)
    q, d = tl.load(Q + pair), tl.load(D + pair)
    slots = tl.arange(0, CACHE)
    initial = tl.load(INITIAL + q*K + slots, slots < K, other=-1)
    at = tl.min(tl.where(initial == d, slots, CACHE), 0)
    if at < K:
        value = tl.load(RAW + q*K + at)
        flag = tl.full((), 0, tl.int32)
    else:
        r = tl.load(RADIUS + q)
        keep = ordered_fast(P, OFF, q, T, TOFF, d, r*r, FILTER_BLOCK)
        if keep:
            keep = ordered_fast(T, TOFF, d, P, OFF, q, r*r, FILTER_BLOCK)
        if keep:
            value, alive = original_value(P, OFF, q, T, TOFF, d, WORK, pair, r*r, BLOCK, True)
            flag = tl.where(alive, 1, 6)
        else:
            value = tl.full((), float('inf'), P.dtype.element_ty)
            flag = tl.full((), 2, tl.int32)
    tl.store(O + pair, value)
    tl.store(FLAGS + pair, flag)


@torch.no_grad()
def cached_pairs(query, database, query_ids, database_ids, initial, raw, threshold):
    size = len(database_ids)
    output = torch.empty(size, device=database_ids.device, dtype=query.points.dtype)
    status = torch.empty(size, device=database_ids.device, dtype=torch.int32)
    if not size:
        return output, status
    maximum = max(MIN_WIDTH, query.dfd_summary['block'], database.dfd_summary['block'])
    widths = [2**i for i in range(6, maximum.bit_length())]
    boundaries = torch.tensor(widths, device=database_ids.device)
    query_lengths = query.offsets[query_ids + 1] - query.offsets[query_ids]
    target_lengths = database.offsets[database_ids + 1] - database.offsets[database_ids]
    target_bucket = torch.bucketize(target_lengths, boundaries)
    filter_bucket = torch.bucketize(torch.maximum(query_lengths, target_lengths), boundaries)
    keys = target_bucket * len(widths) + filter_bucket
    order = keys.argsort(stable=True)
    sizes = torch.bincount(keys, minlength=len(widths)**2).cpu().tolist()
    queries, targets = query_ids[order].contiguous(), database_ids[order].contiguous()
    values, flags = torch.empty_like(output), torch.empty_like(status)
    start = 0
    for key, count in enumerate(sizes):
        if not count:
            continue
        width, filter_width = widths[key // len(widths)], widths[key % len(widths)]
        chunk = max(1, min(count, WORKSPACE_VALUES // (3 * width)))
        work = torch.empty((chunk, 3, width), device=database_ids.device, dtype=query.points.dtype)
        for lo in range(start, start + count, chunk):
            hi = min(lo + chunk, start + count)
            cached_kernel[(hi-lo,)](query.points, query.offsets, database.points, database.offsets,
                queries[lo:hi], targets[lo:hi], initial, raw, threshold, values[lo:hi], flags[lo:hi],
                work, width, initial.shape[1], max(MIN_WIDTH, triton.next_power_of_2(initial.shape[1])),
                filter_width, num_warps=4, enable_fp_fusion=False)
        start += count
    if start != size:
        raise RuntimeError('DFD bucket coverage differs from the pair list')
    output[order], status[order] = values, flags
    return output, status
