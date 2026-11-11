"""Stable single-query compaction with two bounded host count reads."""
import torch
import triton
import triton.language as tl


@triton.jit
def leaf_scan(COUNTS, OFFSETS, TOTAL, PARTS: tl.constexpr):
    j = tl.arange(0, PARTS)
    counts = tl.load(COUNTS + j)
    tl.store(OFFSETS + j, tl.cumsum(counts, 0) - counts)
    tl.store(TOTAL, tl.sum(counts, 0))


@triton.jit
def member_counts(KEEP, EXAMINED, COUNTS, CHECKED, B: tl.constexpr):
    task = tl.program_id(0)
    j = tl.arange(0, B)
    tl.store(COUNTS + task, tl.sum(tl.load(KEEP + task * B + j), 0))
    tl.store(CHECKED + task, tl.sum(tl.load(EXAMINED + task * B + j), 0))


@triton.jit
def pair_scan(COUNTS, CHECKED, OFFSETS, TOTAL, VISITS, ZERO,
              TASKS, WIDTH: tl.constexpr):
    j = tl.arange(0, WIDTH)
    counts = tl.load(COUNTS + j, j < TASKS, other=0)
    checked = tl.load(CHECKED + j, j < TASKS, other=0)
    tl.store(OFFSETS + j, tl.cumsum(counts, 0) - counts, j < TASKS)
    tl.store(TOTAL, tl.sum(counts, 0))
    tl.store(VISITS + 1, tl.sum(checked, 0))
    tl.store(ZERO, 0)


@torch.no_grad()
def enumerate_members(plan, leaves, leaf_counts, visits, parts, queries, index,
                      tree, extra, qgeom, qids, membership, radius, slack,
                      ragged, mark):
    from leap.dtw import dtw_parallel as dp
    from leap.common.runtime import options as opt

    g = tree.groups
    b = tree.block
    s = membership.shape[1]
    device = qids.device
    offsets = plan.storage('dtw_leaf_offsets', (parts,), torch.int32)
    host_count = plan.storage('dtw_leaf_total', (1,), torch.int32)
    leaf_scan[(1,)](leaf_counts, offsets, host_count, parts)
    tasks = int(host_count[0])
    if tasks * b > dp.MAX_MEMBER_CELLS:
        raise MemoryError('Bounded parallel leaf workspace; split query batch')

    lids = plan.storage('dtw_leaf_ids', (tasks,), torch.int32)
    lrows = plan.storage('dtw_leaf_rows', (tasks,), torch.int32)
    if tasks:
        # Every subtree gets its full capacity, including all-pass queries.
        dp.compact_leaves[(parts, triton.cdiv(g // parts, 128))](
            leaves, leaf_counts, offsets, lids, lrows, g // parts, 128, parts)
    keep = plan.storage('dtw_member_keep', (tasks, b), torch.int32)
    examined = plan.storage('dtw_member_examined', (tasks, b), torch.int32)
    mark('leaf_task_layout')
    if tasks:
        block = max(extra['block'], qgeom['block'])
        tile = 4 if block >= 4096 else b
        dp.check_leaf_buckets(
            index.points, index.offsets, queries.points, queries.offsets,
            tree.members, extra['box'], qgeom['box'], radius, slack, qids,
            membership, lids, lrows, keep, examined, b, block, s, tile,
            extra.get('leaf_max') if opt.dtw_length_cache else None)
    mark('parallel_member_filter')

    per_leaf = plan.storage('dtw_member_counts', (tasks,), torch.int32)
    checked = plan.storage('dtw_member_checked', (tasks,), torch.int32)
    pair_offsets = plan.storage('dtw_pair_offsets', (tasks,), torch.int32)
    # Public outputs own their storage; a later request cannot overwrite them.
    count = torch.empty(1, device=device, dtype=torch.int32)
    off = torch.empty(1, device=device, dtype=torch.int64)
    if tasks:
        member_counts[(tasks,)](keep, examined, per_leaf, checked, b)
    pair_scan[(1,)](per_leaf, checked, pair_offsets, count, visits, off,
                    tasks, triton.next_power_of_2(max(1, tasks)), num_warps=8)
    total = int(count[0])
    if (total > 8_000_000 if ragged else s + total > 8_000_000):
        raise MemoryError('DTW range batch exceeds native reduction limit')
    dids = torch.empty(total, device=device, dtype=torch.int64)
    rows = torch.empty_like(dids)
    mark('candidate_layout')
    if tasks:
        dp.write_pairs[(tasks,)](keep, lids, lrows, tree.members, pair_offsets,
                                 dids, rows, b, num_warps=4)
    mark('candidate_write')
    return count, visits, dids, rows, off, total, total, 0
