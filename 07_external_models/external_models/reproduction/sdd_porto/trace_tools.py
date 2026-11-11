"""Count this process's GPU kernel intervals, not CPU launch spans."""
import gzip
import json
from decimal import Decimal
from pathlib import Path


def union_us(intervals):
    intervals = sorted(intervals)
    if not intervals:
        return 0.0
    start, end = intervals[0]
    total = 0.0
    for left, right in intervals[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def extract(trace_path, compact_path):
    with Path(trace_path).open() as stream:
        source = json.load(stream)
    kernels, copies, categories = [], [], {}
    timed = [event for event in source['traceEvents'] if event.get('ph') == 'X' and
             event.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    origin = min((Decimal(str(event['ts'])) for event in timed), default=Decimal(0))
    for event in source['traceEvents']:
        category = event.get('cat', '')
        categories[category] = categories.get(category, 0) + 1
        if event.get('ph') != 'X' or category not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        args = event.get('args', {})
        device = int(args.get('device', 0))
        assert device == 0, 'Only one visible GPU is permitted'
        # Absolute profiler timestamps are too large for subtracting microsecond spans.
        relative = Decimal(str(event['ts'])) - origin
        row = dict(name=event.get('name', ''), ts=float(relative),
                   dur=float(event['dur']), stream=args.get('stream'), device=device)
        assert row['dur'] >= 0
        (kernels if category == 'kernel' else copies).append(row)
    if not kernels:
        raise RuntimeError('No CUPTI GPU kernels captured; do not substitute CPU events')
    intervals = [(row['ts'], row['ts'] + row['dur']) for row in kernels]
    busy = union_us(intervals)
    summed = sum(row['dur'] for row in kernels)
    assert busy <= summed + 1e-5
    compact = dict(schema=1, time_unit='microseconds', timestamp_origin_us=str(origin), kernels=kernels, copies=copies,
                   categories=categories, kernel_union_us=busy, kernel_sum_us=summed)
    with gzip.open(compact_path, 'wt', encoding='utf-8') as stream:
        json.dump(compact, stream, separators=(',', ':'))
    return dict(kernel_seconds=busy / 1e6, kernel_sum_seconds=summed / 1e6,
                kernels=len(kernels), excluded_copy_or_memset_events=len(copies),
                categories=categories)


def self_test():
    assert union_us([(0, 3), (1, 4), (4, 5), (7, 8)]) == 6
    assert union_us([]) == 0
    assert union_us([(0, 2), (0, 2)]) == 2
