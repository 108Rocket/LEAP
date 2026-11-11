"""Read native Kineto CUDA events without serializing unrelated CPU activities."""
import gzip
import json
from pathlib import Path
import torch
from trace_tools import union_us


def extract_native(profile, compact_path):
    result = profile.profiler.kineto_results
    events = result.events()
    selected = [event for event in events if event.device_type() == torch.autograd.DeviceType.CUDA
                and not event.is_user_annotation()]
    if not selected:
        raise RuntimeError('No native CUDA events; never substitute elapsed wall time')
    origin = min(event.start_ns() for event in selected)
    kernels, copies = [], []
    categories = dict(kernel=0, gpu_memcpy=0, gpu_memset=0,
                      non_cuda_ignored=len(events) - len(selected))
    for event in selected:
        name = event.name()
        category = 'gpu_memcpy' if name.startswith('Memcpy') else (
            'gpu_memset' if name.startswith('Memset') else 'kernel')
        device = event.device_index()
        assert device == 0, 'One visible GPU only'
        duration = event.duration_ns()
        assert duration >= 0
        row = dict(name=name, ts=(event.start_ns() - origin) / 1000.0,
                   dur=duration / 1000.0, stream=event.device_resource_id(), device=device)
        categories[category] += 1
        (kernels if category == 'kernel' else copies).append(row)
    if not kernels:
        raise RuntimeError('No CUDA kernels; copies are not encoding computation')
    busy = union_us([(row['ts'], row['ts'] + row['dur']) for row in kernels])
    summed = sum(row['dur'] for row in kernels)
    assert busy <= summed + 1e-5
    compact = dict(schema=2, capture='CUDA-only native Kineto events',
        time_unit='microseconds', timestamp_origin_ns=str(origin), kernels=kernels,
        copies=copies, categories=categories, kernel_union_us=busy, kernel_sum_us=summed)
    with gzip.open(Path(compact_path), 'wt', encoding='utf-8', compresslevel=1) as stream:
        json.dump(compact, stream, separators=(',', ':'))
    return dict(kernel_seconds=busy / 1e6, kernel_sum_seconds=summed / 1e6,
                kernels=len(kernels), excluded_copy_or_memset_events=len(copies),
                categories=categories, capture='CUDA-only native Kineto events')
