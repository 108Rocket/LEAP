"""Complete native database encoding or separate CUDA-only kernel timing."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

MODELS = ('ConvTraj', 'SIMformer', 'TrajCL', 'MovSemCL', 'TSMini', 'TrajAgg')


def arguments():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--metric', required=True, choices=('hausdorff', 'dfd', 'dtw'))
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--offsets', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--prior-asset-root', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('total', 'gpu'), default='gpu')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Use separate interleaved invocations for the three observations')
    return parser.parse_args()


def main():
    args = arguments()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; never overwrite timing evidence')
    os.environ['LEAP_EXTERNAL_ASSETS'] = str(args.asset_root.resolve())
    if args.prior_asset_root:
        os.environ['LEAP_EXTERNAL_PRIOR_ASSETS'] = str(args.prior_asset_root.resolve())
    if args.source_root:
        os.environ['LEAP_EXTERNAL_SOURCES'] = str(args.source_root.resolve())
    import numpy as np
    import torch
    import sdd_models
    from gpu_activity_capture import extract_native

    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2000)
    np.random.seed(2000)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU using CUDA_VISIBLE_DEVICES')
    torch.cuda.set_per_process_memory_fraction(.97 if args.model == 'TSMini' else .65)
    points, offsets = np.load(args.points), np.load(args.offsets)
    if points.ndim != 2 or points.shape[1] != 2 or offsets.ndim != 1:
        raise ValueError('Expected packed XY points and one-dimensional offsets')
    if offsets[0] != 0 or offsets[-1] != len(points) or not np.all(np.diff(offsets) > 0):
        raise ValueError('Invalid packed offsets')
    if not np.isfinite(points).all():
        raise ValueError('Non-finite input coordinates')
    metric = 'discret_frechet' if args.metric == 'dfd' else args.metric
    native, _, prepare, checkpoint = sdd_models.construct(args.model, metric)
    count = len(offsets) - 1

    def batch(ids):
        return native([prepare(points[offsets[i]:offsets[i+1]]) for i in ids])

    unique = {}
    for i, length in enumerate(np.diff(offsets)):
        unique.setdefault(int(length), i)
    warm = sorted(set(range(min(2048, count))) | set(unique.values()))
    with torch.inference_mode():
        for start in range(0, len(warm), 128):
            batch(warm[start:start+128])
    torch.cuda.synchronize()
    args.output.mkdir(parents=True)
    observations = []
    for repeat in range(args.repeats):
        folder = args.output / str(repeat)
        folder.mkdir()
        windows = []

        def capture(fn):
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False, profile_memory=False, with_stack=False) as profile:
                result = fn()
                torch.cuda.synchronize()
            windows.append(extract_native(profile, folder / ('window-%04d.gpu.json.gz' % len(windows))))
            del profile
            gc.collect()
            return result

        def encode():
            pieces = []
            for lo in range(0, count, 4096):
                def part():
                    return [batch(range(start, min(start+128, count)))
                            for start in range(lo, min(lo+4096, count), 128)]
                pieces.extend(capture(part) if args.mode == 'gpu' else part())
            return capture(lambda: torch.cat(pieces)) if args.mode == 'gpu' else torch.cat(pieces)

        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = encode()
            torch.cuda.synchronize()
            wall = time.perf_counter() - start
        if value.shape[0] != count or not torch.isfinite(value).all():
            raise RuntimeError('Incomplete or non-finite encoding')
        array = value.detach().contiguous().cpu().numpy()
        digest = hashlib.sha256()
        digest.update(b'embedding')
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(memoryview(array).cast('B'))
        row = dict(repeat=repeat, total_seconds=wall if args.mode == 'total' else None,
            gpu_kernel_seconds=sum(w['kernel_seconds'] for w in windows) if windows else None,
            observer_wall_seconds=wall if args.mode == 'gpu' else None,
            output_sha256=digest.hexdigest(), shape=list(array.shape), windows=windows)
        if observations and row['output_sha256'] != observations[0]['output_sha256']:
            raise RuntimeError('Encoding changed between observations')
        observations.append(row)
        (folder / 'RESULT.json').write_text(json.dumps(row, indent=2))
        del value, array
    field = 'total_seconds' if args.mode == 'total' else 'gpu_kernel_seconds'
    result = dict(model=args.model, metric=args.metric, mode=args.mode, database_rows=count,
        database_points=len(points), native_batch=128, threads=12, tf32=False,
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        aggregation='Fixed-checkpoint arithmetic mean, not independent training seeds',
        mean_seconds=statistics.mean(row[field] for row in observations), observations=observations)
    (args.output / 'RESULT.json').write_text(json.dumps(result, indent=2))
    (args.output / 'ALL_DONE.json').write_text(json.dumps(dict(success=True,
        result_sha256=hashlib.sha256((args.output / 'RESULT.json').read_bytes()).hexdigest())))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
