"""Train ordered DTW probe positions; select by validation coarse HR@20 only."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from leap.common.config import RetrievalConfig
from leap.dtw.dtw_probes import DEFAULT_ASSET, load_asset


def load_paths(path, train_size, validation_size, asset):
    def unpack(points, offsets):
        if (points.ndim != 2 or points.shape[1] != 2 or offsets.ndim != 1
                or offsets.dtype.kind not in 'iu' or len(offsets) < 2
                or offsets[0] != 0 or offsets[-1] != len(points)
                or (np.diff(offsets) < 2).any() or not np.isfinite(points).all()):
            raise ValueError('Invalid packed original trajectories')
        return [points[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    with np.load(path, allow_pickle=False) as data:
        for key in ('geometry_mean', 'geometry_std'):
            if key in data and not np.array_equal(data[key], np.asarray(asset[key])):
                raise ValueError('Input normalization differs from the probe asset')
        if 'train_points' in data:
            train = unpack(data['train_points'], data['train_offsets'])
            validation = unpack(data['validation_points'], data['validation_offsets'])
            if len(train) != train_size or len(validation) != validation_size:
                raise ValueError('Requested split sizes differ from packed splits')
        else:
            if 'offsets' in data:
                offsets = data['offsets']
            else:
                lengths = data['lengths']
                if lengths.ndim != 1 or lengths.dtype.kind not in 'iu':
                    raise ValueError('Lengths must be a one-dimensional integer array')
                offsets = np.r_[0, lengths.astype(np.int64).cumsum()]
            paths = unpack(data['points'], offsets)
            if len(paths) < train_size + validation_size:
                raise ValueError('Not enough original trajectories for train and validation')
            train = paths[:train_size]
            validation = paths[train_size:train_size + validation_size]
    return train, validation


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def top_ids(pool, score, k=20):
    from leap.common.interval_index import lex_keys
    return lex_keys(score, pool).topk(k, largest=False).values & 0xffffffff


@torch.no_grad()
def prepare_pairs(index, validation=False):
    from leap.dtw.ordered_dtw import candidates
    from leap.dtw.dtw import dtw_pairs
    n = len(index.coordinates)
    own = torch.arange(n, device=index.points.device)
    pool = candidates(index, index, min(128, n - 1), own, max_work=True)
    rows = own[:, None].expand_as(pool)
    raw = dtw_pairs(index, index, rows.reshape(-1), pool.reshape(-1)).reshape_as(pool)
    result = {'pool': pool, 'raw': raw, 'radius': raw.topk(20, largest=False).values[:, -1]}
    if validation:
        ids = own[None].expand(n, -1)
        rows = own[:, None].expand_as(ids)
        distances = dtw_pairs(index, index, rows.reshape(-1), ids.reshape(-1)).reshape_as(ids)
        distances.fill_diagonal_(torch.inf)
        result['truth'] = top_ids(ids, distances)
    return result


def ranking_loss(predicted, truth, radius):
    sp, sn = predicted[:, :20], predicted[:, 20:]
    dp, dn = truth[:, :20], truth[:, 20:]
    margin = .1 * (dn[:, None, :] - dp[:, :, None]).clamp_min(0)
    return ((sp[:, :, None] - sn[:, None, :] + margin)
            / radius[:, None, None].clamp_min(1e-5)).clamp_min(0).mean()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-inputs', type=Path, required=True)
    parser.add_argument('--asset', type=Path, default=DEFAULT_ASSET)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--train-size', type=int, default=7000)
    parser.add_argument('--validation-size', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; existing training is never overwritten')
    if args.epochs < 1 or args.train_size < 29 or args.validation_size < 21:
        raise ValueError('Require positive epochs, at least29 training and21 validation trajectories')
    from leap.common.dual_field import DualFieldRetriever
    from leap.dtw import dtw_signal
    from leap.common.interval_index import lex_keys
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    asset = load_asset(args.asset)
    model = DualFieldRetriever.from_asset(args.asset, device=args.device, config=RetrievalConfig('dtw', 'weighted'))
    paths = load_paths(args.raw_inputs, args.train_size, args.validation_size, asset)
    tr, va = model.encode(paths[0]), model.encode(paths[1])
    initial = np.asarray(asset['initial_signal_probes'], np.float32)
    if (initial.shape != (16, 2) or not np.isfinite(initial).all()
            or hashlib.sha256(initial.tobytes()).hexdigest() != asset['initial_signal_probes_sha256']):
        raise ValueError('Initialization checksum mismatch')
    theta0 = torch.as_tensor(initial.copy(), device=model.device)
    theta = torch.nn.Parameter(theta0.clone())
    optimizer = torch.optim.Adam([theta], lr=.01)
    args.output.mkdir(parents=True)
    protocol = {'epochs': args.epochs, 'seed': args.seed, 'train_size': args.train_size,
                'validation_size': args.validation_size, 'query_batch': 32, 'lr': .01,
                'optimizer': 'Adam', 'probes': 16, 'learned_parameters': 32, 'candidate_pool': 128,
                'positive_k': 20, 'hard_negative_k': 8, 'margin_teacher_gap': .1,
                'teacher': 'original unbanded Euclidean-sum DTW; within-pool training labels',
                'selection': 'global validation HR20; earliest epoch on ties',
                'test_selection_enabled': False, 'test_evaluated_during_training': False,
                'initial_asset_sha256': hashlib.sha256(args.asset.read_bytes()).hexdigest()}
    save(args.output / 'protocol.json', protocol)
    td, vd = prepare_pairs(tr), prepare_pairs(va, validation=True)
    n = tr.offsets[1:] - tr.offsets[:-1]
    j = torch.arange(int(n.max()), device=model.device)[None]
    p = tr.points[(tr.offsets[:-1, None] + j).clamp_max(len(tr.points) - 1)]
    ps = lex_keys(td['raw'], td['pool']).topk(20, largest=False).indices
    positive_mask = torch.zeros_like(td['pool'], dtype=torch.bool).scatter_(1, ps, True)
    tq = torch.arange(len(n), device=model.device)[:, None].expand_as(td['pool']).reshape(-1)
    vq = torch.arange(len(va.coordinates), device=model.device)[:, None].expand_as(vd['pool']).reshape(-1)
    history, best = [], -1.
    for epoch in range(args.epochs + 1):
        started = time.perf_counter()
        losses, gradients = [], []
        if epoch:
            with torch.no_grad():
                fields = dtw_signal.encode(tr, theta)
                scores = dtw_signal.pairs(tr, tr, tq, td['pool'].reshape(-1), fields=(fields, fields)).reshape_as(td['pool'])
                ns = scores.masked_fill(positive_mask, torch.inf).topk(8, largest=False).indices
            generator = torch.Generator(device=model.device).manual_seed(args.seed + epoch)
            for rr in torch.randperm(len(n), device=model.device, generator=generator).split(32):
                selected = torch.cat([ps[rr], ns[rr]], 1)
                ds = td['pool'][rr].gather(1, selected)
                ids, inv = torch.unique(torch.cat([rr, ds.reshape(-1)]), return_inverse=True)
                lengths = n[ids]
                valid = j < lengths[:, None]
                local = SimpleNamespace(points=p[ids][valid].contiguous(),
                    offsets=torch.cat([torch.zeros(1, device=model.device, dtype=torch.int64), lengths.cumsum(0)]))
                qi = inv[:len(rr), None].expand_as(ds).reshape(-1)
                predicted, _ = dtw_signal.differentiable_pairs(local, local, qi, inv[len(rr):], theta)
                loss = ranking_loss(predicted.reshape_as(ds), td['raw'][rr].gather(1, selected), td['radius'][rr])
                optimizer.zero_grad()
                loss.backward()
                if not bool(torch.isfinite(loss) & torch.isfinite(theta.grad).all()):
                    raise RuntimeError('Nonfinite loss or probe gradient')
                gradients.append(float(theta.grad.norm()))
                torch.nn.utils.clip_grad_norm_([theta], 10.)
                optimizer.step()
                losses.append(float(loss))
        with torch.no_grad():
            fields = dtw_signal.encode(va, theta)
            scores = dtw_signal.pairs(va, va, vq, vd['pool'].reshape(-1), fields=(fields, fields)).reshape_as(vd['pool'])
            if not bool(torch.isfinite(scores).all() & (scores <= vd['raw'] + 2e-4 * (1 + vd['raw'])).all()):
                raise RuntimeError('DTW signal failed validation numerical checks')
            predicted = top_ids(vd['pool'], scores)
            hr20 = int((predicted[:, :, None] == vd['truth'][:, None, :20]).sum()) / (len(predicted) * 20)
            hr1 = int((predicted[:, 0] == vd['truth'][:, 0]).sum()) / len(predicted)
        row = {'epoch': epoch, 'seconds': time.perf_counter() - started, 'HR1': hr1, 'HR20': hr20,
               'loss': float(np.mean(losses)) if losses else None,
               'gradient_norm': float(np.mean(gradients)) if gradients else None}
        history.append(row)
        snapshot = copy.deepcopy(asset)
        values = theta.detach().cpu().numpy()
        snapshot.update(signal_probes=values.tolist(), signal_probes_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                        selected_epoch=epoch, seed=args.seed, validation=row, training=protocol,
                        test_selection_enabled=False)
        save(args.output / f'epoch_{epoch:03d}.json', snapshot)
        if hr20 > best:
            best = hr20
            save(args.output / 'best.json', snapshot)
        save(args.output / 'history.json', history)
        torch.save({'theta': theta.detach().cpu(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'protocol': protocol}, args.output / 'latest.pt')
        print(json.dumps(row), flush=True)
    save(args.output / 'TRAINING_DONE.json', {'completed': True, 'epochs': args.epochs})


if __name__ == '__main__':
    main()
