"""Continue a matched DTW router on a frozen train pool; select using validation only."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from leap import RetrievalConfig
from leap import LEAP
from leap.dtw import dtw_routing as routing
from leap.dtw.dtw_aligned_loss import aligned_radius_loss
from leap.dtw.dtw_signal_bucket import pairs
from leap.common.interval_index import lex_keys
from leap.dtw.ordered_dtw import candidates
from leap.training.train_dtw_probes import save
from leap.training.train_dtw_routing import encode, with_summary


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_pool(ids, valid, positive, count):
    if ids.shape != valid.shape or ids.shape != (count, 276) or valid.dtype != np.bool_:
        raise ValueError('Frozen pool must contain 276 columns and a boolean validity mask')
    if not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= count)):
        raise ValueError('Pool IDs are outside the training database')
    if not np.array_equal(ids[:, :20], positive) or not valid[:, :20].all():
        raise ValueError('The first 20 pool columns must be native training neighbors')
    for row, (values, mask) in enumerate(zip(ids, valid)):
        kept = values[mask]
        if len(kept) <= 128 or row in kept or len(np.unique(kept)) != len(kept):
            raise ValueError('Valid pool entries must be distinct, exclude self and exceed 128')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('raw-inputs', 'train-labels', 'validation-labels', 'asset', 'routing-asset', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--frozen-pool', type=Path)
    ap.add_argument('--labels-normalized', action='store_true')
    ap.add_argument('--objective', choices=('coverage', 'aligned'), default='aligned')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=2000)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args(argv)
    if args.epochs < 1 or not args.device.startswith('cuda'):
        raise ValueError('Use positive epochs and a CUDA device')
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = LEAP.from_asset(args.asset, device=args.device, routing_asset=args.routing_asset,
        config=RetrievalConfig('dtw', dtw_seed='signal', dtw_pool=128, dtw_routing='learned'))
    raw = np.load(args.raw_inputs)
    if not np.array_equal(raw['geometry_mean'], model.mean) or not np.array_equal(raw['geometry_std'], model.std):
        raise ValueError('Raw geometry normalization differs from the signal asset')
    if not args.labels_normalized and model.std[0] != model.std[1]:
        raise ValueError('Raw labels need isotropic normalization; otherwise supply normalized labels explicitly')
    scale = 1. if args.labels_normalized else float(model.std[0])
    tr = model.encode_packed(raw['train_points'], raw['train_offsets'])
    va = model.encode_packed(raw['validation_points'], raw['validation_offsets'])
    n, v = len(tr.coordinates), len(va.coordinates)
    if n < 277 or v < 1:
        raise ValueError('Need at least 277 training trajectories and one validation query')
    matrix = np.load(args.train_labels, mmap_mode='r')
    vm = np.load(args.validation_labels, mmap_mode='r')
    if matrix.shape != (n, n) or vm.shape != (v, n):
        raise ValueError('DTW label matrix shapes differ from the raw splits')
    if not np.isfinite(matrix).all() or not np.isfinite(vm).all() or np.any(matrix < 0) or np.any(vm < 0):
        raise ValueError('DTW labels must be finite and nonnegative')
    native = np.array(matrix, copy=True)
    np.fill_diagonal(native, np.inf)
    positive = np.argsort(native, axis=1, kind='stable')[:, :20]
    valtruth = np.argsort(vm, axis=1, kind='stable')[:, :20]
    td, vd = routing.prepare(tr), routing.prepare(va)
    if args.frozen_pool:
        frozen = np.load(args.frozen_pool)
        pool, mask = frozen['ids'], frozen['valid']
    else:
        initial = candidates(tr, tr, 256, torch.arange(n, device=args.device), max_work=True).cpu().numpy()
        pool = np.concatenate((positive, initial), axis=1)
        mask = np.zeros(pool.shape, dtype=bool)
        for row in range(n):
            seen = {row}
            for col, value in enumerate(pool[row]):
                if int(value) not in seen:
                    mask[row, col] = True
                    seen.add(int(value))
    validate_pool(pool, mask, positive, n)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'FROZEN_TRAIN_POOL.npz', ids=pool, valid=mask)
    ids = torch.tensor(pool, device=args.device, dtype=torch.int64)
    valid = torch.tensor(mask, device=args.device)
    distances = torch.tensor(np.asarray(matrix)[np.arange(n)[:, None], pool] / scale,
                             device=args.device, dtype=torch.float32)
    radius = distances[:, :20].amax(1)
    qr = torch.arange(n, device=args.device)[:, None].expand_as(ids).contiguous().flatten()
    flat = ids.contiguous().flatten()
    with torch.no_grad():
        signal = torch.cat([pairs(tr, tr, qr[lo:lo+16384], flat[lo:lo+16384],
            fields=(tr.dtw_probe_summary['signal'], tr.dtw_probe_summary['signal']))
            for lo in range(0, len(flat), 16384)]).reshape_as(ids).detach()
    if not bool(torch.isfinite(signal).all()):
        raise FloatingPointError('Nonfinite fixed signal scores')

    @torch.no_grad()
    def validation(net):
        qs, ds = encode(net, vd), encode(net, td)
        candidate = candidates(with_summary(va, qs), with_summary(tr, ds), 128, None, max_work=True)
        rows = torch.arange(v, device=args.device)[:, None].expand_as(candidate).contiguous()
        score = pairs(va, tr, rows.flatten(), candidate.flatten(),
            fields=(va.dtw_probe_summary['signal'], tr.dtw_probe_summary['signal'])).reshape_as(candidate)
        selected = (lex_keys(score, candidate).topk(20, largest=False).values & 0xffffffff).cpu().numpy()
        def quality(values):
            return float((values[:, :, None] == valtruth[:, None]).any(1).mean())
        return dict(mean_radius=float((np.asarray(vm)[np.arange(v)[:, None], selected] / scale).max(1).mean()),
                    candidate_hr20=quality(selected), pool_coverage=quality(candidate.cpu().numpy()))

    net = deepcopy(model.dtw_router).requires_grad_(True)
    baseline = validation(net)
    protocol = dict(train=n, validation=v, candidate_columns=276, epochs=args.epochs, seed=args.seed,
        objective=args.objective, lr=.0003, batch=64, auxiliary_weight=.5, fixed_probes=True,
        test_loaded=False, pool_refreshed=False, labels_normalized=args.labels_normalized,
        inputs={key: sha(getattr(args, key)) for key in
                ('raw_inputs', 'train_labels', 'validation_labels', 'asset', 'routing_asset')},
        frozen_pool_sha256=sha(args.output / 'FROZEN_TRAIN_POOL.npz'), baseline_validation=baseline,
        selection='No validation pool or ranking decrease; earliest smaller actual radius; E0 fallback')
    save(args.output / 'protocol.json', protocol)
    optimizer = torch.optim.Adam(net.parameters(), lr=.0003)
    best, selected_epoch = baseline['mean_radius'], 0
    selected_state = deepcopy(net.state_dict())
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        losses = []
        generator = torch.Generator(device=args.device).manual_seed(args.seed + epoch)
        for rows in torch.randperm(n, device=args.device, generator=generator).split(64):
            subset = ids[rows]
            unique, inverse = torch.unique(torch.cat((rows, subset.flatten())), return_inverse=True)
            sums, logmass, _ = net(*(x[unique] for x in td), details=True)
            qrows, drows = inverse[:len(rows)], inverse[len(rows):].reshape_as(subset)
            score = routing.pair_scores(sums[qrows, None], sums[drows], td[1][rows, None], td[1][subset])
            negative = valid[rows].clone()
            negative[:, :20] = False
            boundary = score.masked_fill(~negative, torch.inf).topk(109, largest=False).values[:, -1]
            loss = routing.coverage_loss(score[:, :20], boundary, radius[rows])
            loss = loss + 1e-5 * logmass.square().sum() / td[1][unique].sum()
            if args.objective == 'aligned':
                loss = loss + .5 * aligned_radius_loss(score, signal[rows], distances[rows], valid[rows], radius[rows])
            optimizer.zero_grad()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            if not bool(torch.isfinite(loss) & torch.isfinite(gradient)):
                raise FloatingPointError('Nonfinite continuation loss/gradient')
            optimizer.step()
            losses.append(float(loss))
        check = validation(net)
        eligible = (check['pool_coverage'] >= baseline['pool_coverage'] - 1e-12
                    and check['candidate_hr20'] >= baseline['candidate_hr20'] - 1e-12)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), eligible=eligible, **check))
        if eligible and check['mean_radius'] < best:
            best, selected_epoch = check['mean_radius'], epoch
            selected_state = deepcopy(net.state_dict())
        torch.save(dict(epoch=epoch, state_dict=net.state_dict(), optimizer=optimizer.state_dict()), args.output / 'latest.pt')
        save(args.output / 'history.json', history)
        print(json.dumps(history[-1]), flush=True)
    torch.save(selected_state, args.output / 'selected.pt')
    artifact = json.loads(args.routing_asset.read_text(encoding='utf-8-sig'))
    artifact['state_dict'] = {key: value.cpu().tolist() for key, value in selected_state.items()}
    artifact['weights_sha256'] = routing.weights_hash(artifact['state_dict'])
    artifact['continuation_training'] = dict(protocol=protocol, selected_epoch=selected_epoch)
    save(args.output / 'router_asset.json', artifact)
    save(args.output / 'SUMMARY.json', dict(success=True, selected_epoch=selected_epoch, history=history,
        train_seconds=time.perf_counter() - started, baseline_validation=baseline, test_loaded=False))
    save(args.output / 'ALL_DONE.json', dict(success=True, router_sha256=sha(args.output / 'router_asset.json')))


if __name__ == '__main__':
    main()
