"""DTW ordered-probe assets, cached observations, and initial ranking.

Signal positions serve candidate ranking. Independent radial and four-block
positions serve pruning and may be trained separately without changing ranking.
Each checkpoint records its own training stages and parameter checksums.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_ASSET = Path(__file__).resolve().parents[1] / 'assets/porto_dtw_signal16.json'
FORMAT = 'leap.dtw-signals.v1'
PROBE_NAMES = ('signal', 'radial', 'ordered')


def validate_asset(asset):
    if asset.get('format') != FORMAT:
        raise ValueError('Expected a DTW ordered-probe asset')
    for name in PROBE_NAMES:
        values = np.asarray(asset.get(name + '_probes'), dtype=np.float32)
        if values.shape != (16, 2) or not np.isfinite(values).all():
            raise ValueError('Each DTW probe set must contain 16 finite xy positions')
        if hashlib.sha256(values.tobytes()).hexdigest() != asset.get(name + '_probes_sha256'):
            raise ValueError('DTW probe checksum mismatch: ' + name)
    return asset


def load_asset(path=DEFAULT_ASSET):
    return validate_asset(json.loads(Path(path).read_text(encoding='utf-8')))


def create_parameters(asset, anchors, mean, std, device):
    validate_asset(asset)
    if (not np.array_equal(np.asarray(asset['anchors'], np.float32), anchors)
            or not np.array_equal(np.asarray(asset['geometry_mean']), mean)
            or not np.array_equal(np.asarray(asset['geometry_std']), std)):
        raise ValueError('DTW probe asset and base geometry differ; load the matching bundled asset')
    return {name: torch.as_tensor(np.asarray(asset[name + '_probes'], np.float32).copy(),
                                  device=device).contiguous() for name in PROBE_NAMES}


def fingerprint(parameters):
    return hashlib.sha256(b''.join(parameters[name].detach().cpu().numpy().tobytes()
                                  for name in PROBE_NAMES)).hexdigest()


def signature(index, summary, parameters):
    tensors = [index.points, index.offsets]
    tensors.extend(summary[name] for name in PROBE_NAMES)
    tensors.extend(parameters[name] for name in PROBE_NAMES)
    return tuple((t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device) for t in tensors)


@torch.no_grad()
def encode(index, parameters):
    from leap.dtw import dtw_signal
    from leap.dtw import dtw_radial
    from leap.dtw import dtw_probe_intervals
    result = {'signal': dtw_signal.encode(index, parameters['signal']),
              'radial': dtw_radial.encode(index, parameters['radial']),
              'ordered': dtw_probe_intervals.encode(index, parameters['ordered'])}
    if any(not bool(torch.isfinite(t).all()) for t in result.values()):
        raise ValueError('DTW probe encoding overflowed')
    return result


def validate_index(index, parameters):
    s = index.dtw_probe_summary
    n = len(index.coordinates)
    shapes = {'signal': (len(index.points), 16), 'radial': (n, 16, 6), 'ordered': (n, 4, 16, 2)}
    if (not isinstance(s, dict) or set(s) != set(PROBE_NAMES)
            or any(s[name].shape != shapes[name] or s[name].dtype != torch.float32
                   or s[name].device != index.points.device or not s[name].is_contiguous()
                   for name in PROBE_NAMES)
            or index.dtw_probe_signature != signature(index, s, parameters)):
        raise ValueError('DTW probe cache or positions changed; re-encode and rebuild')


def slice_summary(summary, lo, hi):
    # Signals index the shared packed point array; the other fields index rows.
    return {'signal': summary['signal'], 'radial': summary['radial'][lo:hi],
            'ordered': summary['ordered'][lo:hi]}


@torch.no_grad()
def candidates(queries, database, k, qids=None, pool=128, *, routing_tree=None, bucket=False):
    from leap.dtw.ordered_dtw import candidates as initial_candidates
    from leap.common.interval_index import lex_keys
    if bucket:
        from leap.dtw.dtw_signal_bucket import pairs
    else:
        from leap.dtw.dtw_signal import pairs
    available = len(database.coordinates) - int(qids is not None)
    if not 1 <= k <= available:
        raise ValueError('Candidate count exceeds available database trajectories')
    width = min(available, max(int(pool), k))
    ids = (initial_candidates(queries, database, width, qids, max_work=True)
           if routing_tree is None else routing_tree.candidates(queries, width, qids))
    rows = torch.arange(len(ids), device=ids.device)[:, None].expand_as(ids)
    fields = (queries.dtw_probe_summary['signal'], database.dtw_probe_summary['signal'])
    options = {}
    if bucket and getattr(routing_tree, 'signal_plan', None) is not None:
        options['batch_plan'] = routing_tree.signal_plan
    scores = pairs(queries, database, rows.reshape(-1), ids.reshape(-1), fields=fields, **options).reshape_as(ids)
    return (lex_keys(scores, ids).topk(k, largest=False).values & 0xffffffff).contiguous()
