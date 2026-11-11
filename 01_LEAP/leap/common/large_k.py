"""Opt-in large-k SDD retrieval, adopted from the L20 H/DFD parity experiment.

Fused execution preserves FP32 score/ID keys. It still scans every database
summary; refinement is restricted to the selected pool, not global full search.
"""
from dataclasses import replace

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from leap.common.interval_index import lex_keys
from leap.common.runtime.batching import resolve_query_batch


MAX_SCORE_PAIRS = 26_000_000


@triton.jit(do_not_specialize=['QSTART'])
def score_keys(Q, D, QP, DP, QO, DO, OUT, NQ: tl.constexpr, ND: tl.constexpr,
               QS: tl.constexpr, DS: tl.constexpr, DFD: tl.constexpr,
               SELF: tl.constexpr, QSTART,
               BQ: tl.constexpr, BD: tl.constexpr):
    # Flatten the grid to avoid CUDA's grid-y limit on large databases.
    tiles = tl.cdiv(ND, BD)
    qi = (tl.program_id(0) // tiles) * BQ + tl.arange(0, BQ)
    di = (tl.program_id(0) % tiles) * BD + tl.arange(0, BD)
    f = tl.arange(0, 64)
    q = tl.load(Q + qi[:, None] * QS + f[None, :], qi[:, None] < NQ, 0)
    d = tl.load(D + di[:, None] * DS + f[None, :], di[:, None] < ND, 0)
    score = tl.max(tl.abs(q[:, None, :] - d[None, :, :]), 2)
    if DFD:
        for edge in tl.static_range(2):
            qo = tl.load(QO + qi + edge, qi < NQ, 0) - edge
            do = tl.load(DO + di + edge, di < ND, 0) - edge
            qx = tl.load(QP + 2 * qo, qi < NQ, 0)
            qy = tl.load(QP + 2 * qo + 1, qi < NQ, 0)
            dx = tl.load(DP + 2 * do, di < ND, 0)
            dy = tl.load(DP + 2 * do + 1, di < ND, 0)
            x, y = qx[:, None] - dx[None, :], qy[:, None] - dy[None, :]
            score = tl.maximum(score, libdevice.sqrt_rn(x * x + y * y))
    if SELF:
        score = tl.where(qi[:, None] + QSTART == di[None, :], float('inf'), score)
    bits = score.to(tl.int32, bitcast=True).to(tl.int64)
    key = (bits << 32) | di[None, :].to(tl.int64)
    tl.store(OUT + qi[:, None] * ND + di[None, :], key,
             (qi[:, None] < NQ) & (di[None, :] < ND))


def coarse_keys(q, d, distance, *, execution='fused', self_start=None):
    if distance not in ('hausdorff', 'dfd') or execution not in ('fused', 'reference'):
        raise ValueError('Expected H/DFD and fused/reference execution')
    nq, nd = len(q.coordinates), len(d.coordinates)
    if not 0 < nq * nd <= MAX_SCORE_PAIRS or nd >= 2**32:
        raise ValueError('Coarse score matrix exceeds the bounded query batch')
    if execution == 'reference':
        score = torch.cdist(q.coordinates[:, :64], d.coordinates[:, :64], p=float('inf'))
        if distance == 'dfd':
            for edge in (0, 1):
                a = q.points[q.offsets[:-1] if edge == 0 else q.offsets[1:] - 1]
                b = d.points[d.offsets[:-1] if edge == 0 else d.offsets[1:] - 1]
                score = torch.maximum(score, (a[:, None] - b[None]).square().sum(-1).sqrt())
        if self_start is not None:
            rows = torch.arange(nq, device=score.device)
            score[rows, rows + self_start] = torch.inf
        return lex_keys(score, torch.arange(nd, device=score.device)[None])
    if any(t.dtype != torch.float32 or t.stride(1) != 1 or not t.is_cuda
           for t in (q.coordinates, d.coordinates)):
        raise ValueError('Fused scoring requires CUDA FP32 contiguous feature columns')
    out = torch.empty((nq, nd), device=q.coordinates.device, dtype=torch.int64)
    score_keys[(triton.cdiv(nq, 8) * triton.cdiv(nd, 32),)](
        q.coordinates, d.coordinates, q.points, d.points, q.offsets, d.offsets,
        out, nq, nd, q.coordinates.stride(0), d.coordinates.stride(0), distance == 'dfd',
        self_start is not None, 0 if self_start is None else self_start,
        8, 32, num_warps=4, enable_fp_fusion=False)
    return out


def subset(index, lo, hi):
    summary = index.dfd_summary
    if summary is not None:
        summary = {'ids': {s: v[lo:hi] for s, v in summary['ids'].items()},
                   'block': summary['block']}
    return replace(index, coordinates=index.coordinates[lo:hi], offsets=index.offsets[lo:hi+1],
                   dfd_summary=summary)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _selected(keys, k):
    keys = keys.topk(k, largest=False).values
    return keys & 0xffffffff, (keys >> 32).int().contiguous().view(torch.float32)


@torch.no_grad()
def search(model, queries, database=None, *, mode='refine', top_k=1000,
           candidate_k=None, query_batch=None, execution='fused'):
    if not model.config.large_k:
        raise ValueError('Enable large_k with the matching SDD retrieval configuration')
    if mode not in ('ranking', 'refine') or execution not in ('fused', 'reference'):
        raise ValueError('Expected ranking/refine mode and fused/reference execution')
    top_k = _positive_int(top_k, 'top_k')
    same = database is None or database is queries
    database = queries if database is None else database
    model._check_index(queries)
    model._check_index(database)
    available = len(database.coordinates) - int(same)
    if top_k > available:
        raise ValueError('Database has fewer available trajectories than top_k')
    if candidate_k is None:
        candidate_k = max(6400, top_k) if mode == 'refine' else (
            max(2000, 2 * top_k) if model.config.distance == 'dfd' else top_k)
    candidate_k = min(_positive_int(candidate_k, 'candidate_k'), available)
    if candidate_k < top_k:
        raise ValueError('candidate_k must be at least top_k')
    query_batch = resolve_query_batch(model.config, query_batch, default=16,
        database_size=len(database.coordinates), max_score_pairs=MAX_SCORE_PAIRS)
    outputs, values = [], []
    for lo in range(0, len(queries.coordinates), query_batch):
        q = subset(queries, lo, min(lo + query_batch, len(queries.coordinates)))
        keys = coarse_keys(q, database, model.config.distance, execution=execution,
                           self_start=lo if same else None)
        ids, scores = _selected(keys, candidate_k)
        del keys
        if mode == 'refine' or model.config.distance == 'dfd':
            rows = torch.arange(len(ids), device=ids.device)[:, None].expand_as(ids).reshape(-1)
            flat = ids.reshape(-1).contiguous()
            if mode == 'refine':
                if model.config.distance == 'dfd':
                    from leap.dfd.dfd import dfd_pairs_bucketed
                    scores = dfd_pairs_bucketed(q, database, rows, flat)
                else:
                    from leap.common.raw_geometry_refinement import hausdorff_distances
                    scores = hausdorff_distances(q, database, rows, flat, point=True)
            else:
                from leap.dfd.dfd import dfd_pair_bounds
                from leap.dfd.dfd_geometry import witness_bound, ordered_bound
                scores = torch.maximum(dfd_pair_bounds(q, database, rows, flat), torch.maximum(
                    witness_bound(q, database, rows, flat, 8),
                    ordered_bound(q, database, rows, flat, 8)))
            if not torch.isfinite(scores).all():
                raise ValueError('Non-finite large-k scores; check input normalization')
            ids, scores = _selected(lex_keys(scores.reshape_as(ids), ids), top_k)
        else:
            ids, scores = ids[:, :top_k], scores[:, :top_k]
        outputs.append(ids)
        values.append(scores)
    return {'indices': torch.cat(outputs),
            'distances' if mode == 'refine' else 'scores': torch.cat(values),
            'mode': mode, 'candidate_k': candidate_k, 'full_search': False}
