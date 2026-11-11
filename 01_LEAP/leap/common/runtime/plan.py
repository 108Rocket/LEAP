"""Reusable single-request execution plan for DFD and signal DTW."""

from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

import torch
import triton

from leap.common.model import TrajectoryIndex
from leap.common.single_kernels import pair_bucket, signal_bucket, leaf_bucket, dfd_rank_bucket, endpoints_kernel, decode_sort

MIN_BUCKET_WIDTH = 64
DP_WORK_ELEMENTS = 32 * 1024 * 1024
DP_WORK_ROWS = 3
MAX_ENCODING_GRAPHS = 128
CACHED_DFD_KIND = 3


@lru_cache(maxsize=16)
def _length_buckets(block):
    """Immutable (exclusive lower, inclusive upper) bounds for kernel dispatch."""
    widths = (2**i for i in range(6, max(MIN_BUCKET_WIDTH, block).bit_length()))
    lower = 0
    buckets = []
    for width in widths:
        buckets.append((lower, width))
        lower = width
    return tuple(buckets)


def _workspace_chunk(pair_count, block):
    """Keep the existing three-row DP workspace budget, measured in elements."""
    return max(1, min(pair_count, DP_WORK_ELEMENTS // (DP_WORK_ROWS * block)))


class SingleQueryPlan:
    """Session-owned scratch storage and length-specific encoding graphs.

    Scratch views are valid only until the next operation using the same name.
    The session serializes access and copies public results before returning.
    """

    def __init__(self, model, database, tree):
        from leap.common.runtime import options as opt
        self.model, self.database, self.tree = model, database, tree
        self.dtw_box_shortcircuit = opt.dtw_single_box_shortcircuit
        self.device = model.device
        lengths = database.offsets[1:] - database.offsets[:-1]
        self.database_block = triton.next_power_of_2(int(lengths.max()))
        self.leaf_widths = None
        if model.config.distance == 'dtw':
            members = tree.members.reshape(-1, tree.block)
            self.leaf_widths = torch.where(members >= 0, lengths[members.clamp_min(0)], 0).amax(1)
        self.buffers = {}
        self.graphs = {}
        self.capture_enabled = False
        self.query_length = 0
        self.stats = dict(
            graph_builds=0,
            graph_replays=0,
            graph_fallbacks=0,
            eager_requests=0,
            buffer_growths=0,
            max_geometry_reductions=1,
        )

    def storage(self, name, shape, dtype=torch.float32):
        """Return a reusable view; grow capacity without caching query values."""
        size = 1
        for dimension in shape:
            size *= dimension
        key = (name, dtype)
        value = self.buffers.get(key)
        if value is None or value.numel() < size:
            if value is not None:
                getattr(self, 'graphs', {}).clear()
            value = torch.empty(
                max(1, triton.next_power_of_2(size)), dtype=dtype, device=self.device
            )
            self.buffers[key] = value
            self.stats['buffer_growths'] += 1
        return value[:size].view(shape)

    def endpoint(self, query):
        out = self.storage('endpoints', (1, 4))
        endpoints_kernel[(1,)](query.points, query.offsets, out)
        return out

    def dfd_summary(self, index):
        from leap.dfd.dfd_geometry import simplify_kernel

        block = max(32, triton.next_power_of_2(self.query_length))
        ids = {s: self.storage(f'retained_ids_{s}', (1, s), torch.int32) for s in (8, 16, 32)}
        coords = {s: self.storage(f'retained_coords_{s}', (1, s, 2)) for s in (8, 16, 32)}
        simplify_kernel[(1,)](
            index.points,
            index.offsets,
            *ids.values(),
            *coords.values(),
            block,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return dict(ids=ids, block=triton.next_power_of_2(self.query_length))

    def dfd_candidates(self, queries, index, tree, qids, k, pool, strategy):
        from leap.dfd.dfd_tree import candidate_kernel

        if strategy != 'light' or pool not in (None, 64):
            return None
        pool = min(64, len(index.coordinates))
        if not 1 <= k <= pool:
            raise ValueError('Single-query DFD candidate budget out of range')
        keys = self.storage('candidate_tree_keys', (1, pool), torch.int64)
        visits = self.storage('candidate_tree_visits', (1, 2), torch.int32)
        candidate_kernel[(1,)](
            queries.coordinates,
            index.coordinates,
            self.endpoint(queries),
            tree.dfd_geometry['endpoints'],
            qids,
            tree.members,
            *tree.dfd_geometry['native'],
            keys,
            visits,
            tree.database_scale,
            tree.groups,
            tree.block,
            pool,
            64,
            FAST=self.model.config.search_engine == 'fast',
            num_warps=4,
            enable_fp_fusion=False,
        )
        ids = keys & 0xFFFFFFFF
        low = (keys >> 32).int().contiguous().view(torch.float32)
        scores = self.storage('dfd_rank_keys', (pool,), torch.int64)
        block = max(self.database_block, queries.dfd_summary['block'])
        for lower, width in _length_buckets(block):
            dfd_rank_bucket[(pool,)](
                queries.points,
                queries.offsets,
                index.points,
                index.offsets,
                ids,
                low,
                queries.dfd_summary['ids'][8],
                index.dfd_summary['ids'][8],
                scores,
                width,
                lower,
                num_warps=4,
                enable_fp_fusion=False,
            )
        out = self.storage('dfd_rank_out', (1, k), torch.int64)
        if pool == triton.next_power_of_2(pool):
            decode_sort[(1,)](scores, out, pool, k)
        else:
            out.copy_(scores.topk(k, largest=False).values.reshape(1, k) & 0xFFFFFFFF)
        return out

    @staticmethod
    def widths(block):
        return [width for _, width in _length_buckets(block)]

    def pair_values(
        self, query, database, qids, dids, kind, limits=None, initial=None, raw=None, check=8
    ):
        """Score pairs with the unchanged kernel ABI: 0 DFD, 1 DTW, 2 bounded
        DTW, 3 cached DFD. Cached DFD limits are per query; other limits are per
        pair. Returned scores and flags alias this plan's scratch storage.
        """
        qids, dids = qids.contiguous(), dids.contiguous()
        n = len(dids)
        out = self.storage(f'pair_out_{kind}', (n,))
        flags = self.storage(f'pair_flags_{kind}', (n,), torch.int32)
        cached_dfd = kind == CACHED_DFD_KIND
        if n == 0:
            return (out, flags) if cached_dfd else out
        block = max(
            MIN_BUCKET_WIDTH,
            self.database_block,
            triton.next_power_of_2(self.query_length) if cached_dfd else 1,
        )
        chunk = _workspace_chunk(n, block)
        work = self.storage('pair_work', (chunk, DP_WORK_ROWS, block))
        buckets = _length_buckets(block)
        initial_scores = out if initial is None else initial
        raw_scores = out if raw is None else raw
        initial_width = 0 if initial is None else initial.shape[1]
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            # All buckets visit the same chunk. Construct its tensor views once.
            chunk_qids, chunk_dids = qids[start:end], dids[start:end]
            chunk_out, chunk_flags = out[start:end], flags[start:end]
            chunk_limits = limits if cached_dfd else (out if limits is None else limits[start:end])
            for lower, width in buckets:
                pair_bucket[(end - start,)](
                    query.points,
                    query.offsets,
                    database.points,
                    database.offsets,
                    chunk_qids,
                    chunk_dids,
                    chunk_out,
                    chunk_flags,
                    work,
                    chunk_limits,
                    initial_scores,
                    raw_scores,
                    width,
                    lower,
                    kind,
                    initial_width,
                    check,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return (out, flags) if cached_dfd else out

    def signal_values(self, q, db, qids, dids, fields):
        qids, dids = qids.contiguous(), dids.contiguous()
        query_signals, database_signals = fields
        n = len(dids)
        out = self.storage('signal_scores', (n,))
        if n == 0:
            return out
        block = max(MIN_BUCKET_WIDTH, self.database_block)
        chunk = _workspace_chunk(n, block)
        work = self.storage('signal_work', (chunk, DP_WORK_ROWS, block))
        buckets = _length_buckets(block)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            chunk_qids, chunk_dids, chunk_out = qids[start:end], dids[start:end], out[start:end]
            for lower, width in buckets:
                signal_bucket[(end - start,)](
                    query_signals,
                    q.offsets,
                    database_signals,
                    db.offsets,
                    chunk_qids,
                    chunk_dids,
                    chunk_out,
                    work,
                    query_signals.shape[1],
                    width,
                    lower,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return out

    def check_leaves(
        self,
        p,
        off,
        qp,
        qo,
        members,
        boxes,
        qb,
        radius,
        slack,
        self_ids,
        seeds,
        lids,
        lrows,
        keep,
        examined,
        b,
        block,
        s,
        tile,
    ):
        from leap.dtw.dtw_parallel import bucket_tile
        for lower, width in _length_buckets(block):
            bucket_width = bucket_tile(width, b, tile)
            leaf_bucket[(len(lids), triton.cdiv(b, bucket_width))](
                p,
                off,
                qp,
                qo,
                members,
                boxes,
                qb,
                radius,
                slack,
                self_ids,
                seeds,
                lids,
                lrows,
                keep,
                examined,
                self.leaf_widths,
                b,
                width,
                lower,
                s,
                bucket_width,
                self.dtw_box_shortcircuit,
                num_warps=8,
                enable_fp_fusion=False,
            )

    def encode_dtw(self, fields, points, offsets):
        """Capture exact-shape encoding only; query values are never cached."""
        from leap.dtw.dtw_routing import prepare
        from leap.dtw.dtw_signal import encode as signal_encode
        from leap.dtw.dtw_radial import encode_kernel as radial_kernel
        from leap.dtw.dtw_probe_intervals import encode_kernel as ordered_kernel
        from leap.dtw.ordered_dtw import signature
        from leap.dtw.dtw_probes import signature as probe_signature

        model = self.model
        length = self.query_length
        key = (length, points.data_ptr(), offsets.data_ptr())

        def operations():
            obj = SimpleNamespace(points=points, offsets=offsets)
            summary = model.dtw_router(*prepare(obj, length))
            signal = signal_encode(obj, model.dtw_probes['signal'])
            radial = torch.empty((1, 16, 6), device=self.device)
            ordered = torch.empty((1, 4, 16, 2), device=self.device)
            block = triton.next_power_of_2(length)
            radial_kernel[(1, 16)](
                points,
                offsets,
                model.dtw_probes['radial'],
                radial,
                16,
                4,
                block,
                num_warps=4,
                enable_fp_fusion=False,
            )
            ordered_kernel[(1, 16)](
                points,
                offsets,
                model.dtw_probes['ordered'],
                ordered,
                16,
                4,
                block,
                num_warps=4,
                enable_fp_fusion=False,
            )
            finite = torch.stack(
                [torch.isfinite(t).all() for t in (summary, signal, radial, ordered)]
            )
            return summary, dict(signal=signal, radial=radial, ordered=ordered), finite

        if key not in self.graphs:
            # Bound captured memory for workloads with many different lengths.
            if not self.capture_enabled or len(self.graphs) >= MAX_ENCODING_GRAPHS:
                self.stats['eager_requests'] += 1
                if len(self.graphs) >= MAX_ENCODING_GRAPHS:
                    self.stats['graph_fallbacks'] += 1
                summary, cache, finite = operations()
            else:
                operations()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=torch.cuda.current_stream(self.device)):
                    outputs = operations()
                self.graphs[key] = graph, outputs
                self.stats['graph_builds'] += 1
                graph.replay()
                summary, cache, finite = outputs
        else:
            graph, outputs = self.graphs[key]
            graph.replay()
            self.stats['graph_replays'] += 1
            summary, cache, finite = outputs
        if not bool(finite.all()):
            raise ValueError('DTW single-query encoding overflowed')
        index = TrajectoryIndex(
            fields, points, offsets, model.asset_id, summary, signature(points, offsets, summary)
        )
        return replace(
            index,
            dtw_probe_summary=cache,
            dtw_probe_signature=probe_signature(index, cache, model.dtw_probes),
        )
