"""Single database-row search for the viewer, without the batch API's k<=64 cap.

Weights and distance definitions are unchanged. Larger requested sets widen the
initial pool; they are search settings, not the paper's fixed-budget protocol.
Ground-truth matrices are never accessible from this module.
"""
import time
import torch

from leap.common.model import TrajectoryIndex


def query_row(database, row, distance):
    if database.ranking_index is not None:
        from leap.common.precision import subset
        return subset(database, row, row + 1)
    if distance == 'dfd':
        from leap.dfd.dfd_geometry import subset
        return subset(database, row, row + 1)
    query = TrajectoryIndex(database.coordinates[row:row + 1], database.points,
                            database.offsets[row:row + 2], database.asset_id,
                            None if database.ordered_summary is None else
                            database.ordered_summary[row:row + 1])
    if distance == 'dtw':
        from dataclasses import replace
        from leap.dtw.dtw_probes import slice_summary
        query = replace(query, dtw_probe_summary=slice_summary(database.dtw_probe_summary, row, row + 1))
    return query


def _top_ids(scores, ids, k):
    from leap.common.interval_index import lex_keys
    return (lex_keys(scores, ids).topk(k, largest=False).values & 0xffffffff).contiguous()


@torch.no_grad()
def initial_candidates(model, query, database, tree, qids, count):
    distance = model.config.distance
    if distance == 'dtw':
        ids = model._dtw_candidates(query, database, count, qids,
                                   routing_tree=getattr(tree, 'dtw_routing_tree', None))
        return ids, max(model.config.dtw_pool, count), 'routing tree' if count <= 128 else 'wide routing scan'
    if distance == 'hausdorff':
        if model.config.hausdorff_fields == 'near_far':
            from leap.hausdorff.pointset_index import large_candidates
            ids = (tree.candidates(query, count, qids=qids)[0] if count <= 64 else
                   large_candidates(query, database, count, qids=qids))
            return ids, count, 'near/far tree' if count <= 64 else 'wide near/far scan'
        if count <= 64:
            return tree.candidates(query.coordinates, qids, count)[0], count, 'field tree'
        from leap.common.interval_index import reference
        return (reference(query.coordinates, database.coordinates, qids, count) & 0xffffffff), count, 'wide field scan'
    if count <= 64:
        return model._dfd_candidates(query, database, tree, qids, count), 64, 'vertex tree + ordered8'
    # Same DFD light score; widen its routing pool for requested N>64.
    from leap.dfd.dfd_tree import candidate_pool, endpoints
    from leap.dfd.dfd_geometry import witness_bound, ordered_bound
    width = min(len(database.coordinates) - 1, max(64, 2 * count))
    if width <= 256:
        ids, lower, _ = candidate_pool(query, database, tree, qids, width)
    else:
        da = database.coordinates[:, :64]
        lower = (query.coordinates[:, None, :64] - da[None]).abs().amax(-1)
        eq, ed = endpoints(query).reshape(-1, 2, 2), tree.dfd_geometry['endpoints'].reshape(-1, 2, 2)
        for end in range(2):
            lower = torch.maximum(lower, (eq[:, end, None] - ed[None, :, end]).square().sum(-1).sqrt())
        lower[0, qids[0]] = torch.inf
        ids = _top_ids(lower, torch.arange(len(da), device=da.device)[None], width)
        lower = lower.gather(1, ids)
    rows = torch.zeros(ids.numel(), device=ids.device, dtype=torch.int64)
    score = torch.maximum(lower, witness_bound(query, database, rows, ids.flatten(), 8).reshape_as(lower))
    score = torch.maximum(score, ordered_bound(query, database, rows, ids.flatten(), 8).reshape_as(lower))
    return _top_ids(score, ids, count), width, 'wide vertex pool + ordered8'


@torch.no_grad()
def raw_scores(query, database, ids, distance, *, pointset=False):
    from leap.common.raw_geometry_refinement import hausdorff_distances, dtw_distances
    from leap.dfd.dfd import dfd_pairs as dfd_distances
    rows = torch.zeros(ids.numel(), device=ids.device, dtype=torch.int64)
    if distance == 'hausdorff' and pointset:
        return hausdorff_distances(query, database, rows, ids.flatten(), point=True).reshape_as(ids)
    fn = {'hausdorff': hausdorff_distances, 'dfd': dfd_distances, 'dtw': dtw_distances}[distance]
    return fn(query, database, rows, ids.flatten()).reshape_as(ids)


@torch.no_grad()
def search_row(model, database, tree, row, top_n=20, mode='complete', pool_size=50):
    """Return N unique, self-excluded IDs. mode: initial, pool, or complete.

    Complete search uses the Nth original distance in S(q) as r_hat_q
    (paper eq:search-radius); GEF/TBF then consider the entire database.
    It is not a rerank restricted to S(q). Existing empirical FP32 slack remains.
    """
    size = len(database.coordinates)
    if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < size:
        raise ValueError('Query row is outside the database')
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= min(1000, size - 1):
        raise ValueError('Top-N must be between 1 and min(1000, database size - 1)')
    if mode not in ('initial', 'pool', 'complete'):
        raise ValueError('Unknown search mode')
    if mode == 'pool' and (not isinstance(pool_size, int) or isinstance(pool_size, bool)
                           or not top_n <= pool_size <= min(5000, size - 1)):
        raise ValueError('Refinement pool must be >= N and <= min(5000, database size - 1)')
    model._check_index(database)
    tree.validate(database)
    if model.config.precision == 'fp64':
        from leap.common.precision import search_row as precise_row
        return precise_row(model, database, tree, row, top_n, mode, pool_size)
    metric = model.config.distance
    default = {'hausdorff': 50, 'dfd': 32, 'dtw': 20}[metric]
    count = top_n if mode == 'initial' else pool_size if mode == 'pool' else max(default, top_n)
    count = min(count, size - 1)
    query = query_row(database, row, metric)
    qids = torch.tensor([row], device=model.device, dtype=torch.int64)
    torch.cuda.synchronize()
    start = time.perf_counter()
    initial, routing_width, route = initial_candidates(model, query, database, tree, qids, count)
    torch.cuda.synchronize()
    coarse_seconds = time.perf_counter() - start
    detail = {}
    if mode == 'initial':
        result = initial[:, :top_n]
        distances = None
        refined = 0
    elif mode == 'pool':
        distances_all = raw_scores(query, database, initial, metric,
                                   pointset=model.config.hausdorff_fields == 'near_far')
        result = _top_ids(distances_all, initial, top_n)
        # The returned IDs are ordered by original distance; avoid rescoring.
        selected = (result[:, :, None] == initial[:, None, :]).long().argmax(-1)
        distances = distances_all.gather(1, selected)
        refined = count
    else:
        if metric == 'hausdorff':
            if model.config.hausdorff_fields == 'near_far':
                from leap.hausdorff.pointset_index import full
                detail = full(query, database, tree, seed_k=count, top_k=top_n,
                              initial=initial, qids=qids)
            else:
                from leap.hausdorff.range_search import search_chunk
                detail = search_chunk(query, database, tree, qids, top_k=top_n, initial=initial)
        elif metric == 'dfd':
            from leap.dfd.dfd_range import search
            detail = search(query, database, tree, qids, initial, top_n)
        else:
            from leap.dtw.dtw_range import search
            a, b = query.dtw_probe_summary, database.dtw_probe_summary
            geometry = {'box': tree.dtw_geometry['box'][row:row + 1], 'block': tree.dtw_geometry['block']}
            detail = search(query, database, tree, tree.dtw_geometry, geometry, qids, initial, top_n,
                            probe_fields=(a['radial'], b['radial']), probe_tree=tree.dtw_probe_tree,
                            ordered_fields=(a['ordered'], b['ordered']), bounded_dtw=True,
                            bucket_dtw=True, one_pass_capacity=2048,
                            parallel_leaves=model.config.dtw_execution=='optimized')
        result, distances = detail['indices'], detail['distances']
        refined = int(detail['counts'][0])
    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - start
    values = result[0].cpu().tolist()
    if len(values) != top_n or len(set(values)) != top_n or row in values:
        raise RuntimeError('Retrieval violated count, uniqueness or self-exclusion')
    return dict(ids=values, scores=None if distances is None else distances[0].cpu().tolist(),
                coarse_seconds=coarse_seconds, search_seconds=total_seconds,
                refinement_seconds=total_seconds - coarse_seconds,
                initial_count=count, routing_pool=routing_width, route=route,
                survivor_count=refined, scope='database' if mode == 'complete' else mode,
                radius=None if 'radius' not in detail else float(detail['radius'][0]),
                asset_id=model.asset_id,
                numerical_policy='FP32 original-distance kernels; existing empirical pruning slack')
