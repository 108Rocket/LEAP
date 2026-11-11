"""Analytical starting budgets for Full retrieval, never a recall predictor."""
from copy import copy
from dataclasses import dataclass, replace

from leap.common.config import RetrievalConfig, _positive
from leap.common.runtime.batching import resolve_query_batch
from leap.common.candidate_policy import tree_initializer

MAX_SCORE_PAIRS = 26_000_000


@dataclass(frozen=True)
class FullSearchPlan:
    config: RetrievalConfig
    top_k: int
    database_size: int
    available: int
    path: str
    profile_matched: bool

    def as_dict(self):
        return dict(policy='analytical-full-v1', top_k=self.top_k,
                    database_size=self.database_size, available=self.available,
                    initial_k=self.config.initial_k, initial_pool=self.config.initial_pool,
                    query_batch=self.config.query_batch, path=self.path,
                    profile_matched=self.profile_matched, full_database=True,
                    precision=self.config.precision, recall_guarantee=False,
                    search_engine=self.config.search_engine,
                    dynamic_threshold=self.config.dynamic_threshold,
                    benchmarks_run=0)


def _first(*values):
    return next((v for v in values if v is not None), None)


def resolve(config, database_size, top_k=20, *, query_count=None,
            exclude_self=False, initial_k=None, query_batch=None):
    if config.parameter_mode != 'auto':
        raise ValueError('resolve requires parameter_mode=auto')
    _positive(database_size, 'database_size')
    _positive(top_k, 'top_k')
    if not isinstance(exclude_self, bool):
        raise ValueError('exclude_self must be boolean')
    available = database_size - int(exclude_self)
    if top_k > min(1000, available):
        raise ValueError('Require top_k <= min(1000, available database rows)')
    if query_count is not None:
        _positive(query_count, 'query_count')
    profile = next((p for p in config.top_k_profiles if p.top_k == top_k), None)
    option = lambda name: getattr(profile, name, None)
    seed = _first(initial_k, option('initial_k'), config.initial_k)
    if seed is None:
        floor = dict(hausdorff=50, dfd=32 if config.dfd_initial == 'light' else 50, dtw=20)[config.distance]
        seed = min(available, max(top_k, floor) if top_k <= 64 else top_k)
    _positive(seed, 'initial_k')
    if not top_k <= seed <= available:
        raise ValueError('Require top_k <= initial_k <= available database rows')
    wide = seed > 64 or config.full_large_k or config.initial_search == 'full'
    pool = _first(option('initial_pool'), config.initial_pool)
    if config.distance != 'hausdorff':
        if pool is None:
            if wide:
                # Ratios reproduce the native Top1000 budgets (2000 / 6400).
                pool = 2 * seed if config.distance == 'dfd' else (32 * seed + 4) // 5
            elif config.distance == 'dfd':
                pool = max(seed, 64 if config.dfd_initial == 'light' else 200)
            else:
                pool = max(seed, config.dtw_pool)
        _positive(pool, 'initial_pool')
        if pool < seed:
            raise ValueError('initial_pool must cover initial_k')
        pool = min(available, pool)
        # The wide Full engine also accepts tree initialization with larger pools.
        wide = wide or pool > (256 if config.distance == 'dfd' else 128)
    # Point-set vertex-only search has no ordinary polyline Full path.
    wide = wide or (config.hausdorff_target == 'pointset' and config.hausdorff_fields != 'near_far')
    batch = _first(query_batch, option('query_batch'), config.query_batch)
    if batch is None and config.query_profile == 'single_query':
        batch = 1
    dense = wide and not tree_initializer(config,seed if pool is None else pool)
    options = dict(default=128, database_size=database_size,
                   max_score_pairs=MAX_SCORE_PAIRS) if dense else {}
    batch = resolve_query_batch(config, batch, **options)
    if query_count is not None:
        batch = min(batch, query_count)
    resolved = replace(config, parameter_mode='manual', top_k_profiles=(),
                       initial_k=seed, initial_pool=pool, full_large_k=wide,
                       dtw_pool=pool if config.distance == 'dtw' else config.dtw_pool,
                       query_batch=batch)
    path = ('near_far_dense' if dense else 'near_far_tree') if config.hausdorff_fields == 'near_far' else (
        'dense_full' if dense else 'tree_full')
    return FullSearchPlan(resolved, top_k, database_size, available, path, profile is not None)


def configured_model(model, plan):
    """Share immutable weights/index storage without mutating the caller's model."""
    configured = copy(model)
    configured.config = plan.config
    return configured


def search(model, queries, database=None, *, search_index=None, seed_k=None,
           top_k=20, query_batch=None, initial_k=None, **options):
    if initial_k is not None and seed_k is not None and initial_k != seed_k:
        raise ValueError('initial_k and seed_k specify different candidate counts')
    same = database is None or database is queries
    db = queries if database is None else database
    plan = model.config.resolve(len(db.coordinates), top_k,
        query_count=len(queries.coordinates), exclude_self=same,
        initial_k=initial_k if initial_k is not None else seed_k, query_batch=query_batch)
    configured = configured_model(model, plan)
    result = configured.search_range(queries, database, search_index=search_index,
        top_k=top_k, seed_k=plan.config.initial_k, query_batch=plan.config.query_batch, **options)
    result['configuration'] = plan.as_dict()
    return result
