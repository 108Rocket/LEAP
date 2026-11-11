"""Shared query batching policy; batching never changes retrieval budgets."""


def positive_batch(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('query_batch must be a positive integer')
    return value


def resolve_query_batch(config, override=None, *, default=None,
                        database_size=None, max_score_pairs=None):
    """Resolve call > config > profile defaults, then cap dense workspaces."""
    value = override if override is not None else getattr(config, 'query_batch', None)
    if value is None:
        if default is not None:
            value = default
        elif config.distance == 'dtw':
            value = 2000 if (config.dtw_execution == 'optimized' or
                             config.dtw_seed != 'signal') else 64
        else:
            value = 512
    value = positive_batch(value)
    if (database_size is None) != (max_score_pairs is None):
        raise ValueError('Dense batching needs database_size and max_score_pairs')
    if database_size is not None:
        if database_size < 1 or max_score_pairs < database_size:
            raise ValueError('Database exceeds bounded coarse-scoring capacity')
        value = min(value, max_score_pairs // database_size)
    return value
