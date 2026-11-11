"""Candidate workspace limits, independent of leaf size and result count."""

MAX_TREE_CANDIDATES = 8192


def tree_initializer(config, pool, *, has_index=True):
    supported = (has_index and config.precision == 'fp32'
                 and 1 <= pool <= MAX_TREE_CANDIDATES)
    if config.distance == 'dtw':
        supported = supported and config.dtw_seed == 'signal' and config.dtw_execution == 'optimized'
    mode = config.initial_search
    if mode == 'tree' and not supported:
        raise ValueError('Requested tree initializer needs a compatible index and pool <= 8192')
    return mode != 'full' and supported
