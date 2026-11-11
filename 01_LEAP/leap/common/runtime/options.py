"""Default execution fast paths; diagnostic switches do not change search rules."""
dfd_static = True
dfd_capacity = 2048
dfd_grouped_refinement = True
dtw_query_width = True
dtw_length_cache = True
dtw_subtree_parallel = True
dtw_single_layout = True
dtw_bucket_tiles = True
dtw_initial_cache = True
dtw_batch_frontier = True
# Batch gains were validated separately from complete single-request latency.
dtw_box_shortcircuit = True
dtw_single_box_shortcircuit = False
