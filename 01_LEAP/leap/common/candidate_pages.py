"""Exact score/ID pages with bounded per-kernel sorted state."""
import torch
import triton

PAGE_CAPACITY = 1024


def collect(n, k, device, leaf_size, visit_columns, launch):
    pages = []
    total_visits = None
    after = None
    for start in range(0, k, PAGE_CAPACITY):
        count = min(PAGE_CAPACITY, k-start)
        keys = torch.empty((n, count), device=device, dtype=torch.int64)
        visits = torch.empty((n, visit_columns), device=device, dtype=torch.int32)
        width = max(leaf_size, triton.next_power_of_2(count))
        launch(count, width, after, keys, visits)
        pages.append(keys)
        total_visits = visits if total_visits is None else total_visits + visits
        # Strict score/ID ordering excludes previous pages, including tied scores.
        after = keys[:, -1].contiguous() if start+count < k else None
    return (pages[0] if len(pages)==1 else torch.cat(pages, dim=1)), total_visits
