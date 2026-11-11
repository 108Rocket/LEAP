"""Level-wise CUDA construction preserving stable median partition semantics."""
import math

import torch


@torch.no_grad()
def median_members(x, block=64):
    if x.ndim != 2 or not len(x) or not x.is_cuda:
        raise ValueError('Expected a nonempty CUDA feature matrix')
    if block < 1:
        raise ValueError('Leaf capacity must be positive')
    n, dim = x.shape
    depth = max(0, math.ceil(math.log2(max(1, n / block))))
    ids = torch.arange(n, device=x.device).reshape(1, n)
    sizes = torch.tensor([n], device=x.device)
    width = n
    for _ in range(depth):
        # Padding is excluded from both range selection and stable sorting.
        valid = ids >= 0
        values = x[ids.clamp_min(0)]
        high = values.masked_fill(~valid[..., None], -torch.inf).amax(1)
        low = values.masked_fill(~valid[..., None], torch.inf).amin(1)
        axis = (high - low).argmax(1)
        del values, high, low
        keys = x[ids.clamp_min(0), axis[:, None]].masked_fill(~valid, torch.inf)
        order = torch.argsort(keys, dim=1, stable=True)
        ordered = ids.gather(1, order)
        half = sizes // 2
        next_sizes = torch.stack((half, sizes - half), 1).reshape(-1)
        next_width = (width + 1) // 2
        lane = torch.arange(next_width, device=x.device)
        positions = torch.stack((torch.zeros_like(half), half), 1)[..., None] + lane
        children = ordered.gather(1, positions.reshape(len(ids), -1).clamp_max(width - 1))
        children = children.reshape(-1, next_width)
        ids = children.masked_fill(lane >= next_sizes[:, None], -1)
        sizes, width = next_sizes, next_width
    out = torch.full((len(ids), block), -1, dtype=torch.int64, device=x.device)
    out[:, :width] = ids
    return out


@torch.no_grad()
def envelopes(values, members, *, upper_first=True):
    """Same leaf reductions and binary topology, without per-node CPU loops."""
    groups, dim = len(members), values.shape[1]
    packed = values[members.clamp_min(0)]
    valid = (members >= 0)[..., None]
    high = packed.masked_fill(~valid, -torch.inf).amax(1)
    low = packed.masked_fill(~valid, torch.inf).amin(1)
    out = torch.zeros((2 * groups, 2 * dim), dtype=values.dtype, device=values.device)
    out[groups:] = torch.cat((high, low) if upper_first else (low, high), 1)
    level = groups // 2
    first = torch.maximum if upper_first else torch.minimum
    second = torch.minimum if upper_first else torch.maximum
    while level:
        out[level:2*level, :dim] = first(out[2*level:4*level:2, :dim], out[2*level+1:4*level:2, :dim])
        out[level:2*level, dim:] = second(out[2*level:4*level:2, dim:], out[2*level+1:4*level:2, dim:])
        level //= 2
    return out
