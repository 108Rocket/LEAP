"""Differentiable finite-pool summary selection followed by frozen signal selection."""
import torch


def first_selection(scores, valid, k=128, temperature=.1):
    if temperature <= 0 or not bool((valid.sum(1) > k).all()):
        raise ValueError('Soft selection needs positive temperature and more than k valid entries')
    with torch.no_grad():
        detached = scores.detach()
        lo = detached.masked_fill(~valid, torch.inf).amin(1) - 40 * temperature
        hi = detached.masked_fill(~valid, -torch.inf).amax(1) + 40 * temperature
        for _ in range(48):
            b = (lo + hi) / 2
            count = (torch.sigmoid((b[:, None] - detached) / temperature) * valid).sum(1)
            lo, hi = torch.where(count < k, b, lo), torch.where(count >= k, b, hi)
        b = (lo + hi) / 2
        p = torch.sigmoid((b[:, None] - detached) / temperature) * valid
        w = p * (1 - p)
        denominator = w.sum(1)
        if not bool((denominator > 1e-10).all()):
            raise RuntimeError('First soft selection saturated')
    mean = (w * scores).sum(1) / denominator
    b = b + mean - mean.detach()
    return torch.sigmoid((b[:, None] - scores) / temperature) * valid


def conditional_selection(p, signal, k=20, temperature=.1):
    if signal.requires_grad:
        raise ValueError('Conditional selection requires frozen signal scores')
    with torch.no_grad():
        weights = p.detach()
        lo, hi = signal.amin(1) - 40 * temperature, signal.amax(1) + 40 * temperature
        for _ in range(48):
            c = (lo + hi) / 2
            count = (weights * torch.sigmoid((c[:, None] - signal) / temperature)).sum(1)
            lo, hi = torch.where(count < k, c, lo), torch.where(count >= k, c, hi)
        c = (lo + hi) / 2
        g = torch.sigmoid((c[:, None] - signal) / temperature)
        denominator = (weights * g * (1 - g)).sum(1)
        if not bool((denominator > 1e-10).all()):
            raise RuntimeError('Conditional soft selection saturated')
    mass = (p * g).sum(1)
    c = c - temperature * (mass - mass.detach()) / denominator
    return p * torch.sigmoid((c[:, None] - signal) / temperature)


def tail_value(a, distances):
    descending, order = distances.sort(dim=1, descending=True, stable=True)
    mass = a.gather(1, order)
    at = (mass.cumsum(1) < 1).sum(1).clamp_max(a.shape[1] - 1)
    z = descending.gather(1, at[:, None]).squeeze(1).detach()
    return z + (a * (distances - z[:, None]).clamp_min(0)).sum(1)


def aligned_radius_loss(scores, signal, distances, valid, radius):
    scale = radius.clamp_min(1e-5)[:, None]
    p = first_selection(scores / scale, valid)
    a = conditional_selection(p, signal / scale)
    with torch.no_grad():
        if not bool((p.sum(1) - 128).abs().lt(2e-3).all()):
            raise RuntimeError('First selection cardinality error')
        if not bool((a.sum(1) - 20).abs().lt(2e-3).all()):
            raise RuntimeError('Second selection cardinality error')
    return tail_value(a, distances / scale).mean()
