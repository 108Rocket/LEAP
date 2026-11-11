"""Learned ordered-mass routing; a proposal score, never a pruning bound.

Paper eq:dtw-mass and eq:dtw-routing-summary, docs/dtw-candidate-learning.md.
Positive point masses define a monotone clock; eight windows cache 16 floats.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

DEFAULT_ASSET = Path(__file__).resolve().parents[1] / 'assets/porto_dtw_routing_e100.json'


def prepare(index, width=None):
    lengths = index.offsets[1:] - index.offsets[:-1]
    width = int(lengths.max()) if width is None else width
    j = torch.arange(width, device=index.points.device)[None]
    mask = j < lengths[:, None]
    points = index.points[(index.offsets[:-1, None] + j).clamp_max(len(index.points) - 1)]
    points = points.masked_fill(~mask[..., None], 0.)
    incoming = torch.zeros_like(points)
    incoming[:, 1:] = points[:, 1:] - points[:, :-1]
    incoming = incoming.masked_fill(~mask[..., None], 0.)
    outgoing = torch.zeros_like(points)
    outgoing[:, :-1] = incoming[:, 1:]
    li = torch.linalg.vector_norm(incoming, dim=-1)
    lo = torch.linalg.vector_norm(outgoing, dim=-1)
    total = li.sum(1, keepdim=True)
    mean = (total / (lengths[:, None] - 1)).clamp_min(1e-6)
    denom = (li * lo).clamp_min(1e-10)
    cosine = (incoming * outgoing).sum(-1) / denom
    sine = (incoming[..., 0] * outgoing[..., 1] - incoming[..., 1] * outgoing[..., 0]) / denom
    progress = j / (lengths[:, None] - 1)
    arc = li.cumsum(1) / total.clamp_min(1e-6)
    features = torch.stack([torch.log1p(li / mean), torch.log1p(lo / mean),
                            cosine, sine, progress, arc], -1).masked_fill(~mask[..., None], 0.)
    return points, lengths, features


class OrderedMass(torch.nn.Module):
    """f_psi(x_i): 6 geometry/progress features -> 16 hidden units -> log mass."""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, points, lengths, features, details=False):
        j = torch.arange(points.shape[1], device=points.device)[None]
        mask = j < lengths[:, None]
        # eq:dtw-mass: m_i = exp(1.5 tanh(f_psi(x_i)/1.5)), u_i = cumulative mass.
        log_mass = 1.5 * torch.tanh(self.net(features).squeeze(-1) / 1.5)
        mass = log_mass.exp() * mask
        right = mass.cumsum(1) / mass.sum(1, keepdim=True)
        left = torch.cat([torch.zeros_like(right[:, :1]), right[:, :-1]], 1)
        edges = torch.arange(9, device=points.device, dtype=points.dtype) / 8
        # eq:dtw-routing-summary: interval overlap alpha_ik transports original xy.
        overlap = (torch.minimum(right[..., None], edges[1:])
                   - torch.maximum(left[..., None], edges[:-1])).clamp_min(0)
        summary = 8 * torch.einsum('bnk,bnd->bkd', overlap, points)
        if details:
            return summary.flatten(1).contiguous(), log_mass.masked_fill(~mask, 0.), overlap
        return summary.flatten(1).contiguous()


def weights_hash(state):
    return hashlib.sha256(b''.join(np.asarray(state[k], dtype=np.float32).tobytes()
                                  for k in sorted(state))).hexdigest()


def load_model(mean, std, device, path=DEFAULT_ASSET):
    asset = json.loads(Path(path).read_text(encoding='utf-8'))
    if asset.get('format') != 'leap.dtw-routing.v1':
        raise ValueError('Expected a DTW routing asset')
    if (not np.array_equal(asset['geometry_mean'], mean)
            or not np.array_equal(asset['geometry_std'], std)):
        raise ValueError('DTW routing normalization differs; load matching geometry')
    state = asset['state_dict']
    if weights_hash(state) != asset['weights_sha256']:
        raise ValueError('DTW routing checksum mismatch')
    model = OrderedMass().to(device)
    expected = model.state_dict()
    if (set(state) != set(expected) or any(np.shape(state[k]) != tuple(expected[k].shape)
            or not np.isfinite(state[k]).all() for k in state)):
        raise ValueError('Malformed DTW routing parameters')
    model.load_state_dict({k: torch.tensor(v, device=device) for k, v in state.items()})
    return model.eval().requires_grad_(False), asset['weights_sha256']


def signature(model):
    return tuple((p.data_ptr(), p._version) for p in model.parameters())


@torch.no_grad()
def encode_index(model, index, batch=512):
    if (index.points.is_cuda and index.points.dtype == torch.float32
            and 64 <= len(index.offsets) - 1 <= 512 and batch == 512
            and type(model) is OrderedMass
            and not torch.backends.cuda.matmul.allow_tf32
            and all(p.dtype == torch.float32 and p.is_contiguous()
                    and p.device == index.points.device for p in model.parameters())):
        from leap.common import _query_encoding
        with torch.cuda.device(index.points.device):
            if _query_encoding.supported(index):
                return _query_encoding.routing(model, index)
    if (index.points.is_cuda and index.points.dtype == torch.float32
            and len(index.offsets) - 1 > 512 and batch == 512
            and type(model) is OrderedMass
            and not torch.backends.cuda.matmul.allow_tf32
            and all(p.dtype == torch.float32 and p.is_contiguous()
                    for p in model.parameters())):
        from leap.dtw._routing_sparse import encode
        with torch.cuda.device(index.points.device):
            return encode(model, index, batch)
    return _encode_reference(model, index, batch)


@torch.no_grad()
def _encode_reference(model, index, batch=512):
    width = int((index.offsets[1:] - index.offsets[:-1]).max())
    summaries = []
    for lo in range(0, len(index.offsets) - 1, batch):
        sub = SimpleNamespace(points=index.points, offsets=index.offsets[lo:lo+batch+1])
        summaries.append(model(*prepare(sub, width)))
    return torch.cat(summaries).contiguous()


def pair_scores(q_summary, d_summary, q_length, d_length):
    """eq:dtw-seed-scan: max(n,m)/8 * sum_k ||mu_k(A)-mu_k(B)||."""
    q = q_summary.reshape(*q_summary.shape[:-1], 8, 2)
    d = d_summary.reshape(*d_summary.shape[:-1], 8, 2)
    return torch.linalg.vector_norm(q-d, dim=-1).mean(-1) * torch.maximum(q_length, d_length)


def coverage_loss(positive, negative_boundary, radius):
    """eq:dtw-routing-loss; Top20 in128 uses the 109th false candidate."""
    scale = radius.clamp_min(1e-5)
    return ((positive - negative_boundary[:, None]) / scale[:, None] + .02).clamp_min(0).mean()
