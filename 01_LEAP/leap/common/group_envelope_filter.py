"""Level 1: Group Envelope Filter (GEF).

Paper labels: eq:node-bound, eq:dtw-node.
Node U stores (a_bar_U, b_under_U); ell_qU lower-bounds every member.
The Triton helper is inlined into traversal, not a separate GPU pass.
"""
import torch
import triton
import triton.language as tl


def node_envelope(z_T, valid):
    """Reduce member fields [..., members, 2M] to the node envelope [..., 2M]."""
    a_T, b_T = z_T.chunk(2, dim=-1)
    a_bar_U = a_T.masked_fill(~valid, -torch.inf).amax(-2)
    b_under_U = b_T.masked_fill(~valid, torch.inf).amin(-2)
    return torch.cat((a_bar_U, b_under_U), dim=-1)


def node_lower_bound(z_q, envelope_U):
    """eq:node-bound: ell(q,U) = max(0, max_j(b_q-a_bar_U, b_under_U-a_q))."""
    a_q, b_q = z_q.chunk(2, dim=-1)
    a_bar_U, b_under_U = envelope_U.chunk(2, dim=-1)
    ell_qU = torch.maximum(b_q - a_bar_U, b_under_U - a_q).amax(-1)
    return ell_qU.clamp_min(0)


@triton.jit
def node_bound_kernel(a_q, b_q, a_bar_U, b_under_U, AXIS: tl.constexpr):
    # eq:node-bound; callers apply floating-point slack at the threshold.
    ell_qU = tl.maximum(0., tl.max(tl.maximum(b_q - a_bar_U, b_under_U - a_q), AXIS))
    return ell_qU


@triton.jit
def dtw_node_bound_kernel(ell_qU, query_box_sum):
    # eq:dtw-node: ell_W = max(ell_Theta, sum_i dist(q_i, B(U))).
    return tl.maximum(query_box_sum, ell_qU)
