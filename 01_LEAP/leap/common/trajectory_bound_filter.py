"""Level 2: Trajectory Bound Filter (TBF).

Paper labels: eq:lower-bound, eq:dtw-individual.
z_q/z_T store vertex fields a before segment fields b.
"""
import torch
import triton
import triton.language as tl


def cross_field_lower_bound(z_q, z_T):
    """eq:lower-bound: L_Theta(q,T), with broadcasting over query/member axes."""
    a_q, b_q = z_q.chunk(2, dim=-1)
    a_T, b_T = z_T.chunk(2, dim=-1)
    L_qT = torch.maximum(b_q - a_T, b_T - a_q).amax(-1)
    return L_qT.clamp_min(0)


@triton.jit
def cross_field_bound_kernel(a_q, b_q, a_T, b_T, AXIS: tl.constexpr):
    # eq:lower-bound; the final axis contains the M shared probes.
    L_qT = tl.maximum(0., tl.max(tl.maximum(b_q - a_T, b_T - a_q), AXIS))
    return L_qT


@triton.jit
def dtw_individual_bound_kernel(query_box_sum, trajectory_box_sum):
    # eq:dtw-individual: maximum, not sum, avoids counting matches twice.
    return tl.maximum(query_box_sum, trajectory_box_sum)
