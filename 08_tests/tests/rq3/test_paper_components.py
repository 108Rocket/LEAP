# Test group: RQ3.
# Paper RQ: RQ2: Role of Full Search; RQ3: Contributions of C1--C3.
# Checks: Geometric field equations, filtering hierarchy and component wiring.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Paper equations, fused component wiring and backwards-compatible public names."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="L20 CUDA validation")


def directed_reference(a, b):
    edge = b[1:] - b[:-1]
    residual = a[:, None] - b[:-1]
    t_star = np.clip((residual * edge).sum(-1) / np.maximum((edge * edge).sum(-1), 1e-30), 0, 1)
    return np.sqrt(((residual - t_star[..., None] * edge) ** 2).sum(-1)).min(1).max()


def test_geometric_equations_and_node_hierarchy():
    from leap import LEAP
    from leap.common.dual_field import score_torch
    from leap.common.group_envelope_filter import node_envelope, node_lower_bound
    from leap.common.trajectory_bound_filter import cross_field_lower_bound
    from leap.common.raw_geometry_refinement import hausdorff_candidate_distances

    rng = np.random.default_rng(281)
    theta = rng.normal(size=(64, 2)).astype(np.float32)
    model = LEAP(theta, [0, 0], [1, 1])
    paths = [rng.normal(size=(2 + i % 11, 2)).astype(np.float32) for i in range(19)]
    paths[1] = paths[0].copy()
    paths[2] = np.zeros((3, 2), dtype=np.float32)
    index = model.encode(paths)
    z_T = index.coordinates
    expected = []
    for points in paths:
        delta = theta[:, None] - points[None]
        a_T = np.sqrt((delta ** 2).sum(-1)).min(1)
        edge = points[1:] - points[:-1]
        residual = delta[:, :-1]
        t_star = np.clip((residual * edge).sum(-1) / np.maximum((edge ** 2).sum(-1), 1e-30), 0, 1)
        b_T = np.sqrt(((residual - t_star[..., None] * edge) ** 2).sum(-1)).min(1)
        expected.append(np.r_[a_T, b_T])
    np.testing.assert_allclose(z_T.cpu(), expected, atol=2e-6, rtol=2e-6)

    L_qT = cross_field_lower_bound(z_T[:, None], z_T[None])
    assert torch.equal(L_qT, score_torch(query=z_T[:, None], database=z_T[None]))
    assert torch.equal(L_qT, L_qT.T)
    # Real-arithmetic self bounds are zero; packed FP32 projection can differ by ulps.
    assert (torch.diag(L_qT) <= 2e-6).all()
    ids = torch.arange(len(paths), device="cuda")[None].expand(len(paths), -1).contiguous()
    H_qT = hausdorff_candidate_distances(index, index, ids)
    raw = np.array([[max(directed_reference(a, b), directed_reference(b, a))
                     for b in paths] for a in paths])
    np.testing.assert_allclose(H_qT.cpu(), raw, atol=2e-6, rtol=2e-6)
    assert (L_qT <= H_qT + 2e-6).all()

    # Masked members cannot change a_bar_U or b_under_U.
    packed = z_T[:5][None]
    valid = torch.tensor([True, True, False, True, False], device="cuda")[None, :, None]
    envelope = node_envelope(packed, valid)
    chosen = z_T[[0, 1, 3]]
    expected_envelope = torch.cat((chosen[:, :64].amax(0), chosen[:, 64:].amin(0)))[None]
    assert torch.equal(envelope, expected_envelope)
    ell_qU = node_lower_bound(z_T, envelope)
    assert (ell_qU[:, None] <= L_qT[:, [0, 1, 3]] + 1e-7).all()
    empty = node_envelope(packed, torch.zeros_like(valid))
    assert torch.isinf(node_lower_bound(z_T, empty)).all()


@pytest.mark.parametrize("distance", ["hausdorff", "dtw"])
def test_paper_api_preserves_results(distance):
    from leap import LEAP
    from leap import DualFieldRetriever
    from leap import RetrievalConfig
    from leap.common import raw
    from leap.hausdorff import range_search
    from leap.common import raw_geometry_refinement as rgr
    assert LEAP is DualFieldRetriever
    assert raw.direction is rgr.directed_hausdorff_squared
    assert raw.fused_raw_scores is rgr.hausdorff_candidate_distances
    assert range_search.pair_values is rgr.hausdorff_distances
    rng = np.random.default_rng(733)
    model = LEAP(rng.normal(size=(64, 2)), [0, 0], [1, 1], config=RetrievalConfig(distance))
    assert model.theta is model.anchors
    paths = [rng.normal(size=(2 + i % 7, 2)) for i in range(73)]
    paths[1] = paths[0].copy()
    database = model.encode(paths)
    tree = model.build_search_index(database)
    for queries in (database, model.encode(paths[:3])):
        old = model.search_range(queries, database, search_index=tree, seed_k=32, query_batch=13)
        new = model.search_range(queries, database, search_index=tree, initial_k=32, query_batch=13)
        both = model.search_range(queries, database, search_index=tree, initial_k=32, seed_k=32)
        assert old.keys() == new.keys() == both.keys()
        for key in old:
            assert torch.equal(old[key], new[key]), key
            assert torch.equal(old[key], both[key]), key
    with pytest.raises(ValueError, match="different candidate counts"):
        model.search_range(database, search_index=tree, initial_k=32, seed_k=50)
    with pytest.raises(ValueError):
        model.search_range(database, search_index=tree, initial_k=19)
