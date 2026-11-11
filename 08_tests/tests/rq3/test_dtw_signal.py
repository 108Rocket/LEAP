# Test group: RQ3.
# Paper RQ: RQ3: Contributions of C1--C3.
# Checks: Ordered probe signals, radial DTW bounds and learning gradients.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Ordered radial DTW: public API, independent recurrence and learned gradient."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from leap import RetrievalConfig
from leap import LEAP
from leap.dtw.dtw_probes import load_asset, validate_asset


def recurrence(cost):
    prev = np.full(cost.shape[1] + 1, np.inf)
    prev[0] = 0.
    for row in cost:
        cur = np.full(len(prev), np.inf)
        for j, value in enumerate(row):
            cur[j + 1] = value + min(prev[j], prev[j + 1], cur[j])
        prev = cur
    return prev[-1]


class SignalAssetTests(unittest.TestCase):
    def test_training_input_formats(self):
        from leap.training.train_dtw_probes import load_paths
        asset = load_asset()
        points = np.arange(24, dtype=np.float64).reshape(12, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'raw.npz'
            for metadata in ({'lengths': np.array([2, 3, 4, 3])},
                             {'offsets': np.array([0, 2, 5, 9, 12])}):
                np.savez(path, points=points, **metadata)
                tr, va = load_paths(path, 2, 1, asset)
                self.assertEqual([len(x) for x in tr], [2, 3])
                np.testing.assert_array_equal(va[0], points[5:9])
            np.savez(path, points=points, lengths=np.array([2, 3, 4, 3]),
                     geometry_mean=np.zeros(2))
            with self.assertRaisesRegex(ValueError, 'normalization'):
                load_paths(path, 2, 1, asset)

    def test_config_and_checksum(self):
        self.assertEqual(RetrievalConfig('dtw').dtw_seed, 'weighted')
        self.assertEqual(RetrievalConfig('dtw', 'signal').dtw_seed, 'signal')
        asset = load_asset()
        self.assertEqual(asset['selected_epoch'], 5)
        bad = copy.deepcopy(asset)
        bad['signal_probes'][0][0] += 1
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_asset(bad)
        bad['format'] = 'other'
        with self.assertRaises(ValueError):
            validate_asset(bad)


class SignalBoundProofTests(unittest.TestCase):
    """Appendix D: independent path sums and the CPU signal encoder."""

    @staticmethod
    def paths(n, m):
        def visit(i, j, prefix):
            path = prefix + [(i, j)]
            if (i, j) == (n-1, m-1):
                yield path
                return
            for di, dj in ((1, 0), (0, 1), (1, 1)):
                if i+di < n and j+dj < m:
                    yield from visit(i+di, j+dj, path)
        return visit(0, 0, [])

    def test_point_path_and_minimum_inequalities(self):
        from types import SimpleNamespace
        from leap.dtw.dtw_signal import encode
        rng = np.random.default_rng(2000)
        for n in range(1, 5):
            for m in range(1, 5):
                for count in (1, 2, 16):
                    with self.subTest(n=n, m=m, probes=count):
                        a, b = rng.normal(size=(n, 2)), rng.normal(size=(m, 2))
                        if n > 1:
                            a[-1] = a[0]
                        theta = torch.tensor(rng.normal(size=(count, 2)), dtype=torch.float64)
                        ra = encode(SimpleNamespace(points=torch.tensor(a)), theta).numpy()
                        rb = encode(SimpleNamespace(points=torch.tensor(b)), theta).numpy()
                        signal = np.abs(ra[:, None]-rb[None]).max(-1)
                        raw = np.linalg.norm(a[:, None]-b[None], axis=-1)
                        self.assertTrue(np.all(signal <= raw + 1e-12))
                        costs = [(sum(signal[i, j] for i, j in path),
                                  sum(raw[i, j] for i, j in path)) for path in self.paths(n, m)]
                        self.assertTrue(all(s <= d + 1e-11 for s, d in costs))
                        self.assertAlmostEqual(recurrence(signal), min(s for s, _ in costs))
                        self.assertAlmostEqual(recurrence(raw), min(d for _, d in costs))
                        self.assertLessEqual(recurrence(signal), recurrence(raw) + 1e-11)

    def test_max_over_probes_is_not_a_sum(self):
        # x=0, y=1 and probes at both endpoints give two identical local bounds.
        differences = np.abs(np.array([0., 1.])-np.array([1., 0.]))
        self.assertEqual(differences.max(), 1.)
        self.assertGreater(differences.sum(), 1.)

    def test_equal_signals_need_not_mean_equal_vertices(self):
        # Opposite vertices can have identical distances to a shared probe.
        self.assertEqual(recurrence(np.zeros((2, 3))), 0.)
        self.assertEqual(recurrence(np.full((2, 3), 2.)), 6.)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA ordered-probe tests')
class SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LEAP.from_asset(config=RetrievalConfig('dtw', 'signal'))
        rng = np.random.default_rng(771)
        cls.paths = [rng.normal(size=(2 + i % 17, 2)).cumsum(0) * cls.model.std + cls.model.mean
                     for i in range(73)]
        cls.paths[1] = cls.paths[0].copy()
        cls.db = cls.model.encode(cls.paths)
        cls.tree = cls.model.build_search_index(cls.db)

    def oracle(self, q, same, k):
        from leap.dtw.dtw import dtw_pairs
        from leap.common.interval_index import lex_keys
        ids = torch.arange(len(self.db.coordinates), device='cuda')[None].expand(len(q.coordinates), -1)
        rows = torch.arange(len(q.coordinates), device='cuda')[:, None].expand_as(ids)
        scores = dtw_pairs(q, self.db, rows.reshape(-1), ids.reshape(-1)).reshape_as(ids)
        if same:
            scores.fill_diagonal_(torch.inf)
        return lex_keys(scores, ids).topk(k, largest=False).values & 0xffffffff

    def test_forward_bound_and_gradient(self):
        from leap.dtw.dtw_signal import pairs, differentiable_pairs
        q = torch.tensor([0, 4, 10, 12], device='cuda')
        d = torch.tensor([8, 1, 6, 7], device='cuda')
        theta = self.model.dtw_probes['signal'].clone().requires_grad_()
        values = pairs(self.db, self.db, q, d, theta=theta)
        score, replay = differentiable_pairs(self.db, self.db, q, d, theta)
        torch.testing.assert_close(score, replay, rtol=3e-6, atol=3e-5)
        theta64 = theta.detach().double().requires_grad_()
        expected = []
        for i, j in zip(q.tolist(), d.tolist()):
            a = self.db.points[self.db.offsets[i]:self.db.offsets[i+1]].double()
            b = self.db.points[self.db.offsets[j]:self.db.offsets[j+1]].double()
            ra = torch.linalg.vector_norm(a[:, None] - theta64, dim=-1)
            rb = torch.linalg.vector_norm(b[:, None] - theta64, dim=-1)
            cost = (ra[:, None] - rb[None]).abs().max(-1).values
            prev = [cost.new_tensor(0.)] + [cost.new_tensor(float('inf'))] * len(b)
            for row in cost:
                cur = [cost.new_tensor(float('inf'))]
                for column, cell in enumerate(row):
                    cur.append(cell + torch.stack([prev[column], prev[column+1], cur[column]]).min())
                prev = cur
            expected.append(prev[-1])
            raw = recurrence(torch.cdist(a, b).detach().cpu().numpy())
            self.assertLessEqual(float(prev[-1].detach()), raw + 1e-6)
        expected = torch.stack(expected)
        torch.testing.assert_close(values.double(), expected.detach(), rtol=3e-6, atol=3e-5)
        score.sum().backward()
        expected.sum().backward()
        self.assertTrue(torch.isfinite(theta.grad).all())
        torch.testing.assert_close(theta.grad.double(), theta64.grad, rtol=1e-3, atol=1e-3)

    def test_coarse_has_no_original_dtw(self):
        with patch('leap.dtw.dtw.dtw_pairs', side_effect=AssertionError('raw scorer in coarse')):
            ids = self.model.candidate_indices(self.db, candidate_k=20)
        self.assertEqual(tuple(ids.shape), (73, 20))
        self.assertFalse(bool((ids == torch.arange(73, device='cuda')[:, None]).any()))

    def test_complete_self_and_external_batches(self):
        expected = self.oracle(self.db, True, 20)
        for batch in (7, 64):
            actual = self.model.search_range(self.db, search_index=self.tree, query_batch=batch)
            self.assertTrue(torch.equal(actual['indices'], expected))
        q = self.model.encode(self.paths[:5])
        for k in (1, 5, 20):
            actual = self.model.search_range(q, self.db, search_index=self.tree, query_batch=2, top_k=k)
            self.assertTrue(torch.equal(actual['indices'], self.oracle(q, False, k)))

    def test_ties_and_recursive_split(self):
        from leap.dtw.dtw_range import search
        def bounded(q, *args, **kwargs):
            if len(q.coordinates) > 11:
                raise MemoryError('exercise bounded batches')
            return search(q, *args, **kwargs)
        with patch('leap.dtw.dtw_range.search', side_effect=bounded):
            actual = self.model.search_range(self.db, search_index=self.tree)
        self.assertTrue(torch.equal(actual['indices'], self.oracle(self.db, True, 20)))
        db = self.model.encode([self.paths[0]] * 6)
        got = self.model.search_range(db, search_index=self.model.build_search_index(db), top_k=5, seed_k=5)
        ids = np.array([[j for j in range(6) if j != i] for i in range(6)])
        np.testing.assert_array_equal(got['indices'].cpu(), ids)

    def test_cache_and_asset_safety(self):
        for field in ('signal', 'radial', 'ordered'):
            db = self.model.encode(self.paths[:23])
            db.dtw_probe_summary[field].add_(.1)
            with self.assertRaisesRegex(ValueError, 'cache'):
                self.model.candidate_indices(db, candidate_k=20)
        model = LEAP.from_asset(config=RetrievalConfig('dtw', 'signal'))
        db = model.encode(self.paths[:23])
        model.dtw_probes['signal'].add_(.01)
        with self.assertRaises(ValueError):
            model.candidate_indices(db, candidate_k=20)
        with self.assertRaisesRegex(ValueError, 'base geometry'):
            LEAP(np.zeros((64, 2), np.float32), [0., 0.], [1., 1.],
                   config=RetrievalConfig('dtw', 'signal'))

    def test_training_loss_is_differentiable(self):
        from leap.training.train_dtw_probes import ranking_loss
        predicted = torch.arange(28, device='cuda', dtype=torch.float32)[None].flip(1).requires_grad_()
        truth = torch.arange(28, device='cuda', dtype=torch.float32)[None]
        loss = ranking_loss(predicted, truth, torch.tensor([19.], device='cuda'))
        loss.backward()
        self.assertGreater(float(loss), 0.)
        self.assertTrue(torch.isfinite(predicted.grad).all())


if __name__ == '__main__':
    unittest.main()
