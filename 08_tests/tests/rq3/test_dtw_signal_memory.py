# Test group: RQ3.
# Paper RQ: RQ1: Overall Performance; RQ3: Contributions of C1--C3.
# Checks: Memory-bounded probe signals preserve encodings, ties and gradients.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Bounded DTW temporaries preserve encoding, path ties and batch gradients."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


def square_reference(index, qids, dids, theta):
    """Original square layout, retaining the public kernels' default signature."""
    import triton
    from leap.dtw import dtw_signal as signal
    with torch.no_grad():
        fields = signal.encode(index, theta)
        lengths = index.offsets[1:] - index.offsets[:-1]
        block = triton.next_power_of_2(int(torch.maximum(lengths[qids], lengths[dids]).max()))
        width = 2 * block
        size = len(qids)
        trace = torch.empty((size, block, block), device='cuda', dtype=torch.uint8)
        work = torch.empty((size, 3, block), device='cuda')
        values = torch.empty(size, device='cuda')
        pa = torch.empty((size, width), device='cuda', dtype=torch.int64)
        pb, probes = torch.empty_like(pa), torch.empty_like(pa)
        counts = torch.empty(size, device='cuda', dtype=torch.int32)
        signal.signal_kernel[(size,)](
            fields, index.offsets, fields, index.offsets, qids, dids, values, work,
            trace, fields.shape[1], block, True, num_warps=4, enable_fp_fusion=False)
        signal.traceback_kernel[(size,)](
            index.offsets, index.offsets, qids, dids, trace, pa, pb, probes, counts,
            block, width, num_warps=4)
        mask = torch.arange(width, device='cuda')[None] < counts[:, None]
    centers = theta[probes]
    a = torch.linalg.vector_norm(index.points[pa] - centers, dim=-1)
    b = torch.linalg.vector_norm(index.points[pb] - centers, dim=-1)
    return ((a - b).abs() * mask).sum(-1), values


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA DTW memory regression tests')
class SignalMemoryTests(unittest.TestCase):
    def test_chunked_encoding_boundaries_and_dtypes(self):
        from leap.dtw import dtw_signal as signal
        for pdtype, tdtype in ((torch.float32, torch.float32),
                              (torch.float64, torch.float64),
                              (torch.float32, torch.float64)):
            for size in (0, 1, 7, 8, 23):
                with self.subTest(size=size, points=pdtype, probes=tdtype):
                    points = torch.arange(size * 4, device='cuda', dtype=pdtype).reshape(size, 4)[:, ::2]
                    theta = torch.linspace(-1., 1., 8, device='cuda', dtype=tdtype).reshape(4, 2)
                    with torch.no_grad(), patch.object(signal, '_ENCODE_CHUNK_POINTS', 7):
                        expected = (points[:, None] - theta[None]).square().sum(-1).sqrt()
                        actual = signal.encode(SimpleNamespace(points=points), theta)
                    self.assertTrue(torch.equal(expected, actual))
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertTrue(actual.is_contiguous())

    def test_chunked_encoding_retains_gradients(self):
        from leap.dtw import dtw_signal as signal
        for point_grad in (False, True):
            with self.subTest(point_grad=point_grad):
                p = torch.arange(46, device='cuda', dtype=torch.float64).reshape(23, 2) / 13
                t = torch.linspace(-2., -1., 8, device='cuda', dtype=torch.float64).reshape(4, 2)
                p0, p1 = p.clone().requires_grad_(point_grad), p.clone().requires_grad_(point_grad)
                t0, t1 = t.clone().requires_grad_(), t.clone().requires_grad_()
                expected = (p0[:, None] - t0[None]).square().sum(-1).sqrt()
                with patch.object(signal, '_ENCODE_CHUNK_POINTS', 7):
                    actual = signal.encode(SimpleNamespace(points=p1), t1)
                weights = torch.arange(1, expected.numel() + 1, device='cuda').reshape_as(expected)
                (expected * weights).mean().backward()
                (actual * weights).mean().backward()
                self.assertTrue(torch.equal(expected, actual))
                torch.testing.assert_close(t0.grad, t1.grad, atol=1e-12, rtol=1e-12)
                if point_grad:
                    torch.testing.assert_close(p0.grad, p1.grad, atol=1e-12, rtol=1e-12)

    def geometry(self, ties=False):
        lengths = torch.tensor([1, 3, 8, 17, 65], device='cuda')
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        x = torch.arange(94, device='cuda', dtype=torch.float32)
        points = torch.stack([x.sin(), (x * .7).cos()], dim=1)
        if ties:
            points.zero_()
        theta = torch.tensor([[2., 3.], [-2., 1.], [.2, -2.], [.2, -2.]], device='cuda')
        qids = torch.tensor([4, 1, 4, 2, 3, 0, 4, 4], device='cuda')
        dids = torch.tensor([1, 4, 1, 0, 2, 3, 1, 1], device='cuda')
        return SimpleNamespace(points=points, offsets=offsets), qids, dids, theta

    def test_rectangular_trace_chunking_order_and_ties(self):
        from leap.dtw import dtw_signal as signal
        for ties in (False, True):
            with self.subTest(ties=ties):
                index, qids, dids, theta = self.geometry(ties)
                t0, t1 = theta.clone().requires_grad_(), theta.clone().requires_grad_()
                expected, replay0 = square_reference(index, qids, dids, t0)
                trace_shapes = []
                empty = torch.empty

                def tracked_empty(shape, *args, **kwargs):
                    if kwargs.get('dtype') == torch.uint8:
                        trace_shapes.append(tuple(shape))
                    return empty(shape, *args, **kwargs)

                # Three pairs fit the short-target bucket; the fourth exercises its tail.
                with patch.object(signal, '_TRACE_MAX_BYTES', 780), \
                        patch.object(signal.torch, 'empty', side_effect=tracked_empty):
                    actual, replay1 = signal.differentiable_pairs(index, index, qids, dids, t1)
                self.assertIn((3, 65, 4), trace_shapes)
                self.assertTrue(all(n * h * w <= 780 for n, h, w in trace_shapes))
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(replay0, replay1))
                weights = torch.linspace(.1, 1., len(qids), device='cuda')
                (expected * weights).sum().backward()
                (actual * weights).sum().backward()
                self.assertTrue(torch.equal(t0.grad, t1.grad))

    def test_trace_budget_rejects_one_oversized_pair(self):
        from leap.dtw import dtw_signal as signal
        index, qids, dids, theta = self.geometry()
        with patch.object(signal, '_TRACE_MAX_BYTES', 1):
            with self.assertRaisesRegex(MemoryError, 'single DTW traceback'):
                signal.differentiable_pairs(index, index, qids[:1], dids[:1], theta)

    def test_mining_reuses_self_encoding_and_preserves_pair_order(self):
        from leap.dtw import dtw_signal as signal
        index, qids, dids, theta = self.geometry()
        _, expected = square_reference(index, qids, dids, theta)
        with patch.object(signal, 'encode', wraps=signal.encode) as encode:
            actual = signal.pairs(index, index, qids, dids, theta=theta, chunk=2)
        self.assertEqual(encode.call_count, 1)
        self.assertTrue(torch.equal(actual, expected))
        fields = signal.encode(index, theta)
        empty = signal.pairs(index, index, qids[:0], dids[:0], fields=(fields, fields))
        self.assertEqual(tuple(empty.shape), (0,))


if __name__ == '__main__':
    unittest.main()
