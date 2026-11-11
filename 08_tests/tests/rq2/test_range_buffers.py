# Test group: RQ2.
# Paper RQ: RQ2: Role of Full Search.
# Checks: Candidate buffers preserve complete enumeration through overflow.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""One-pass candidate storage must preserve complete enumeration and overflow."""
import unittest
from unittest.mock import patch

import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA range-buffer tests')
class RangeBufferTests(unittest.TestCase):
    def test_staging_overflow_self_exclusion_and_padding(self):
        from leap import LEAP
        from leap import RetrievalConfig
        from leap.common import range_buffers
        from leap.hausdorff import range_search
        from leap.dfd import dfd_range
        for metric, module in [('hausdorff', range_search), ('dfd', dfd_range)]:
            with self.subTest(metric=metric):
                model = LEAP.from_asset(config=RetrievalConfig(distance=metric))
                rng = np.random.default_rng(918)
                paths = [rng.normal(size=(2+i % 13, 2)).cumsum(0)*40+model.mean for i in range(70)]
                paths[1] = paths[0].copy()
                paths[2] = np.repeat(paths[0][:1], 3, axis=0)
                db = model.encode(paths)
                tree = model.build_search_index(db)
                original = module.enumerate_candidates
                audits = []

                def checked(*args, **kwargs):
                    actual = original(*args, **kwargs)
                    with patch.object(range_buffers, 'STAGE_CAPACITY', 0):
                        expected = original(*args, **kwargs)
                    for a, b in zip(actual, expected):
                        self.assertTrue(torch.equal(a, b) if torch.is_tensor(a) else a == b)
                    audits.append(actual[-1])
                    return actual

                with patch.object(range_buffers, 'STAGE_CAPACITY', 0):
                    reference = model.search_range(db, search_index=tree, query_batch=13)
                self.assertFalse((reference['indices'] == torch.arange(70, device='cuda')[:, None]).any())
                for cap in (2048, 8):
                    with patch.object(range_buffers, 'STAGE_CAPACITY', cap), \
                         patch.object(module, 'enumerate_candidates', checked):
                        actual = model.search_range(db, search_index=tree, query_batch=13)
                    for key in ('indices', 'distances', 'counts', 'radius'):
                        self.assertTrue(torch.equal(actual[key], reference[key]), key)
                self.assertTrue(audits and min(audits) > 8)

    def test_key_scatter_preserves_ties_and_infinity(self):
        from leap.common.range_buffers import top_keys
        from leap.common.interval_index import lex_keys
        ids = torch.tensor([7, 3, 2, 9, 1], device='cuda')
        values = torch.tensor([1., 1., float('inf'), 0., 0.], device='cuda')
        rows = torch.tensor([0, 0, 0, 1, 1], device='cuda')
        offsets = torch.tensor([0, 3], device='cuda')
        expected = torch.full((2, 3), 9187343244130779135, device='cuda', dtype=torch.int64)
        expected[rows, torch.arange(5, device='cuda')-offsets[rows]] = lex_keys(values, ids)
        self.assertTrue(torch.equal(top_keys(values, ids, rows, offsets, 2, 3, 2),
                                    expected.topk(2, largest=False).values))


if __name__ == '__main__':
    unittest.main()
