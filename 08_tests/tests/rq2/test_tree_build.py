# Test group: RQ2.
# Checks: Exact stable partition and group envelopes used by all three distances.
"""Construction equivalence across dimensions, ties and uneven leaf sizes."""
import unittest

import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA tree construction tests')
class TreeBuildTests(unittest.TestCase):
    def test_stable_partitions_and_envelopes(self):
        from leap.common.tree_build import median_members, envelopes
        from leap.hausdorff.pointset_index import joint_partition

        rng = np.random.default_rng(2000)
        for dim in (16, 64, 128):
            for block in (64, 128):
                for n in (1, 63, 64, 65, 128, 129, 257, 1001):
                    for kind in ('random', 'ties', 'constant'):
                        with self.subTest(dim=dim, block=block, n=n, kind=kind):
                            x = rng.normal(size=(n, dim)).astype(np.float32)
                            if kind == 'ties':
                                x = np.round(x)
                            if kind == 'constant':
                                x[:] = 0
                            expected = joint_partition(x, block)
                            backing = torch.zeros((n, dim * 2), device='cuda')
                            backing[:, ::2] = torch.from_numpy(x).cuda()
                            values = backing[:, ::2]
                            members = median_members(values, block)
                            np.testing.assert_array_equal(members.cpu().numpy(), expected)
                            groups = len(expected)
                            ref = np.zeros((2 * groups, 2 * dim), np.float32)
                            for i, ids in enumerate(expected):
                                leaf = x[ids[ids >= 0]]
                                ref[groups + i, :dim] = leaf.max(0)
                                ref[groups + i, dim:] = leaf.min(0)
                            for i in range(groups - 1, 0, -1):
                                ref[i, :dim] = np.maximum(ref[2*i, :dim], ref[2*i+1, :dim])
                                ref[i, dim:] = np.minimum(ref[2*i, dim:], ref[2*i+1, dim:])
                            actual = envelopes(values, members).cpu().numpy()
                            np.testing.assert_array_equal(actual, ref)
                            reverse = envelopes(values, members, upper_first=False).cpu().numpy()
                            np.testing.assert_array_equal(reverse, np.concatenate((ref[:, dim:], ref[:, :dim]), 1))

    def test_invalid_input(self):
        from leap.common.tree_build import median_members

        for x, block in ((torch.zeros((1, 16)), 64),
                         (torch.zeros((0, 16), device='cuda'), 64),
                         (torch.zeros((1, 16), device='cuda'), 0)):
            with self.assertRaises(ValueError):
                median_members(x, block)


if __name__ == '__main__':
    unittest.main()
