# Test group: RQ2.
# Paper RQ: RQ2: Role of Full Search; RQ3: Contributions of C1--C3.
# Checks: Probe fields, tree retrieval and exact candidate refinement.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""CUDA integration tests for the adopted dual-field/tree retrieval path."""
import unittest
from unittest.mock import patch
import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA retrieval tests')
class IntervalIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from leap import DualFieldRetriever
        cls.model=DualFieldRetriever.from_asset()
        rng=np.random.default_rng(2000)
        cls.paths=[rng.normal(size=(3+i%11,2)).cumsum(0)*30+cls.model.mean for i in range(193)]
        cls.paths[1]=cls.paths[0].copy()
        cls.db=cls.model.encode(cls.paths)
        cls.tree=cls.model.build_search_index(cls.db)

    def test_same_database_exact_candidates(self):
        for k in (5,20,50,64,65,100,192):
            expected=self.model.candidate_indices(self.db,candidate_k=k,candidate_search='full')
            actual=self.model.candidate_indices(self.db,candidate_k=k,candidate_search='tree',search_index=self.tree)
            self.assertTrue(torch.equal(expected,actual))
            self.assertFalse((actual==torch.arange(len(actual),device='cuda')[:,None]).any())

    def test_external_queries_and_raw_refinement(self):
        q=self.model.encode(self.paths[:13])
        expected=self.model.search(q,self.db,candidate_search='full',refinement='full')
        for refinement in ('full','auto'):
            actual=self.model.search(q,self.db,search_index=self.tree,candidate_search='tree',refinement=refinement)
            for key in ('indices','distances'):
                self.assertTrue(torch.equal(expected[key],actual[key]))

    def test_auto_uses_supplied_tree_without_size_thresholds(self):
        expected=self.model.candidate_indices(self.db,candidate_search='full')
        with patch.object(self.tree,'candidates',wraps=self.tree.candidates) as candidates:
            with patch('leap.common.interval_index.reference',side_effect=AssertionError('unexpected scan')):
                actual=self.model.candidate_indices(self.db,search_index=self.tree)
            candidates.assert_called_once()
        self.assertTrue(torch.equal(actual,expected))
        q=self.model.encode(self.paths[:13])
        expected=self.model.candidate_indices(q,self.db,candidate_search='full')
        with patch.object(self.tree,'candidates',wraps=self.tree.candidates) as candidates:
            with patch('leap.common.interval_index.reference',side_effect=AssertionError('unexpected scan')):
                actual=self.model.candidate_indices(q,self.db,search_index=self.tree)
            candidates.assert_called_once()
        self.assertTrue(torch.equal(actual,expected))

    def test_full_override_and_auto_without_index(self):
        expected=self.model.candidate_indices(self.db,candidate_search='full')
        with patch.object(self.tree,'candidates',side_effect=AssertionError('unexpected tree')):
            actual=self.model.candidate_indices(self.db,search_index=self.tree,candidate_search='full')
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(torch.equal(self.model.candidate_indices(self.db),expected))

    def test_auto_large_candidate_pool_keeps_tree(self):
        with patch.object(self.tree,'candidates',wraps=self.tree.candidates) as candidates:
            with patch('leap.common.interval_index.reference',side_effect=AssertionError('unexpected scan')):
                actual=self.model.candidate_indices(self.db,search_index=self.tree,candidate_k=100)
            candidates.assert_called_once()
        expected=self.model.candidate_indices(self.db,candidate_search='full',candidate_k=100)
        self.assertTrue(torch.equal(actual,expected))

    def test_wrong_and_stale_database(self):
        other=self.model.encode(self.paths)
        with self.assertRaises(ValueError):
            self.model.candidate_indices(other,search_index=self.tree)
        tree=self.model.build_search_index(other)
        other.coordinates.add_(.01)
        with self.assertRaises(ValueError):
            self.model.candidate_indices(other,search_index=tree,candidate_search='tree')

    def test_invalid_controls(self):
        for kwargs in ({'candidate_search':'unknown'},{'candidate_search':'tree'},
                       {'candidate_search':'tree','candidate_k':0,'search_index':self.tree}):
            with self.assertRaises(ValueError):
                self.model.candidate_indices(self.db,**kwargs)
        with self.assertRaises(ValueError):
            self.model.build_search_index(self.db,leaf_capacity=32)

    def test_reference_fields(self):
        from leap.common.dual_field import fields_torch
        from leap.common.raw import normalize_points
        paths=self.paths[:16]
        lengths=torch.tensor([len(p) for p in paths],device='cuda')
        padded=np.zeros((len(paths),max(map(len,paths)),2),np.float32)
        for i,p in enumerate(paths):
            padded[i,:len(p)]=normalize_points(p.astype(np.float64),self.model.mean,self.model.std)
        expected=fields_torch(torch.from_numpy(padded).cuda(),lengths,self.model.anchors)
        actual=self.model.encode(paths).coordinates
        self.assertTrue(torch.allclose(expected,actual,atol=2e-6,rtol=2e-6),
                        f'max field discrepancy: {(expected-actual).abs().max().item()}')


if __name__=='__main__':
    unittest.main()
