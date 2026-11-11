# Test group: RQ3.
# Paper RQ: RQ2: Role of Full Search; RQ3: Contributions of C1--C3.
# Checks: DTW recurrence, external queries and bounded-search correctness.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""DTW API integration: independent recurrence, external queries and bounded search."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from leap import RetrievalConfig


def reference(a,b):
    prev=np.full(len(b)+1,np.inf);prev[0]=0.
    for x in a:
        cur=np.full(len(b)+1,np.inf)
        for j,y in enumerate(b):
            cur[j+1]=np.linalg.norm(x-y)+min(prev[j],prev[j+1],cur[j])
        prev=cur
    return prev[-1]


class ConfigTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        self.assertEqual(RetrievalConfig().distance,'hausdorff')
        self.assertEqual(RetrievalConfig('dtw').distance,'dtw')
        with self.assertRaises(ValueError):RetrievalConfig('squared_dtw')


@unittest.skipUnless(torch.cuda.is_available(),'CUDA DTW tests')
class DTWTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from leap import DualFieldRetriever
        rng=np.random.default_rng(901)
        cls.model=DualFieldRetriever(rng.normal(size=(64,2)).astype(np.float32),[0.,0.],[1.,1.],config=RetrievalConfig('dtw'))
        cls.paths=[rng.normal(size=(2+i%19,2)).cumsum(0) for i in range(137)]
        cls.paths[1]=cls.paths[0].copy()
        cls.paths[2]=np.zeros((3,2))
        cls.db=cls.model.encode(cls.paths)
        cls.tree=cls.model.build_search_index(cls.db)

    def oracle(self,q,d,same,k):
        from leap.dtw.dtw import dtw_pairs
        from leap.common.interval_index import lex_keys
        ids=torch.arange(len(d.coordinates),device=q.points.device)[None].expand(len(q.coordinates),-1)
        rows=torch.arange(len(q.coordinates),device=q.points.device)[:,None].expand_as(ids)
        dist=dtw_pairs(q,d,rows.reshape(-1),ids.reshape(-1)).reshape_as(ids)
        if same:dist[torch.arange(len(q.coordinates)),torch.arange(len(q.coordinates))]=torch.inf
        keys=lex_keys(dist,ids).topk(k,largest=False).values
        return keys&0xffffffff

    def test_recurrence(self):
        from leap.dtw.dtw import dtw_pairs
        q=torch.tensor([0,2,3,10,30,100],device='cuda')
        d=torch.tensor([1,8,20,9,102,77],device='cuda')
        actual=dtw_pairs(self.db,self.db,q,d).cpu().numpy()
        expected=[reference(self.paths[i].astype(np.float32).astype(np.float64),self.paths[j].astype(np.float32).astype(np.float64)) for i,j in zip(q.tolist(),d.tolist())]
        np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=2e-5)

    def test_self_batches(self):
        expected=self.oracle(self.db,self.db,True,20)
        for batch in (17,2000):
            got=self.model.search_range(self.db,search_index=self.tree,query_batch=batch)
            self.assertTrue(torch.equal(got['indices'],expected))

    def test_external_and_topk(self):
        q=self.model.encode([p[::-1].copy() for p in self.paths[:9]])
        for k in (1,5,20):
            got=self.model.search_range(q,self.db,search_index=self.tree,seed_k=32,top_k=k,query_batch=4)
            self.assertTrue(torch.equal(got['indices'],self.oracle(q,self.db,False,k)))

    def test_external_duplicates_are_not_excluded(self):
        q=self.model.encode(self.paths[:4])
        got=self.model.search(q,self.db,search_index=self.tree,candidate_search='range',top_k=5)
        self.assertTrue(torch.equal(got['indices'],self.oracle(q,self.db,False,5)))

    def test_small_ties_and_fixed(self):
        d=self.model.encode([np.zeros((3,2))]*6)
        tree=self.model.build_search_index(d)
        for mode in ('range','full'):
            got=self.model.search(d,candidate_search=mode,search_index=tree,top_k=5)
            self.assertTrue(torch.equal(got['indices'],self.oracle(d,d,True,5)))

    def test_invalid_stale_and_memory_split(self):
        from leap.dtw.dtw_range import search
        def bounded(q,*args,**kwargs):
            if len(q.coordinates)>40:raise MemoryError('force split')
            return search(q,*args,**kwargs)
        with patch('leap.dtw.dtw_range.search',side_effect=bounded):
            got=self.model.search_range(self.db,search_index=self.tree)
        self.assertTrue(torch.equal(got['indices'],self.oracle(self.db,self.db,True,20)))
        with self.assertRaises(ValueError):self.model.search_range(self.db,search_index=self.tree,query_batch=0)
        with self.assertRaises(ValueError):self.model.candidate_indices(self.db,candidate_search='tree',search_index=self.tree)
        changed=self.model.encode(self.paths)
        tree=self.model.build_search_index(changed)
        changed.points.add_(.1)
        with self.assertRaises(ValueError):self.model.search_range(changed,search_index=tree)
        changed=self.model.encode(self.paths)
        tree=self.model.build_search_index(changed)
        object.__setattr__(changed, 'points', changed.points.clone())
        with self.assertRaises(ValueError):self.model.search_range(changed,search_index=tree)

    def test_blocked_pse_database_and_queries(self):
        from leap.dtw.dtw import pse_candidates
        from leap.common.interval_index import lex_keys
        rng=np.random.default_rng(19)
        d=self.model.encode([rng.normal(size=(2,2)) for _ in range(4101)])
        q=self.model.encode([rng.normal(size=(2,2)) for _ in range(2003)])
        actual=pse_candidates(q,d,7)
        for row in (0,1999,2000,2002):
            vertex=(q.coordinates[row,:64]-d.coordinates[:,:64]).abs().amax(1)
            ends=torch.stack((d.points[d.offsets[:-1]],d.points[d.offsets[1:]-1]),1)
            qends=torch.stack((q.points[q.offsets[row]],q.points[q.offsets[row+1]-1]))
            score=torch.maximum(vertex,(ends-qends).square().sum(-1).sqrt().amax(1))
            ids=torch.arange(len(d.coordinates),device='cuda')
            expected=lex_keys(score,ids).topk(7,largest=False).values&0xffffffff
            self.assertTrue(torch.equal(actual[row],expected))


class PSEDTWTests(DTWTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model.config=RetrievalConfig('dtw','pse')


class OrderedDTWTests(DTWTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model.config=RetrievalConfig('dtw','ordered')


if __name__=='__main__':unittest.main()
