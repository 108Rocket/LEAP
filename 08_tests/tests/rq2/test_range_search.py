# Test group: RQ2.
# Paper RQ: RQ2: Role of Full Search.
# Checks: Full range search agrees with raw distances for ties and external queries.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Adopted range API: raw oracle, self exclusion, batching and validation."""
import unittest
from unittest.mock import patch
import numpy as np
import torch

@unittest.skipUnless(torch.cuda.is_available(), 'CUDA retrieval tests')
class RangeSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from leap import DualFieldRetriever
        cls.model=DualFieldRetriever.from_asset()
        rng=np.random.default_rng(903)
        cls.paths=[rng.normal(size=(2+i%19,2)).cumsum(0)*50+cls.model.mean for i in range(137)]
        cls.paths[1]=cls.paths[0].copy()
        cls.paths[2]=np.repeat(cls.paths[0][:1],3,axis=0)
        cls.db=cls.model.encode(cls.paths)
        cls.tree=cls.model.build_search_index(cls.db)

    def oracle(self,q,d,same,k):
        from leap.common.raw import fused_raw_scores
        from leap.common.interval_index import lex_keys
        ids=torch.arange(len(d.coordinates),device='cuda')[None,:].expand(len(q.coordinates),-1).contiguous()
        scores=fused_raw_scores(q,d,ids)
        if same:
            scores[torch.arange(len(q.coordinates),device='cuda'),torch.arange(len(q.coordinates),device='cuda')]=torch.inf
        keys=lex_keys(scores,ids).topk(k,largest=False).values
        return {'indices':keys&0xffffffff,'distances':(keys>>32).int().contiguous().view(torch.float32)}

    def test_self_and_chunk_offsets(self):
        expected=self.oracle(self.db,self.db,True,20)
        for batch in (17,512):
            actual=self.model.search_range(self.db,search_index=self.tree,query_batch=batch)
            for key in expected:self.assertTrue(torch.equal(actual[key],expected[key]),key)
            self.assertFalse((actual['indices']==torch.arange(137,device='cuda')[:,None]).any())

    def test_external_and_configurable_topk(self):
        q=self.model.encode(self.paths[:11])
        for k in (1,5,20):
            expected=self.oracle(q,self.db,False,k)
            actual=self.model.search(q,self.db,candidate_search='range',search_index=self.tree,top_k=k)
            for key in expected:self.assertTrue(torch.equal(actual[key],expected[key]),key)

    def test_small_database_and_all_ties(self):
        db=self.model.encode([self.paths[2]]*6)
        tree=self.model.build_search_index(db)
        actual=self.model.search_range(db,search_index=tree,top_k=5)
        expected=self.oracle(db,db,True,5)
        self.assertTrue(torch.equal(actual['indices'],expected['indices']))

    def test_invalid_and_stale(self):
        for kw in ({'seed_k':65},{'top_k':51},{'query_batch':0}):
            with self.assertRaises(ValueError):self.model.search_range(self.db,search_index=self.tree,**kw)
        with self.assertRaises(ValueError):self.model.search_range(self.db)
        other=self.model.encode(self.paths)
        with self.assertRaises(ValueError):self.model.search_range(other,search_index=self.tree)
        tree=self.model.build_search_index(other)
        other.coordinates.add_(.01)
        with self.assertRaises(ValueError):self.model.search_range(other,search_index=tree)

    def test_memory_guard_splits_not_truncates(self):
        from leap.hausdorff.range_search import search_chunk
        def guarded(q,*args,**kwargs):
            if len(q.coordinates)>40:raise MemoryError('test guard')
            return search_chunk(q,*args,**kwargs)
        with patch('leap.hausdorff.range_search.search_chunk',side_effect=guarded):
            actual=self.model.search_range(self.db,search_index=self.tree)
        expected=self.oracle(self.db,self.db,True,20)
        self.assertTrue(torch.equal(actual['indices'],expected['indices']))

if __name__=='__main__':unittest.main()
