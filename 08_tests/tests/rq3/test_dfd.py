# Test group: RQ3.
# Paper RQ: RQ2: Role of Full Search; RQ3: Contributions of C1--C3.
# Checks: DFD recurrence, ordered bounds and Full-search equivalence.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""DFD API: independent recurrence, bounds, external queries and raw parity."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from leap import RetrievalConfig


def reference(a,b):
    previous=np.full(len(b)+1,np.inf); previous[0]=0.
    for x in a:
        current=np.full(len(b)+1,np.inf)
        for j,y in enumerate(b):
            current[j+1]=max(np.linalg.norm(x-y),min(previous[j],previous[j+1],current[j]))
        previous=current
    return previous[-1]


class ConfigTests(unittest.TestCase):
    def test_dfd_opt_in(self):
        self.assertEqual(RetrievalConfig('dfd').distance,'dfd')
        self.assertEqual(RetrievalConfig().distance,'hausdorff')
        self.assertEqual(RetrievalConfig('dfd').dfd_initial,'light')
        with self.assertRaises(ValueError): RetrievalConfig('dfd',dfd_initial='unknown')


@unittest.skipUnless(torch.cuda.is_available(),'CUDA DFD tests')
class DFDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from leap import LEAP
        rng=np.random.default_rng(319)
        cls.model=LEAP(rng.normal(size=(64,2)),[0.,0.],[1.,1.],config=RetrievalConfig('dfd'))
        cls.paths=[rng.normal(size=(2+i%67,2)).cumsum(0) for i in range(137)]
        cls.paths[1]=cls.paths[0].copy(); cls.paths[2]=np.zeros((3,2))
        cls.db=cls.model.encode(cls.paths)
        cls.tree=cls.model.build_search_index(cls.db)

    def oracle(self,q,d,same,k):
        from leap.dfd.dfd import dfd_pairs
        from leap.common.interval_index import lex_keys
        ids=torch.arange(len(d.coordinates),device='cuda')[None].expand(len(q.coordinates),-1)
        rows=torch.arange(len(q.coordinates),device='cuda')[:,None].expand_as(ids)
        values=dfd_pairs(q,d,rows.reshape(-1),ids.reshape(-1)).reshape_as(ids)
        if same: values[torch.arange(len(q.coordinates)),torch.arange(len(q.coordinates))]=torch.inf
        keys=lex_keys(values,ids).topk(k,largest=False).values
        return keys&0xffffffff,(keys>>32).int().contiguous().view(torch.float32)

    def test_recurrence_and_order_bound(self):
        from leap.dfd.dfd import dfd_pairs
        from leap.dfd.dfd_geometry import ordered_bound
        q=torch.tensor([0,2,3,10,30,100],device='cuda')
        d=torch.tensor([1,8,20,9,102,77],device='cuda')
        actual=dfd_pairs(self.db,self.db,q,d)
        expected=[reference(self.paths[i].astype(np.float32).astype(np.float64),
                            self.paths[j].astype(np.float32).astype(np.float64))
                  for i,j in zip(q.tolist(),d.tolist())]
        np.testing.assert_allclose(actual.cpu(),expected,rtol=3e-6,atol=2e-5)
        for s in (8,16,32):
            bound=ordered_bound(self.db,self.db,q,d,s)
            self.assertTrue(bool((bound<=actual+2e-5).all()))

    def test_retained_endpoints_and_nesting(self):
        lengths=(self.db.offsets[1:]-self.db.offsets[:-1]).cpu().numpy()
        ids={s:v.cpu().numpy() for s,v in self.db.dfd_summary['ids'].items()}
        for s,a in ids.items():
            self.assertTrue((a[:,0]==0).all() and (a[:,-1]==lengths-1).all())
            self.assertTrue((np.diff(a,axis=1)>=0).all() and (a<lengths[:,None]).all())
        self.assertTrue((ids[8][:,:,None]==ids[32][:,None,:]).any(2).all())

    def test_exact_pse_tree_self_and_external(self):
        from leap.dfd.dfd_tree import candidate_pool
        from leap.dfd.dfd import dfd_pair_bounds
        from leap.common.interval_index import lex_keys
        for same in (True,False):
            queries=self.db if same else self.model.encode(self.paths[:5])
            n=len(queries.coordinates)
            qids=torch.arange(n,device='cuda') if same else torch.full((n,),-1,device='cuda',dtype=torch.int64)
            ids=torch.arange(len(self.paths),device='cuda')[None].expand(n,-1)
            rows=torch.arange(n,device='cuda')[:,None].expand_as(ids)
            values=dfd_pair_bounds(queries,self.db,rows.reshape(-1),ids.reshape(-1)).reshape_as(ids)
            if same: values[torch.arange(n),torch.arange(n)]=torch.inf
            expected=lex_keys(values,ids).topk(100,largest=False).values
            actual,scores,_=candidate_pool(queries,self.db,self.tree,qids,100)
            self.assertTrue(torch.equal(actual,expected&0xffffffff))
            self.assertTrue(torch.equal(scores,(expected>>32).int().contiguous().view(torch.float32)))

    def test_endpoint_bound_uses_round_to_nearest(self):
        from types import SimpleNamespace
        from leap.dfd.dfd import dfd_pair_bounds
        rng = np.random.default_rng(2031)
        source = [rng.normal(size=(512, 2)) * 100 for _ in range(2)]
        for dtype in (np.float32, np.float64):
            a, b = (value.astype(dtype) for value in source)
            def index(points):
                packed = torch.from_numpy(np.repeat(points, 2, axis=0)).cuda()
                return SimpleNamespace(points=packed,
                    offsets=torch.arange(0, 1025, 2, device='cuda'),
                    coordinates=torch.zeros((512, 128), dtype=packed.dtype, device='cuda'))
            ids = torch.arange(512, device='cuda')
            actual = dfd_pair_bounds(index(a), index(b), ids, ids).cpu().numpy()
            delta = a - b
            expected = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])
            np.testing.assert_array_equal(actual, expected)

    def test_self_batches(self):
        ids,scores=self.oracle(self.db,self.db,True,20)
        for batch in (17,512):
            got=self.model.search_range(self.db,search_index=self.tree,query_batch=batch)
            self.assertTrue(torch.equal(got['indices'],ids))
            self.assertTrue(torch.equal(got['distances'],scores))

    def test_light_budget_and_progressive_reference(self):
        from leap import LEAP
        from leap.dfd.dfd_tree import candidate_pool, initial_candidates
        import leap.dfd.dfd_geometry as geometry
        expected=self.oracle(self.db,self.db,True,20)[0]
        with patch('leap.dfd.dfd_tree.candidate_pool',wraps=candidate_pool) as pool:
            with patch('leap.dfd.dfd_geometry.ordered_bound',wraps=geometry.ordered_bound) as ordered:
                initial=self.model.candidate_indices(self.db,search_index=self.tree)
        self.assertEqual(initial.shape,(len(self.paths),32))
        self.assertEqual(pool.call_args.args[-1],64)
        self.assertEqual([c.args[-1] for c in ordered.call_args_list],[8])
        old=LEAP(self.model.theta.cpu().numpy(),[0.,0.],[1.,1.],
                   config=RetrievalConfig('dfd',dfd_initial='progressive'))
        qids=torch.arange(len(self.paths),device='cuda')
        self.assertTrue(torch.equal(old.candidate_indices(self.db,search_index=self.tree),
            initial_candidates(self.db,self.db,self.tree,qids,50,strategy='progressive')))
        self.assertTrue(torch.equal(old.search_range(self.db,search_index=self.tree)['indices'],expected))

    def test_external_duplicates_and_topk(self):
        queries=self.model.encode(self.paths[:6]+[self.paths[9][::-1].copy()])
        for k in (1,5,20):
            got=self.model.search_range(queries,self.db,search_index=self.tree,initial_k=32,
                                        top_k=k,query_batch=3)
            ids,scores=self.oracle(queries,self.db,False,k)
            self.assertTrue(torch.equal(got['indices'],ids))
            self.assertTrue(torch.equal(got['distances'],scores))

    def test_small_ties(self):
        db=self.model.encode([np.zeros((2,2))]*6)
        tree=self.model.build_search_index(db)
        expected,_=self.oracle(db,db,True,5)
        for mode in ('range','tree'):
            got=self.model.search(db,search_index=tree,candidate_search=mode,top_k=5)
            self.assertTrue(torch.equal(got['indices'],expected))

    def test_invalid_stale_and_bounded_batch(self):
        from leap.dfd.dfd_range import search
        def bounded(q,*args,**kwargs):
            if len(q.coordinates)>40: raise MemoryError('force recursive split')
            return search(q,*args,**kwargs)
        with patch('leap.dfd.dfd_range.search',side_effect=bounded):
            got=self.model.search_range(self.db,search_index=self.tree)
        self.assertTrue(torch.equal(got['indices'],self.oracle(self.db,self.db,True,20)[0]))
        with self.assertRaises(ValueError): self.model.candidate_indices(self.db,candidate_search='full')
        with self.assertRaises(ValueError): self.model.candidate_indices(self.db)
        with self.assertRaises(ValueError): self.model.search_range(self.db,search_index=self.tree,query_batch=0)
        db=self.model.encode(self.paths[:8]); tree=self.model.build_search_index(db)
        db.points.add_(.1)
        with self.assertRaises(ValueError): self.model.search_range(db,search_index=tree,top_k=5)
        db=self.model.encode(self.paths[:8]); tree=self.model.build_search_index(db)
        db.dfd_summary['ids'][8].zero_()
        with self.assertRaises(ValueError): self.model.search_range(db,search_index=tree,top_k=5)


if __name__=='__main__': unittest.main()
