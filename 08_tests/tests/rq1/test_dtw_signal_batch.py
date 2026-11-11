"""Batched signal dispatch preserves scores, ranking and cache lifetime."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from leap import LEAP, RetrievalConfig
from leap.dtw.dtw_signal_batch import SignalPairPlan
from leap.dtw.dtw_signal_bucket import pairs


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA signal dispatch tests')
class SignalBatchTests(unittest.TestCase):
    def test_bitwise_scores_and_empty_inputs(self):
        lengths = np.array([1,2,3,15,31,32,33,63,64,65,127,128,129,255,513])
        offsets = torch.tensor(np.r_[0,lengths.cumsum()], device='cuda')
        index = SimpleNamespace(offsets=offsets)
        plan = SignalPairPlan(index)
        rows = torch.arange(len(lengths),device='cuda').repeat_interleave(len(lengths))
        ids = torch.arange(len(lengths),device='cuda').repeat(len(lengths))
        generator = torch.Generator(device='cuda').manual_seed(2000)
        for probes in (8,16,32):
            for scale in (0.,1e-10,1.,1e10):
                a = torch.rand((int(offsets[-1]),probes),device='cuda',generator=generator)*scale
                b = torch.rand(a.shape,device='cuda',generator=generator)*scale
                reference = pairs(index,index,rows,ids,fields=(a,b))
                actual = pairs(index,index,rows,ids,fields=(a,b),batch_plan=plan,chunk=37)
                self.assertTrue(torch.equal(actual.view(torch.int32),reference.view(torch.int32)))
        empty = pairs(index,index,rows[:0],ids[:0],fields=(a,b),batch_plan=plan)
        self.assertEqual(empty.numel(),0)
        offsets.add_(0)
        with self.assertRaisesRegex(ValueError,'offsets changed'):
            plan.pairs(index,index,rows,ids,fields=(a,b))
        other = SimpleNamespace(offsets=offsets.clone())
        with self.assertRaisesRegex(ValueError,'offsets changed'):
            plan.pairs(other,other,rows,ids,fields=(a,b))

    def test_single_runtime_plan_takes_precedence(self):
        offsets = torch.tensor([0,2,5],device='cuda')
        index = SimpleNamespace(offsets=offsets)
        plan = SignalPairPlan(index)
        rows = torch.tensor([0],device='cuda')
        fields = (torch.zeros((5,16),device='cuda'),)*2
        expected = torch.tensor([7.],device='cuda')
        runtime = SimpleNamespace(signal_values=lambda *args: expected)
        with patch('leap.common.runtime.context.current',return_value=runtime):
            actual = pairs(index,index,rows,rows,fields=fields,batch_plan=plan)
        self.assertIs(actual,expected)

    def test_tree_plan_is_reused_for_small_and_large_k(self):
        cfg = RetrievalConfig('dtw','signal',dtw_execution='optimized',search_engine='fast')
        model = LEAP.from_asset(config=cfg)
        rng = np.random.default_rng(77)
        paths = [rng.normal(size=(2+i%15,2)).cumsum(0)*model.std+model.mean for i in range(145)]
        paths[1] = paths[0].copy()
        db = model.encode(paths)
        query = model.encode(paths[:3])
        tree = model.build_search_index(db)
        routing = tree.dtw_routing_tree
        plan = routing.signal_plan
        self.assertIsInstance(plan,SignalPairPlan)
        for k in (20,100):
            model.config = replace(cfg,full_large_k=k>64,initial_pool=128 if k>64 else None)
            with patch.object(plan,'pairs',wraps=plan.pairs) as observed:
                actual = model.candidate_indices(query,db,candidate_k=k,search_index=tree)
                self.assertGreater(observed.call_count,0)
            routing.signal_plan = None
            try:
                reference = model.candidate_indices(query,db,candidate_k=k,search_index=tree)
            finally:
                routing.signal_plan = plan
            self.assertTrue(torch.equal(actual,reference))
            with patch.object(plan,'pairs',wraps=plan.pairs) as observed:
                again = model.candidate_indices(query,db,candidate_k=k,search_index=tree)
                self.assertGreater(observed.call_count,0)
            self.assertTrue(torch.equal(actual,again))
            self.assertIs(routing.signal_plan,plan)
            if k>64:
                from leap.common.full_large_k import initial_candidates
                qids=torch.full((3,),-1,device='cuda',dtype=torch.int64)
                with patch.object(plan,'pairs',wraps=plan.pairs) as observed:
                    wide=initial_candidates(model,query,db,qids,k,search_index=tree)
                    self.assertGreater(observed.call_count,0)
                self.assertTrue(torch.equal(wide,actual))
            with patch.object(plan,'pairs',wraps=plan.pairs) as observed:
                full=model.search_range(query,db,search_index=tree,top_k=k,seed_k=k)
                self.assertGreater(observed.call_count,0)
            routing.signal_plan=None
            try:
                old=model.search_range(query,db,search_index=tree,top_k=k,seed_k=k)
            finally:
                routing.signal_plan=plan
            for name in ('indices','distances','radius'):
                self.assertTrue(torch.equal(full[name],old[name]))


if __name__ == '__main__':
    unittest.main()
