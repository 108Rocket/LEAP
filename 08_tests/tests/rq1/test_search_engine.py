"""RQ1/RQ2: optional search engine compatibility and unchanged ranking outputs."""
from copy import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import unittest

import numpy as np
import torch

from leap.common.config import RetrievalConfig
from leap.common.dual_field import DualFieldRetriever


ROOT = Path(__file__).resolve().parents[3]


class ConfigTests(unittest.TestCase):
    def test_old_files_default_to_original(self):
        for path in (ROOT / '02_configs').rglob('*.json'):
            if path.name.startswith('fast_'):
                continue
            with self.subTest(path=path):
                config = RetrievalConfig.from_json(path)
                self.assertEqual(config.search_engine, 'original')
                self.assertNotIn('search_engine', json.loads(path.read_text()))

    def test_fast_and_auto(self):
        for metric in ('hausdorff', 'dfd', 'dtw'):
            name = 'fast_pointset_hd.json' if metric == 'hausdorff' else f'fast_{metric}.json'
            config = RetrievalConfig.from_json(ROOT / '02_configs' / metric / name)
            auto = replace(config, parameter_mode='auto')
            for k in (1, 20, 50, 100, 1000):
                self.assertEqual(auto.resolve(100000, k).config.search_engine, 'fast')
        for value in (True, None, 'automatic', 'FAST'):
            with self.assertRaises(ValueError):
                RetrievalConfig(distance='dfd', search_engine=value)

    def test_unsupported_combinations_fail_explicitly(self):
        for extra in (dict(precision='fp64'), dict(large_k=True)):
            with self.assertRaises(ValueError):
                RetrievalConfig(distance='dfd', search_engine='fast', **extra)
        with self.assertRaises(ValueError):
            RetrievalConfig(search_engine='fast')

    def test_switching_requires_matching_tree(self):
        model = object.__new__(DualFieldRetriever)
        for engine in ('original', 'fast'):
            model.config = RetrievalConfig(distance='dfd', search_engine=engine)
            model._check_search_engine(SimpleNamespace(search_engine=engine))
            model._check_search_engine(None)
            other = 'original' if engine == 'fast' else 'fast'
            with self.assertRaisesRegex(ValueError, 'rebuild'):
                model._check_search_engine(SimpleNamespace(search_engine=other))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class SearchParityTests(unittest.TestCase):
    def test_three_metrics_ties_paging_and_self_exclusion(self):
        rng = np.random.default_rng(2000)
        paths = [np.cumsum(rng.normal(size=(4 + i % 9, 2)), axis=0) for i in range(1100)]
        paths[20:24] = [paths[0].copy() for _ in range(4)]
        for metric in ('hausdorff', 'dfd', 'dtw'):
            with self.subTest(metric=metric):
                name = 'fast_pointset_hd.json' if metric == 'hausdorff' else f'fast_{metric}.json'
                cfg = RetrievalConfig.from_json(ROOT / '02_configs' / metric / name)
                cfg = replace(cfg, search_engine='original')
                folder = ROOT / '03_checkpoints/seed2000/sdd/porto' / metric
                model = DualFieldRetriever.from_asset(folder / 'asset.json', config=cfg,
                    routing_asset=folder / 'router.json' if metric == 'dtw' else None)
                # Generate around this model's normalization, not arbitrary raw coordinates.
                raw = [p * model.std + model.mean for p in paths]
                db = model.encode(raw)
                query = model.encode(raw[:5])
                tree = model.build_search_index(db)
                fast = copy(model)
                fast.config = replace(cfg, search_engine='fast')
                fast._dtw_route_cache = None
                ftree = fast.build_search_index(db)
                for k in (1, 20, 50, 128, 1000, 1025):
                    expected = model.candidate_indices(query, db, k, search_index=tree)
                    actual = fast.candidate_indices(query, db, k, search_index=ftree)
                    self.assertTrue(torch.equal(expected, actual), (metric, k))
                for m, t in ((model, tree), (fast, ftree)):
                    ids = m.candidate_indices(db, candidate_k=1, search_index=t)
                    self.assertFalse(bool((ids[:, 0] == torch.arange(len(paths), device='cuda')).any()))
                old = model.search_range(query, db, search_index=tree)
                new = fast.search_range(query, db, search_index=ftree)
                self.assertTrue(torch.equal(old['indices'], new['indices']))
                self.assertTrue(torch.equal(old['distances'], new['distances']))
                for m in (model, fast):
                    m.config = replace(m.config, parameter_mode='auto')
                for k in (1, 50, 1000):
                    old = model.search_range(query, db, top_k=k, search_index=tree)
                    new = fast.search_range(query, db, top_k=k, search_index=ftree)
                    self.assertTrue(torch.equal(old['indices'], new['indices']), (metric,k,'auto_full'))
                    self.assertTrue(torch.equal(old['distances'], new['distances']))
                    self.assertEqual(new['configuration']['search_engine'], 'fast')
                del model, fast, tree, ftree, db, query, old, new
                torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
