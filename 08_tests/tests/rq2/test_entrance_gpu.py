# Test group: RQ2.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search.
# Checks: Public replay parity and separation of reference labels from search.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Linux CUDA integration only; no training and no paper-result claims."""
import json
from pathlib import Path
import platform

import numpy as np
import pytest
import torch

from leap.reproduction.config import digest, plan
from leap.reproduction.replay import run

pytestmark = pytest.mark.skipif(platform.system() != "Linux" or not torch.cuda.is_available(),
                                reason="L20/Linux CUDA integration")


@pytest.mark.parametrize("metric", ["hausdorff", "dfd", "dtw"])
@pytest.mark.parametrize("same", [True, False])
def test_replay_parity_and_no_reference_in_search(tmp_path, metric, same, monkeypatch):
    from leap import LEAP
    from leap import RetrievalConfig
    root = Path(__file__).resolve().parents[3]
    model = LEAP.from_asset(config=RetrievalConfig.from_json(root / "02_configs" / metric / (metric + ".json")))
    rng = np.random.default_rng(592)
    paths = [model.mean + rng.normal(size=(3 + i % 6, 2)).cumsum(0) * 40 for i in range(144)]

    def pack(path, rows):
        np.savez(path, points=np.concatenate(rows), offsets=np.r_[0, np.cumsum([len(p) for p in rows])],
                 geometry_mean=model.mean, geometry_std=model.std)
    database, queries = tmp_path / "db.npz", tmp_path / "queries.npz"
    pack(database, paths[:128])
    pack(queries, paths[128:])
    db = model.encode(paths[:128])
    q = db if same else model.encode(paths[128:])
    tree = model.build_search_index(db)
    standalone = {
        "initial": model.candidate_indices(q, None if same else db, candidate_k=20, search_index=tree).cpu().numpy(),
        "complete": model.search_range(q, None if same else db, search_index=tree, top_k=20)["indices"].cpu().numpy(),
    }
    # Exhaustive raw distances produce the comparison truth separately.
    from leap.common.raw_geometry_refinement import hausdorff_distances, dtw_distances
    from leap.dfd.dfd import dfd_pairs
    from leap.common.interval_index import lex_keys
    fn = {"hausdorff": hausdorff_distances, "dfd": dfd_pairs, "dtw": dtw_distances}[metric]
    rows = torch.arange(len(q.coordinates), device="cuda")[:, None].expand(-1, 128)
    cols = torch.arange(128, device="cuda")[None].expand_as(rows)
    raw = fn(q, db, rows.flatten(), cols.flatten()).reshape_as(rows)
    if same:
        raw.fill_diagonal_(torch.inf)
    truth = (lex_keys(raw, cols).topk(20, largest=False).values & 0xffffffff).cpu().numpy()
    target = tmp_path / "truth.npy"
    np.save(target, truth)
    data = dict(database=str(database), self_query=same, references={metric: {
        "path": str(target), "format": "top_ids", "database_sha256": digest(database),
        "queries_sha256": digest(queries)}})
    if not same:
        data["queries"] = str(queries)
    cfg = dict(version=1, datasets={"fixture": data}, jobs={"check": {
        "metric": metric, "dataset": "fixture", "warmup": 1, "repeats": 2}})
    p = plan(tmp_path / "config.json", cfg, "check")
    calls = []
    original_candidates = LEAP.candidate_indices
    def observed_candidates(self, *args, **kwargs):
        calls.append(kwargs.get("candidate_search"))
        return original_candidates(self, *args, **kwargs)
    monkeypatch.setattr(LEAP, "candidate_indices", observed_candidates)
    out = run(p, tmp_path / "result")
    result = json.loads((out / "result.json").read_text())
    assert result["reference_used_for_search"] is False
    backend = "tree" if metric == "hausdorff" else "auto"
    assert result["candidate_search"] == backend
    assert calls == [backend] * 3
    assert result["metrics"]["complete"]["HR@20"] == 1.0
    for name in standalone:
        np.testing.assert_array_equal(np.load(out / (name + ".indices.npy")), standalone[name])
        assert len(result["resident_seconds"][name]) == 2
    assert (out / "summary.csv").is_file() and (out / "ALL_DONE.json").is_file()
    with pytest.raises(FileExistsError):
        run(p, out)
