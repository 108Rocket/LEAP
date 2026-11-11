# Test group: RQ3.
# Paper RQ: RQ3: Contributions of C1--C3.
# Checks: Ordered DTW summaries and alignment across shape boundaries.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from leap import RetrievalConfig


def test_config_roundtrip(tmp_path):
    assert RetrievalConfig('dtw').dtw_seed=='weighted'
    p=tmp_path/'config.json'
    p.write_text(json.dumps({'distance':'dtw','dtw_seed':'pse'}))
    assert RetrievalConfig.from_json(p)==RetrievalConfig('dtw','pse')
    with pytest.raises(ValueError):RetrievalConfig('dtw','other')
    p.write_text('{"unknown":1}')
    with pytest.raises(ValueError):RetrievalConfig.from_json(p)


@pytest.fixture
def model():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from leap import DualFieldRetriever
    return DualFieldRetriever(np.random.default_rng(4).normal(size=(64,2)),[0,0],[1,1],config=RetrievalConfig('dtw','ordered'))


def test_summary_reference_and_stale(model):
    rng=np.random.default_rng(5)
    paths=[rng.normal(size=(n,2)) for n in (2,3,7,8,9,20,33,199,200,257)]
    index=model.encode(paths)
    expected=[]
    for path in paths:
        p=path.astype(np.float32);n=len(p);bins=np.arange(n)*8//n
        expected.append([p[bins==b].mean(0) if (bins==b).any() else p[min(((2*b+1)*n)//16,n-1)] for b in range(8)])
    np.testing.assert_allclose(index.ordered_summary.cpu().numpy().reshape(-1,8,2),expected,atol=2e-6,rtol=2e-6)
    with pytest.raises(ValueError):model.candidate_indices(replace(index,ordered_summary=None))
    with pytest.raises(ValueError):model.candidate_indices(replace(index,ordered_summary=index.ordered_summary.clone()))
    index.ordered_summary.add_(.1)
    with pytest.raises(ValueError):model.candidate_indices(index)


def test_rectangular_and_self_block_boundaries(model):
    from leap.common.interval_index import lex_keys
    rng=np.random.default_rng(8)
    d=model.encode([rng.normal(size=(2+i%19,2)) for i in range(4101)])
    q=model.encode([rng.normal(size=(2+i%11,2)) for i in range(2003)])
    for query,same in ((q,False),(d,True)):
        actual=model.candidate_indices(query,d,candidate_k=7)
        c=d.ordered_summary.reshape(-1,8,2)
        n=d.offsets[1:]-d.offsets[:-1]
        for row in (0,1999,2000,len(query.coordinates)-1):
            qc=query.ordered_summary[row].reshape(8,2)
            qn=query.offsets[row+1]-query.offsets[row]
            score=(c-qc).square().sum(-1).sqrt().sum(1)/8*((n+qn)*.5)
            if same:score[row]=torch.inf
            ids=torch.arange(len(c),device='cuda')
            expected=lex_keys(score,ids).topk(7,largest=False).values&0xffffffff
            assert torch.equal(actual[row],expected)
