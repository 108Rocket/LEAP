# Test group: RQ5.
# Paper RQ: RQ1: Overall Performance; RQ2: Role of Full Search; RQ5: Initialization Budget and Result Count.
# Checks: Large-k requests select the configured Full-search path.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""The general search API must honor the global large-k switch, too."""
from types import SimpleNamespace

import pytest
from leap import LEAP
from leap import RetrievalConfig


@pytest.mark.parametrize('mode',['auto','range'])
@pytest.mark.parametrize('explicit',[False,True])
def test_full_profile_routes_general_search(mode,explicit):
    calls=[]
    model=SimpleNamespace(config=RetrievalConfig(full_large_k=True),
        search_range=lambda *a,**kw:calls.append((a,kw)) or 'global')
    kw=dict(top_k=1001,candidate_k=1500) if explicit else {}
    assert LEAP.search(model,'q','d',search_index='tree',candidate_search=mode,**kw)=='global'
    assert calls==[(('q','d'),dict(search_index='tree',seed_k=1500 if explicit else None,
                                 top_k=1001 if explicit else 1000))]


@pytest.mark.parametrize('mode',['tree','full'])
def test_full_profile_rejects_candidate_only_override(mode):
    model=SimpleNamespace(config=RetrievalConfig(full_large_k=True))
    with pytest.raises(ValueError,match='global range search'):
        LEAP.search(model,'q','d',candidate_search=mode)
