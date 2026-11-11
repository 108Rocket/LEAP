# Test group: RQ4.
# Paper RQ: RQ4: Database Scalability.
# Checks: Candidate generation remains valid beyond one million database rows.
# Role: supporting regression test. See 08_tests/tests/README.md for the paper mapping.

"""Database-size boundary for the Hausdorff GPU launch grid."""
from types import SimpleNamespace

import pytest
import torch

from leap import RetrievalConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA retrieval tests')


@torch.no_grad()
def test_large_database_hausdorff_grid_does_not_exceed_grid_y():
    from leap.common.full_large_k import initial_candidates
    n = 1048577
    q = SimpleNamespace(coordinates=torch.zeros((1,128),device='cuda'),
                        points=torch.empty((0,2),device='cuda'))
    d = SimpleNamespace(coordinates=torch.zeros((n,128),device='cuda'))
    model = SimpleNamespace(config=RetrievalConfig(full_large_k=True))
    ids = initial_candidates(model,q,d,torch.full((1,),-1,device='cuda',dtype=torch.int64),1000)
    assert torch.equal(ids,torch.arange(1000,device='cuda')[None])
