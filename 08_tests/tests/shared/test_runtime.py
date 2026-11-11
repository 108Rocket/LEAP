# Test group: SHARED.
# Paper RQ: Shared support for RQ1--RQ5.
# Checks: Shared session context, configuration serialization and module ownership.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Session context, configuration serialization and module ownership."""
from contextvars import copy_context
import pickle
from types import SimpleNamespace

import pytest

from leap import RetrievalConfig
from leap.common.runtime import context, options
from leap.common.runtime.configuration import FullSearchPlan, resolve


def test_runtime_options_share_one_module(monkeypatch):
    from leap.common.runtime import options as second_import
    monkeypatch.setattr(options, 'dtw_single_layout', False)
    assert second_import.dtw_single_layout is False
    monkeypatch.setattr(second_import, 'dfd_capacity', 4096)
    assert options.dfd_capacity == 4096


def test_context_isolation_and_exception_cleanup():
    outer, inner = SimpleNamespace(), SimpleNamespace()
    assert context.current() is None
    with context.activate(outer, 32):
        copied = copy_context()
        with pytest.raises(RuntimeError):
            with context.activate(inner, 64):
                assert context.current() is inner
                assert copied.run(context.current) is outer
                raise RuntimeError('request failed')
        assert context.current() is outer
    assert context.current() is None


def test_configuration_plan_roundtrip():
    plan = resolve(RetrievalConfig(parameter_mode='auto'), 1000, 20)
    restored = pickle.loads(pickle.dumps(plan))
    assert type(restored) is FullSearchPlan
    assert restored == plan
