"""Session-local execution state, never installed on the batch path."""
from contextvars import ContextVar
from contextlib import contextmanager

CURRENT = ContextVar('trigeo_single_query_runtime', default=None)


@contextmanager
def activate(plan, length):
    plan.query_length = length
    token = CURRENT.set(plan)
    try:
        yield plan
    finally:
        CURRENT.reset(token)


def current():
    return CURRENT.get()
