"""Preserve native FP32 division while supporting FP64 geometry."""
import triton
import triton.language as tl


@triton.jit
def geometry_divide(a, b):
    if a.dtype == tl.float64:
        return a / b
    else:
        return tl.div_rn(a, b)
