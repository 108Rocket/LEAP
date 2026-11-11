"""Small DTW inference batches, with native reductions and network arithmetic."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def supported(index):
    points = index.points
    offsets = getattr(index, 'offsets', None)
    return (points.is_cuda and points.dtype == torch.float32
            and points.ndim == 2 and points.shape[1] == 2 and points.is_contiguous()
            and offsets is not None and 64 <= len(offsets) - 1 <= 512
            and offsets.device == points.device and offsets.is_contiguous()
            and offsets.dtype in (torch.int32, torch.int64)
            and torch.__version__.split('+')[0] == '2.5.1'
            and torch.version.cuda == '12.4' and triton.__version__ == '3.1.0'
            and torch.cuda.get_device_capability(points.device) == (8, 9))


@triton.jit
def _signals(P, TH, OUT, COUNT: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    row, probe = i // M, i % M
    x = tl.load(P + row * 2, valid, other=0.) - tl.load(TH + probe * 2, valid, other=0.)
    y = tl.load(P + row * 2 + 1, valid, other=0.) - tl.load(TH + probe * 2 + 1, valid, other=0.)
    tl.store(OUT + i, libdevice.sqrt_rn(x * x + y * y), valid)


def signals(index, theta):
    output = torch.empty((len(index.points), len(theta)), device=index.points.device,
                         dtype=torch.float32)
    if output.numel():
        _signals[(triton.cdiv(output.numel(), 256),)](
            index.points, theta, output, output.numel(), len(theta), 256,
            num_warps=4, enable_fp_fusion=False)
    return output


@torch.no_grad()
def routing(model, index):
    from leap.dtw import _routing_encoding as fused
    width = int((index.offsets[1:] - index.offsets[:-1]).max())
    # Query shapes retain both native GEMMs, including a full 512-row batch.
    return fused.forward(model, *fused.prepare(index, width), fast_first=False)
