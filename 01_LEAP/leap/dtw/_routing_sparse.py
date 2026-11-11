"""Sparse ordered transport with call-local graphs and independent outputs.

Global reduction order is reproduced without processing padded global rows.
Positive mass makes each summary window a contiguous interval. Only its
nonzero weights enter the original ordered FP32 multiply-add sequence.
"""
from types import SimpleNamespace

import numpy as np
import torch
import triton

from leap.dtw import _routing_encoding as fused
from leap.dtw import _routing_workspace as base
from leap.dtw._routing_transport import interval_kernel, scan_native, sum_native


def supported_backend(device):
    # Native reduction and cuBLAS arithmetic are backend-dependent.
    return (torch.__version__.split('+')[0] == '2.5.1'
            and torch.version.cuda == '12.4'
            and triton.__version__ == '3.1.0'
            and torch.cuda.get_device_capability(device) == (8, 9)
            and not torch.cuda.is_current_stream_capturing())


class Workspace:
    def __init__(self, model, width, batch):
        self.model, self.width, self.batch = model, width, batch
        self.device = next(model.parameters()).device
        self.total = self.empty(batch, 1)
        self.padding = torch.tanh(model.net[0].bias)
        self.locals = {}
        self.scan_chunk = 2 << max(4, min(9, (width - 1).bit_length() // 2))

    def empty(self, *shape):
        return torch.empty(shape, device=self.device, dtype=torch.float32)

    def local(self, width):
        if width not in self.locals:
            n = self.batch
            buffers = dict(points=self.empty(n, width, 2), features=self.empty(n, width, 6),
                           hidden=self.empty(n, width, 16), logits=self.empty(n, width))
            buffers.update({name: self.empty(n, width)
                            for name in ('li', 'arc', 'mass', 'cumulative')})
            self.locals[width] = buffers
        return self.locals[width]

    def reduce(self, source, cumulative):
        n, local = source.shape
        sum_native[(n,)](source, self.total, local, self.width,
                         num_warps=1, enable_fp_fusion=False)
        scan_native[(n,)](source, cumulative, local, self.scan_chunk,
                          num_warps=4, enable_fp_fusion=False)

    def run(self, index, local, output):
        n = self.batch
        buffers = self.local(local)
        count = n * local
        lengths = index.offsets[1:] - index.offsets[:-1]
        grid = (triton.cdiv(count, 256),)
        base.geometry_compact[grid](index.points, index.offsets, buffers['points'], buffers['li'],
                                    count, local, local, 256, num_warps=4, enable_fp_fusion=False)
        self.reduce(buffers['li'], buffers['arc'])
        base.features_compact[grid](index.points, index.offsets, buffers['li'], self.total,
                                    buffers['arc'], buffers['features'], count, local, local, 256,
                                    num_warps=4, enable_fp_fusion=False)
        fused.first_layer[(n, triton.cdiv(local, 64))](
            buffers['features'], lengths, self.model.net[0].weight, self.model.net[0].bias,
            self.padding, buffers['hidden'], local, 64, num_warps=4, enable_fp_fusion=False)
        torch.addmm(self.model.net[2].bias, buffers['hidden'].reshape(-1, 16),
                    self.model.net[2].weight.t(), out=buffers['logits'].reshape(-1, 1))
        base.compact_mass[grid](buffers['logits'], lengths, buffers['mass'], count,
                               local, local, 256, num_warps=4, enable_fp_fusion=False)
        self.reduce(buffers['mass'], buffers['cumulative'])
        interval_kernel[(n // 2,)](index.points, index.offsets, buffers['cumulative'], self.total,
                                  output, local, 2, num_warps=1, enable_fp_fusion=False)


@torch.no_grad()
def encode(model, index, batch=512):
    if (not index.points.is_cuda or index.points.dtype != torch.float32
            or len(index.offsets) - 1 <= 512 or batch != 512
            or not supported_backend(index.points.device)):
        return base.encode(model, index, batch)
    lengths = np.diff(index.offsets.cpu().numpy())
    width = int(lengths.max())
    if width <= 128 or width > 4096 or width % 4:
        return base.encode(model, index, batch)

    device = index.points.device
    output = torch.empty((len(lengths), 16), device=device, dtype=index.points.dtype)
    workspace = Workspace(model, width, batch)
    offsets = torch.empty(batch + 1, device=device, dtype=index.offsets.dtype)
    temporary = torch.empty((batch, 16), device=device, dtype=index.points.dtype)
    sub = SimpleNamespace(points=index.points, offsets=offsets)
    widths = {lo: min(width, max(64, triton.next_power_of_2(int(lengths[lo:lo+batch].max()))))
              for lo in range(0, len(lengths) - batch + 1, batch)}
    graphs = {}
    owner = torch.cuda.current_stream(device)
    warm = torch.cuda.Stream(device=device)
    for lo, local in widths.items():
        if local in graphs:
            continue
        offsets.copy_(index.offsets[lo:lo+batch+1])
        workspace.local(local)
        warm.wait_stream(owner)
        with torch.cuda.stream(warm):
            workspace.run(sub, local, temporary)
        owner.wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            workspace.run(sub, local, temporary)
        graphs[local] = graph
    for lo in range(0, len(lengths), batch):
        if lo not in widths:
            tail = SimpleNamespace(points=index.points, offsets=index.offsets[lo:lo+batch+1])
            output[lo:lo+batch] = fused.forward(model, *fused.prepare(tail, width), fast_first=False)
        else:
            offsets.copy_(index.offsets[lo:lo+batch+1])
            graphs[widths[lo]].replay()
            output[lo:lo+batch].copy_(temporary)
    return output
