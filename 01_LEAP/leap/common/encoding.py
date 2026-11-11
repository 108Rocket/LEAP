"""Checked FP32 encoding of original packed geometry with owned GPU outputs."""
from threading import Lock

import numba as nb
import numpy as np
import torch
import triton

from leap.common.data import checked_pack, checked_packed


@nb.njit(cache=True, parallel=True)
def _normalize_parallel(raw, mean, std, out):
    blocks = (len(raw) + 4095) // 4096
    invalid = np.zeros(blocks, np.uint8)
    for block in nb.prange(blocks):
        bad = False
        for i in range(block * 4096, min((block + 1) * 4096, len(raw))):
            for d in range(2):
                value = np.float64(raw[i, d])
                normalized = np.float32((value - mean[d]) / std[d])
                out[i, d] = normalized
                bad = bad or not np.isfinite(value) or not np.isfinite(normalized)
        invalid[block] = bad
    return invalid.sum() == 0


@nb.njit(cache=True)
def _normalize_serial(raw, mean, std, out):
    valid = True
    for i in range(len(raw)):
        for d in range(2):
            value = np.float64(raw[i, d])
            normalized = np.float32((value - mean[d]) / std[d])
            out[i, d] = normalized
            valid = valid and np.isfinite(value) and np.isfinite(normalized)
    return valid


def _normalize(raw, mean, std, out):
    function = _normalize_serial if len(raw) <= 65536 else _normalize_parallel
    if not function(raw, mean, std, out):
        raise ValueError('Raw or normalized geometry overflowed or is not finite')


class PackedEncoder:
    """Serial encoder with private host staging and independently owned outputs.

    The input is still raw absolute geometry. Every call validates it. No model,
    input arrays, global thread settings, or previously returned data is mutated.
    """

    def __init__(self, model, *, buffered=None, trajectory_chunk=None,
                 point_chunk=4194304):
        if model.config.precision != 'fp32':
            raise ValueError('Reusable packed encoding requires FP32; use model.encode_packed for FP64')
        if buffered is not None and not isinstance(buffered, bool):
            raise ValueError('buffered must be bool or None')
        if trajectory_chunk is None:
            trajectory_chunk = model.batch_size
        if trajectory_chunk < 1 or point_chunk < 2:
            raise ValueError('Chunk capacities must be positive')
        self.model = model
        self.buffered = buffered
        self.trajectory_chunk = trajectory_chunk
        self.point_chunk = point_chunk
        self._staging = [None, None]
        self._events = [None, None]
        self._lock = Lock()

    def _stage(self, slot, count):
        event = self._events[slot]
        if event is not None:
            event.synchronize()
        host = self._staging[slot]
        if host is None or len(host) < count:
            host = torch.empty((count, 2), dtype=torch.float32, pin_memory=True)
            self._staging[slot] = host
        return host[:count]

    def _fields(self, points, offsets, output, start, stop):
        m = len(self.model.anchors)
        if self.model.config.hausdorff_fields == 'near_far':
            from leap.hausdorff.pointset_index import field_kernel
            field_kernel[(stop - start, triton.cdiv(m, 32))](
                points, offsets[start:stop + 1], self.model.anchors,
                output[start:stop], True, m, PB=32,
                num_warps=4, enable_fp_fusion=False)
        else:
            from leap.common.dual_field import field_kernel
            field_kernel[((stop - start) * m,)](
                points, offsets[start:stop + 1], self.model.anchors,
                output[start:stop], m, num_warps=4, enable_fp_fusion=False)

    @torch.no_grad()
    def encode(self, points, offsets):
        with self._lock:
            points, offsets = checked_packed(points, offsets)
            model = self.model
            if model.config.precision != 'fp32':
                raise ValueError('Encoder precision changed; create a new encoder')
            n, m = len(offsets) - 1, len(model.anchors)
            if not offsets.flags.writeable:
                offsets = offsets.copy()
            device_offsets = torch.from_numpy(offsets).to(model.device)
            output = torch.empty((n, 2 * m), dtype=torch.float32, device=model.device)
            buffered = len(points) <= 65536 if self.buffered is None else self.buffered
            if not buffered:
                normalized = np.empty(points.shape, dtype=np.float32)
                _normalize(points, model.mean, model.std, normalized)
                device_points = torch.from_numpy(normalized).to(model.device)
                for start in range(0, n, self.trajectory_chunk):
                    self._fields(device_points, device_offsets, output, start,
                                 min(start + self.trajectory_chunk, n))
            else:
                device_points = torch.empty(points.shape, dtype=torch.float32,
                                            device=model.device)
                start, iteration = 0, 0
                while start < n:
                    point_start = int(offsets[start])
                    point_limit = point_start + self.point_chunk
                    limit_row = int(np.searchsorted(offsets, point_limit, side='right')) - 1
                    stop = min(n, start + self.trajectory_chunk, max(start + 1, limit_row))
                    point_stop = int(offsets[stop])
                    slot = iteration % 2
                    host = self._stage(slot, point_stop - point_start)
                    _normalize(points[point_start:point_stop], model.mean, model.std,
                               host.numpy())
                    device_points[point_start:point_stop].copy_(host, non_blocking=True)
                    # A buffer may be overwritten only after its H2D copy completes.
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(model.device))
                    self._events[slot] = event
                    self._fields(device_points, device_offsets, output, start, stop)
                    start, iteration = stop, iteration + 1
            if (model.config.hausdorff_target == 'pointset'
                    and model.config.hausdorff_fields != 'near_far'):
                output = torch.cat((output[:, :m], output[:, :m]), dim=1).contiguous()
            if not torch.isfinite(output).all():
                raise ValueError('Dual-field computation overflowed')
            return model._finish_encoding(output, device_points, device_offsets)

    def encode_paths(self, paths):
        # Conversion remains inside this API, and therefore inside its timer.
        points, offsets = checked_pack(paths)
        return self.encode(points, offsets)
