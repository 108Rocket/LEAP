"""Private storage for synchronous raw-input, single-request retrieval."""

import threading
from contextlib import nullcontext

import numpy as np
import torch

from leap.common.data import checked_pack
from leap.common.dual_field import field_kernel
from leap.common.interval_index import tree_kernel
from leap.common.raw import normalize_points_into

OFFSET_BYTES = 16
POINT_BYTES = 8


class SingleQuerySession:
    """Reuse storage without returning aliases to buffers overwritten next time.

    Each session serializes its calls on one CUDA stream. It never changes the
    global thread settings, model weights, query points or retrieval budget.
    """

    def __init__(self, model, database, search_index, top_k=20, initial_capacity=256):
        if model.config.query_profile != 'single_query':
            raise ValueError('A query session requires query_profile=single_query')
        if model.config.initial_search == 'full':
            raise ValueError('Explicit scan sessions require parameter_mode=auto')
        if (
            model.config.large_k
            or model.config.full_large_k
            or model.config.hausdorff_target != 'polyline'
        ):
            raise ValueError(
                'Single-query sessions support ordinary Top-k polyline/DFD/DTW profiles'
            )
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= min(64, len(database.coordinates))
        ):
            raise ValueError('top_k must be an integer in [1,min(64,N)]')
        if (
            isinstance(initial_capacity, bool)
            or not isinstance(initial_capacity, int)
            or initial_capacity < 2
        ):
            raise ValueError('initial_capacity must be an integer >=2')
        model._check_index(database)
        search_index.validate(database)
        self.model, self.database, self.tree = model, database, search_index
        self.config = model.config
        self.asset_id = model.asset_id
        self.probe_version = (model.anchors.data_ptr(), model.anchors._version)
        self.normalization = (model.mean.tobytes(), model.std.tobytes())
        self.top_k = top_k
        self._lock = threading.RLock()
        self.stream = torch.cuda.Stream(device=model.device)
        self.stream.wait_stream(torch.cuda.current_stream(model.device))
        self.fields = torch.empty((1, 128), device=model.device, dtype=torch.float32)
        self.valid = torch.empty(64, device=model.device, dtype=torch.uint8)
        self.host_valid = torch.empty(64, dtype=torch.uint8, pin_memory=True)
        self.result = torch.empty((1, top_k), device=model.device, dtype=torch.int64)
        self.host_result = torch.empty((1, top_k), dtype=torch.int64, pin_memory=True)
        self.visits = torch.empty((1, 3), device=model.device, dtype=torch.int32)
        self._reserve(initial_capacity)
        self.plan = None
        supported = model.config.distance == 'dfd' or (
            model.config.distance == 'dtw'
            and model.config.dtw_seed == 'signal'
            and model.config.dtw_routing == 'learned'
            and model.config.dtw_execution == 'optimized'
        )
        if supported and model.config.single_query_engine == 'specialized':
            from leap.common.runtime.plan import SingleQueryPlan

            self.plan = SingleQueryPlan(model, database, search_index)
            self.stream.wait_stream(torch.cuda.current_stream(model.device))

    def _reserve(self, size):
        if getattr(self, 'plan', None) is not None:
            # Graphs bind input addresses; a new packet invalidates those graphs.
            self.plan.graphs.clear()
        self.capacity = 1 << (max(2, size) - 1).bit_length()
        # One aligned packet: two int64 offsets followed by float32 xy points.
        self.host_packet = torch.empty(
            OFFSET_BYTES + POINT_BYTES * self.capacity, dtype=torch.uint8, pin_memory=True
        )
        self.device_packet = torch.empty_like(self.host_packet, device=self.model.device)
        self.host_offsets = self.host_packet[:OFFSET_BYTES].view(torch.int64).numpy()
        self.host_points = (
            self.host_packet[OFFSET_BYTES:].view(torch.float32).reshape(self.capacity, 2).numpy()
        )
        self.offsets = self.device_packet[:OFFSET_BYTES].view(torch.int64)
        self.points = (
            self.device_packet[OFFSET_BYTES:].view(torch.float32).reshape(self.capacity, 2)
        )

    def _validate(self):
        model = self.model
        if (
            model.config != self.config
            or model.asset_id != self.asset_id
            or (model.anchors.data_ptr(), model.anchors._version) != self.probe_version
            or (model.mean.tobytes(), model.std.tobytes()) != self.normalization
        ):
            raise ValueError('Model changed; create a new query session')
        model._check_index(self.database)
        self.tree.validate(self.database)

    def _prepare_input(self, raw_trajectory):
        """Normalize and transfer on the caller's active session stream."""
        raw, _ = checked_pack([raw_trajectory])
        count = len(raw)
        if count > self.capacity:
            self._reserve(count)
        normalized = self.host_points[:count]
        normalize_points_into(raw, self.model.mean, self.model.std, normalized)
        if not np.isfinite(normalized).all():
            raise ValueError('Normalized raw geometry overflowed float32')
        self.host_offsets[:] = (0, count)
        packet_bytes = OFFSET_BYTES + POINT_BYTES * count
        self.device_packet[:packet_bytes].copy_(self.host_packet[:packet_bytes], non_blocking=True)
        return self.points[:count], count

    @torch.no_grad()
    def warmup(self, raw_trajectories, *, modes=('ranking', 'full')):
        """Explicit startup preparation; online misses never capture a graph."""
        if not modes or any(mode not in ('ranking', 'full') for mode in modes):
            raise ValueError('warmup modes must contain ranking or full')
        with self._lock:
            if self.plan is not None:
                self.plan.capture_enabled = True
            try:
                for raw in raw_trajectories:
                    for mode in modes:
                        self.search(raw, mode=mode)
            finally:
                if self.plan is not None:
                    self.plan.capture_enabled = False

    @torch.no_grad()
    def search(self, raw_trajectory, *, mode='full'):
        """Return owned CPU int64 Top-k IDs, with no cached query answers."""
        if mode not in ('ranking', 'full'):
            raise ValueError('mode must be ranking or full')
        with self._lock:
            self._validate()
            with torch.cuda.stream(self.stream):
                try:
                    points, count = self._prepare_input(raw_trajectory)
                    field_kernel[(64,)](
                        points,
                        self.offsets,
                        self.model.anchors,
                        self.fields,
                        64,
                        self.valid,
                        True,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
                    if mode == 'ranking' and self.config.distance == 'hausdorff':
                        tree = self.tree
                        tree_kernel[(1,)](
                            self.fields,
                            self.fields,
                            tree.d,
                            tree.members,
                            tree.tree,
                            self.offsets,
                            tree.database_scale,
                            self.result,
                            self.visits,
                            tree.groups,
                            tree.block,
                            self.top_k,
                            64,
                            True,
                            num_warps=4,
                            enable_fp_fusion=False,
                        )
                        ids = self.result
                    else:
                        # Full search and distance-specific sidecars keep their
                        # original validation barrier and exact native backend.
                        self.host_valid.copy_(self.valid, non_blocking=True)
                        self.stream.synchronize()
                        if not self.host_valid.numpy().all():
                            raise ValueError('Dual-field computation overflowed')
                        from leap.common.runtime.context import activate

                        context = (
                            activate(self.plan, count) if self.plan is not None else nullcontext()
                        )
                        with context:
                            query = self.model._finish_encoding(self.fields, points, self.offsets)
                            if mode == 'ranking':
                                ids = self.model.candidate_indices(
                                    query,
                                    self.database,
                                    candidate_k=self.top_k,
                                    search_index=self.tree,
                                )
                            else:
                                ids = self.model.search_range(
                                    query, self.database, top_k=self.top_k, search_index=self.tree
                                )['indices']
                    self.host_result.copy_(ids, non_blocking=True)
                    self.stream.synchronize()
                    result = self.host_result.numpy().copy()
                    if (result < 0).any():
                        raise ValueError('Dual-field computation overflowed')
                    return result
                except BaseException:
                    # A failed request cannot leave a pinned input in flight.
                    self.stream.synchronize()
                    raise
