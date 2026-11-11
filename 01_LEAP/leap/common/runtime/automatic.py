"""Resolve a Full configuration once per session, never benchmark per request."""
import threading

import torch

from leap.common.runtime.configuration import configured_model


class AutoQuerySession:
    def __init__(self, model, database, tree, top_k, initial_capacity):
        if model.config.query_profile != 'single_query':
            raise ValueError('query_session requires query_profile=single_query')
        if isinstance(initial_capacity, bool) or not isinstance(initial_capacity, int) or initial_capacity < 2:
            raise ValueError('initial_capacity must be an integer >=2')
        self.source = model
        self.source_signature = self._signature()
        self.plan = model.config.resolve(len(database.coordinates), top_k, query_count=1)
        self.model = configured_model(model, self.plan)
        self.database, self.tree, self.top_k = database, tree, top_k
        self.model._check_index(database)
        tree.validate(database)
        self._lock = threading.RLock()
        # Keep the warmed small-k engines. Large-k uses the bounded batch API
        # on one query, not a claimed optimized single-query CUDA graph.
        native = not self.model.config.full_large_k or self.model.config.hausdorff_fields == 'near_far'
        self.native = (self.model.query_session(database, search_index=tree,
            top_k=top_k, initial_capacity=initial_capacity) if native else None)
        self.configuration = self.plan.as_dict()
        self.configuration['session_engine'] = 'native' if native else 'generic_full'

    def _signature(self):
        m = self.source
        return (m.config, m.asset_id, m.anchors.data_ptr(), m.anchors._version,
                m.mean.tobytes(), m.std.tobytes(), id(m.dtw_probes), id(m.dtw_router))

    def _validate(self):
        if self._signature() != self.source_signature:
            raise ValueError('Model changed; create a new query session')
        self.model._check_index(self.database)
        self.tree.validate(self.database)

    @torch.no_grad()
    def search(self, raw_trajectory, *, mode='full'):
        with self._lock:
            self._validate()
            if self.native is not None:
                return self.native.search(raw_trajectory, mode=mode)
            if mode != 'full':
                raise ValueError('Automatic large-k sessions support Full only')
            query = self.model.encode([raw_trajectory])
            result = self.model.search_range(query, self.database,
                search_index=self.tree, top_k=self.top_k)
            return result['indices'].cpu().numpy().copy()

    @torch.no_grad()
    def warmup(self, raw_trajectories, *, modes=('full',)):
        with self._lock:
            self._validate()
            if self.native is not None:
                return self.native.warmup(raw_trajectories, modes=modes)
            if not modes or any(mode != 'full' for mode in modes):
                raise ValueError('Automatic large-k sessions support Full only')
            for raw in raw_trajectories:
                self.search(raw)
