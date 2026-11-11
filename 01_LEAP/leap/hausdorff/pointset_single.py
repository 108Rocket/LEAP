"""Owned serial near/far query session; warmed graphs never truncate inputs."""
import threading

import numpy as np
import torch

from leap.common.data import checked_pack
from leap.common.interval_index import lex_keys
from leap.hausdorff.pointset_plan import BufferedSingle, BufferedSingleFull, BufferedSingleRanking, SingleInput
from leap.hausdorff.pointset_search import dynamic, positive_k


class PointSetSession:
    def __init__(self,model,database,tree,top_k=20,initial_capacity=256):
        if model.config.query_profile!='single_query':
            raise ValueError('query_session requires query_profile=single_query')
        if model.config.hausdorff_fields!='near_far':raise ValueError('Expected near_far configuration')
        model._check_index(database);tree.validate(database)
        self.k=positive_k(top_k)
        if self.k>min(1000,len(database.coordinates)):raise ValueError('top_k exceeds available rows or1000')
        if positive_k(initial_capacity,'initial_capacity')<2:raise ValueError('initial_capacity must be>=2')
        self.model=model;self.database=database;self.tree=tree
        self.config=model.config;self.asset=model.asset_id
        self.signature=self._signature()
        self._lock=threading.RLock();self.stream=torch.cuda.Stream(device=model.device)
        self.stream.wait_stream(torch.cuda.current_stream(model.device))
        self.plan=None;self.ranking_plan=None;self.input=None
        self._reserve(initial_capacity)
        self.stream.wait_stream(torch.cuda.current_stream(model.device))

    def _signature(self):
        m=self.model
        return (m.anchors.data_ptr(),m.anchors._version,m.mean.tobytes(),m.std.tobytes())

    def _validate(self):
        if self.model.config!=self.config or self.model.asset_id!=self.asset or self._signature()!=self.signature:
            raise ValueError('Model changed; create a new query session')
        self.model._check_index(self.database);self.tree.validate(self.database)

    def _reserve(self,size):
        self.capacity=1<<(max(2,size)-1).bit_length()
        asset=dict(anchors=self.model.anchors.detach().cpu().numpy(),
                   geometry_mean=self.model.mean,geometry_std=self.model.std)
        self.input=SingleInput(self.database,asset,self.capacity)
        self.plan=None;self.ranking_plan=None

    def _fallback(self):
        if self.config.hausdorff_execution=='two_wave':
            answer=dynamic(self.input.q,self.database,self.tree)
        else:
            from leap.hausdorff.pointset_index import full
            answer=full(self.input.q,self.database,self.tree)
        return lex_keys(answer['distances'],answer['indices'])

    def _make_plan(self,capacity=2048):
        cls=BufferedSingle if self.config.hausdorff_execution=='two_wave' else BufferedSingleFull
        return cls(self.input.q,self.database,self.tree,anchors=self.input.anchors,capacity=capacity)

    @torch.no_grad()
    def warmup(self,raw_trajectories,*,modes=('ranking','full')):
        if not modes or any(m not in ('ranking','full') for m in modes):
            raise ValueError('warmup modes must contain ranking or full')
        with self._lock:
            for raw in raw_trajectories:
                for mode in modes:self.search_result(raw,mode=mode,_capture=True)

    @torch.no_grad()
    def search(self,raw_trajectory,*,mode='full'):
        return self.search_result(raw_trajectory,mode=mode)['indices']

    @torch.no_grad()
    def search_result(self,raw_trajectory,*,mode='full',_capture=False):
        """Return owned CPU arrays. Full includes nativeFP32 distances.

        Warmed ranking Top-k<=64 and Full Top20 use graphs. Other profiles
        retain the validated generic near/far target, never the polyline path.
        """
        if mode not in ('ranking','full'):raise ValueError('mode must be ranking or full')
        with self._lock:
            self._validate()
            with torch.cuda.stream(self.stream):
                try:
                    path,_=checked_pack([raw_trajectory])
                    if len(path)>self.capacity:
                        self.stream.synchronize();self._reserve(len(path))
                    self.input.load(path)
                    optimized=(mode=='full' and self.k==20 and len(self.database.coordinates)>=50
                               and self.config.initial_k in (None,50)
                               and self.config.initial_search!='full')
                    if (mode=='ranking' and self.k<=64 and self.config.single_query_engine=='specialized'
                            and self.config.initial_search!='full'):
                        if self.ranking_plan is None:
                            self.ranking_plan=BufferedSingleRanking(self.input.q,self.database,self.tree,
                                anchors=self.input.anchors,top_k=self.k)
                        if _capture and self.ranking_plan.graph is None:self.ranking_plan.warmup()
                        keys=self.ranking_plan.run(graph=self.ranking_plan.graph is not None)
                        ids=(keys.cpu().numpy()&0xffffffff).copy();scores=None
                    elif optimized:
                        if self.plan is None:
                            self.plan=self._make_plan()
                        if _capture and self.config.single_query_engine=='specialized' and self.plan.graph is None:
                            self.plan.warmup()
                        use_graph=self.config.single_query_engine=='specialized' and self.plan.graph is not None
                        keys=self.plan.run(self._fallback,graph=use_graph)
                        packed=keys.cpu().numpy().copy()
                        ids=(packed&0xffffffff).copy()
                        scores=(packed>>32).astype('int32').view('float32').copy()
                    else:
                        self.input.encode()
                        if not bool(torch.isfinite(self.input.q.coordinates).all()):
                            raise ValueError('Near/far query fields overflowed float32')
                        if mode=='ranking':
                            got=self.model.candidate_indices(self.input.q,self.database,self.k,search_index=self.tree)
                            ids=got.cpu().numpy().copy();scores=None
                        else:
                            answer=self.model.search_range(self.input.q,self.database,search_index=self.tree,top_k=self.k)
                            ids=answer['indices'].cpu().numpy().copy();scores=answer['distances'].cpu().numpy().copy()
                    self.stream.synchronize()
                    if ids.shape!=(1,self.k) or len(np.unique(ids))!=self.k or (ids<0).any() or (ids>=len(self.database.coordinates)).any():
                        raise RuntimeError('Retrieval violated result count or ID validity')
                    return dict(indices=ids,distances=scores)
                except BaseException:
                    self.stream.synchronize()
                    raise
