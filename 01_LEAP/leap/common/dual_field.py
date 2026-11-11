"""LEAP: Dual Distance Fields (DDF) and the GEF -> TBF -> RGR query API.

For probe c_j in theta, a_j(T)=dist(c_j,V_T), b_j(T)=dist(c_j,E_T).
max_j(b_j(B)-a_j(A), b_j(A)-a_j(B), 0) <= raw vertex/segment H(A,B)
in exact arithmetic. FP32 roundoff is audited, not used for certified pruning.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl
from leap.common.numeric import geometry_divide

from leap.common.model import TrajectoryIndex
from leap.common.raw import fused_raw_scores, normalize_points, normalize_points_serial
from leap.common.data import checked_pack, PackedTrajectories, NORMALIZATION_VERSION
from leap.common.config import RetrievalConfig
from leap.common.runtime.batching import resolve_query_batch
from leap.common.trajectory_bound_filter import cross_field_lower_bound

DEFAULT_DUAL_ASSET = Path(__file__).resolve().parents[1] / 'assets/porto_dual_field64_protected_e100.json'


def fields_torch(points, lengths, anchors):
    """eq:dual-fields and sec:fields encoding; padded differentiable reference.

    anchors is the compatible argument name for theta = {c_j}.
    """
    theta = anchors
    c_minus_p = theta[None,None]-points[:,:,None]
    valid = torch.arange(points.shape[1],device=points.device)[None] < lengths[:,None]
    a_T_squared = c_minus_p.square().sum(-1).masked_fill(~valid[:,:,None],torch.inf).amin(1)
    edge = points[:,1:]-points[:,:-1]
    c_minus_p = c_minus_p[:,:-1]
    # eq:projection / eq:segment-distance: t* = clip((c_j-u).e / ||e||^2, 0, 1).
    t_star = (c_minus_p*edge[:,:,None]).sum(-1)/edge.square().sum(-1)[:,:,None].clamp_min(1e-30)
    d = (c_minus_p-t_star.clamp(0,1)[...,None]*edge[:,:,None]).square().sum(-1)
    segment_valid = torch.arange(edge.shape[1],device=points.device)[None] < lengths[:,None]-1
    b_T_squared = d.masked_fill(~segment_valid[:,:,None],torch.inf).amin(1)
    # The floor defines a finite zero-distance gradient; max distance error1e-12.
    return torch.cat((a_T_squared.clamp_min(1e-24).sqrt(),b_T_squared.clamp_min(1e-24).sqrt()),dim=-1)


def score_torch(query, database):
    """Compatible entry point for TBF's L_Theta(q,T), eq:lower-bound."""
    return cross_field_lower_bound(query, database)


@triton.jit
def field_kernel(P,OFF,ANCHOR,OUT,M:tl.constexpr,VALID=None,WRITE_VALID:tl.constexpr=False):
    # eq:dual-fields: min_x ||c_j-x||, min_y ||c_j-y||.
    pair=tl.program_id(0)
    row=pair//M; probe_j=pair%M
    a0=tl.load(OFF+row); a1=tl.load(OFF+row+1)
    c_j_x=tl.load(ANCHOR+probe_j*2);c_j_y=tl.load(ANCHOR+probe_j*2+1)
    j=tl.arange(0,32)
    a_T_squared=tl.full((),float('inf'),P.dtype.element_ty)
    b_T_squared=tl.full((),float('inf'),P.dtype.element_ty)
    for start in range(a0,a1,32):
        valid=start+j<a1
        ax=tl.load(P+(start+j)*2,valid,other=0)
        ay=tl.load(P+(start+j)*2+1,valid,other=0)
        rx=c_j_x-ax;ry=c_j_y-ay
        a_T_squared=tl.minimum(a_T_squared,tl.min(tl.where(valid,rx*rx+ry*ry,float('inf')),0))
        seg=start+j+1<a1
        bx=tl.load(P+(start+j+1)*2,seg,other=0)
        by=tl.load(P+(start+j+1)*2+1,seg,other=0)
        ex=bx-ax;ey=by-ay
        t_star=geometry_divide(rx*ex+ry*ey,tl.maximum(ex*ex+ey*ey,1e-30))
        t_star=tl.minimum(1.,tl.maximum(0.,t_star))
        dx=rx-t_star*ex;dy=ry-t_star*ey
        b_T_squared=tl.minimum(b_T_squared,tl.min(tl.where(seg,dx*dx+dy*dy,float('inf')),0))
    a_value=tl.sqrt(tl.maximum(a_T_squared,1e-24))
    b_value=tl.sqrt(tl.maximum(b_T_squared,1e-24))
    tl.store(OUT+row*(2*M)+probe_j,a_value)
    tl.store(OUT+row*(2*M)+M+probe_j,b_value)
    if WRITE_VALID:
        tl.store(VALID+pair,(tl.abs(a_value)<float('inf'))&(tl.abs(b_value)<float('inf')))


@torch.no_grad()
def fields_packed(points,offsets,anchors):
    """sec:fields inline encoding: z_theta(T) = [a_1,...,a_M,b_1,...,b_M]."""
    out=torch.empty((len(offsets)-1,2*len(anchors)),device=points.device,dtype=points.dtype)
    field_kernel[((len(offsets)-1)*len(anchors),)](points,offsets,anchors,out,len(anchors),
                                                num_warps=4,enable_fp_fusion=False)
    return out


@triton.jit
def scan_kernel(Q,D,O,N:tl.constexpr,M:tl.constexpr,SAME:tl.constexpr):
    q=tl.program_id(0)
    j=tl.program_id(1)*16+tl.arange(0,16)
    f=tl.arange(0,32)
    L_qT=tl.full((16,),0,tl.float32)
    for start in range(0,M,32):
        a_q=tl.load(Q+q*(2*M)+start+f,start+f<M,other=0)
        b_q=tl.load(Q+q*(2*M)+M+start+f,start+f<M,other=0)
        a_T=tl.load(D+j[:,None]*(2*M)+start+f[None,:],(j[:,None]<N)&(start+f[None,:]<M),other=0)
        b_T=tl.load(D+j[:,None]*(2*M)+M+start+f[None,:],(j[:,None]<N)&(start+f[None,:]<M),other=0)
        L_qT=tl.maximum(L_qT,tl.max(tl.maximum(b_q[None,:]-a_T,b_T-a_q[None,:]),1))
    if SAME:
        L_qT=tl.where(q==j,float('inf'),L_qT)
    tl.store(O+q*N+j,L_qT,j<N)


@torch.no_grad()
def scan(queries,database,same=False):
    assert queries.shape[1]==database.shape[1] and queries.shape[1]%2==0
    assert queries.dtype==database.dtype==torch.float32 and queries.is_cuda
    queries,database=queries.contiguous(),database.contiguous()
    out=torch.empty((len(queries),len(database)),device=queries.device)
    scan_kernel[(len(queries),triton.cdiv(len(database),16))](queries,database,out,len(database),queries.shape[1]//2,same,num_warps=4)
    return out


class DualFieldRetriever:
    """Paper implementation: encode DDF, build GEF envelopes, run GEF/TBF/RGR.

    LEAP is the public alias; serialized asset keys remain unchanged.
    """
    def __init__(self,anchors,mean,std,device='cuda:0',batch_size=None,*,config=None,dtw_probes=None,routing_asset=None):
        self.config=RetrievalConfig() if config is None else config
        if not isinstance(self.config,RetrievalConfig):
            raise TypeError('config must be RetrievalConfig')
        if batch_size is None:
            batch_size=1000 if self.config.precision=='fp64' else 65536
        values=np.asarray(anchors,dtype=np.float32)
        if values.shape!=(64,2) or not np.isfinite(values).all():
            raise ValueError('Expected 64 finite two-dimensional anchors')
        mean,std=np.asarray(mean,dtype=np.float64),np.asarray(std,dtype=np.float64)
        if mean.shape!=(2,) or std.shape!=(2,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
            raise ValueError('Expected finite normalization and positive scales')
        if torch.device(device).type!='cuda' or int(batch_size)<1:
            raise ValueError('Retrieval requires CUDA and a positive batch size')
        self.anchors=torch.as_tensor(values.copy(),device=device).contiguous()
        self.mean=np.asarray(mean,dtype=np.float64).copy()
        self.std=np.asarray(std,dtype=np.float64).copy()
        self.device=self.anchors.device
        self.batch_size=int(batch_size)
        self.asset_id=hashlib.sha256(values.tobytes()+self.mean.tobytes()+self.std.tobytes()).hexdigest()+':dual-field:'+NORMALIZATION_VERSION
        if self.config.hausdorff_target=='pointset':
            self.asset_id+=':pointset'
        if self.config.hausdorff_fields=='near_far':
            self.asset_id+=':near-far-v1'
        self.dtw_probes=None
        self.dtw_router=None
        self._dtw_route_cache=None
        if self.config.distance=='dtw' and self.config.dtw_seed=='signal':
            from leap.dtw.dtw_probes import create_parameters, fingerprint, load_asset
            bundle=load_asset() if dtw_probes is None else dtw_probes
            self.dtw_probes=create_parameters(bundle,values,self.mean,self.std,self.device)
            self.asset_id+=':dtw-signal:'+fingerprint(self.dtw_probes)
        if routing_asset is not None and self.config.dtw_routing!='learned':
            raise ValueError('routing_asset requires learned DTW routing')
        if self.config.dtw_routing=='learned':
            from leap.dtw.dtw_routing import load_model, signature
            routing_args={} if routing_asset is None else {'path':routing_asset}
            self.dtw_router,router_hash=load_model(self.mean,self.std,self.device,**routing_args)
            self.dtw_router_signature=signature(self.dtw_router)
            self.asset_id+=':dtw-routing:'+router_hash
        if self.config.precision == 'fp64':
            self.asset_id += ':fp64'

    @property
    def theta(self):
        """Shared probe positions {c_j}, shape [M,2], sec:fields."""
        return self.anchors

    @classmethod
    def from_asset(cls,path=None,device='cuda:0',batch_size=None,*,config=None,routing_asset=None):
        signal=config is not None and config.distance=='dtw' and config.dtw_seed=='signal'
        if path is None:
            from leap.dtw.dtw_probes import DEFAULT_ASSET as signal_asset
            if config is not None and (config.large_k or config.hausdorff_fields=='near_far' or ((config.full_large_k or config.parameter_mode=='auto') and config.hausdorff_target=='pointset')):
                path=DEFAULT_DUAL_ASSET.parent / f'sdd_porto_{config.distance}64.json'
            else:
                path=signal_asset if signal else DEFAULT_DUAL_ASSET
        asset=json.loads(Path(path).read_text(encoding='utf-8'))
        if config is not None and (config.large_k or config.hausdorff_fields=='near_far' or ((config.full_large_k or config.parameter_mode=='auto') and config.hausdorff_target=='pointset')):
            expected='SDD point-set Hausdorff' if config.distance=='hausdorff' else 'dfd'
            if asset.get('supervision_distance')!=expected:
                raise ValueError('Large-k profile requires an explicitly matching city/distance asset')
        values=np.asarray(asset['anchors'],dtype=np.float32)
        if hashlib.sha256(values.tobytes()).hexdigest()!=asset['anchors_sha256']:
            raise ValueError('Anchor checksum mismatch')
        if signal and config.dtw_routing == 'learned' and routing_asset is None:
            paired=asset.get('default_routing_asset')
            if paired is not None:
                if (not isinstance(paired,str) or not paired or '/' in paired or '\\' in paired
                        or Path(paired).name!=paired or Path(paired).suffix!='.json'):
                    raise ValueError('Paired routing asset must be a local JSON filename')
                routing_asset=Path(path).resolve().with_name(paired)
        return cls(values,asset['geometry_mean'],asset['geometry_std'],device,batch_size,
                   config=config,dtw_probes=asset if signal else None,routing_asset=routing_asset)

    def _check_index(self,index):
        if (self.config.hausdorff_fields=='near_far')!=(':near-far-v1' in self.asset_id):
            raise ValueError('Field configuration changed; construct a new model and re-encode')
        if self.config.precision == 'fp64':
            from leap.common.precision import validate
            return validate(self, index)
        if self.config.dtw_routing=='learned':
            from leap.dtw.dtw_routing import signature
            if self.dtw_router is None or signature(self.dtw_router)!=self.dtw_router_signature:
                raise ValueError('DTW routing weights changed; construct a new model and re-encode')
        if not isinstance(index,TrajectoryIndex) or index.asset_id!=self.asset_id:
            raise ValueError('Index has different model, normalization or precision; re-encode and rebuild')
        if index.coordinates.ndim!=2 or index.coordinates.shape[1]!=128 or not len(index.coordinates):
            raise ValueError('Expected a nonempty 128-dimensional dual-field index')
        if len(index.offsets)!=len(index.coordinates)+1 or index.coordinates.dtype!=torch.float32:
            raise ValueError('Malformed dual-field index')
        if any(t.device!=self.device for t in (index.coordinates,index.points,index.offsets)):
            raise ValueError('Index and model must share a CUDA device')
        if self.config.distance=='dfd':
            from leap.dfd.dfd_geometry import signature
            summary=index.dfd_summary
            if (summary is None or set(summary.get('ids',{}))!={8,16,32} or
                    any(summary['ids'][s].shape!=(len(index.coordinates),s) for s in (8,16,32)) or
                    index.dfd_signature!=signature(index,summary)):
                raise ValueError('DFD retained geometry is missing or changed; re-encode')
        if self.config.distance=='dtw' and self.config.dtw_seed in ('ordered','weighted','signal'):
            from leap.dtw.ordered_dtw import signature
            summary=index.ordered_summary
            if (summary is None or summary.shape!=(len(index.coordinates),16) or
                    summary.dtype!=torch.float32 or summary.device!=self.device or
                    index.ordered_signature!=signature(index.points,index.offsets,summary)):
                raise ValueError('Ordered summary is missing or changed; re-encode raw geometry')
            if self.config.dtw_seed=='signal':
                from leap.dtw.dtw_probes import validate_index
                if self.dtw_probes is None:
                    raise ValueError('Signal mode needs a matching probe asset; construct a new model')
                validate_index(index,self.dtw_probes)

    def _check_search_engine(self, tree):
        if tree is not None and getattr(tree, 'search_engine', 'original') != self.config.search_engine:
            raise ValueError('Search engine changed; rebuild the search index and query session')

    def build_search_index(self,database,leaf_capacity=64):
        """Build once after encoding the database, not once per query."""
        if self.config.hausdorff_fields=='near_far':
            from leap.hausdorff.pointset_index import SearchTree
            self._check_index(database)
            if leaf_capacity!=64:
                raise ValueError('near_far uses the validated64-member tree leaves')
            return SearchTree(database, fast=self.config.search_engine == 'fast')
        if self.config.precision == 'fp64':
            from leap.common.precision import PrecisionSearchIndex
            self._check_index(database)
            return PrecisionSearchIndex(self, database, leaf_capacity)
        if self.config.large_k:
            raise ValueError('large_k uses a fused full scan, not a summary tree')
        from leap.common.interval_index import IntervalSearchIndex
        self._check_index(database)
        tree=IntervalSearchIndex(database,leaf_capacity)
        tree.search_engine=self.config.search_engine
        if self.config.distance=='dfd':
            from leap.dfd.dfd_tree import prepare
            tree.dfd_geometry=prepare(database,tree)
            tree.dfd_versions=database.dfd_signature
        if self.config.distance=='dtw':
            from leap.dtw.dtw_range import sidecars
            tree.dtw_geometry=sidecars(database,tree)
            tree.dtw_versions=(database.points.data_ptr(),database.points._version,
                               database.offsets.data_ptr(),database.offsets._version)
            if self.config.dtw_seed=='signal':
                from leap.dtw.dtw_radial import node_envelopes
                tree.dtw_probe_tree=node_envelopes(database.dtw_probe_summary['radial'],tree)
                tree.dtw_probe_versions=database.dtw_probe_signature
                pool=self.config.dtw_pool if self.config.initial_pool is None else self.config.initial_pool
                from leap.common.candidate_policy import MAX_TREE_CANDIDATES
                if self.config.dtw_execution=='optimized' and pool<=MAX_TREE_CANDIDATES:
                    tree.dtw_routing_tree=self._routing_tree(database)
        return tree

    @torch.no_grad()
    def encode(self,paths):
        if isinstance(paths, PackedTrajectories):
            return self.encode_packed(paths.points, paths.offsets)
        if self.config.precision == 'fp64':
            from leap.common.precision import encode
            return encode(self, paths)
        return self.encode_packed(*checked_pack(paths))

    @torch.no_grad()
    def encode_packed(self, points, offsets):
        """Encode original coordinates directly, including validation and transfers."""
        if self.config.precision == 'fp64':
            from leap.common.precision import encode
            return encode(self, PackedTrajectories(points, offsets))
        from leap.common.encoding import PackedEncoder
        return PackedEncoder(self).encode(points, offsets)

    def packed_encoder(self):
        """Create a serial FP32 encoder with reusable private host staging."""
        from leap.common.encoding import PackedEncoder
        return PackedEncoder(self)

    def query_session(self,database,*,search_index,top_k=20,initial_capacity=256):
        """Serial request session with private, reusable I/O buffers."""
        self._check_search_engine(search_index)
        if self.config.parameter_mode=='auto':
            from leap.common.runtime.automatic import AutoQuerySession
            return AutoQuerySession(self,database,search_index,top_k,initial_capacity)
        if self.config.hausdorff_fields=='near_far':
            from leap.hausdorff.pointset_single import PointSetSession
            return PointSetSession(self,database,search_index,top_k,initial_capacity)
        if self.config.precision == 'fp64':
            from leap.common.precision import PrecisionQuerySession
            return PrecisionQuerySession(self,database,search_index,top_k,initial_capacity)
        from leap.common.runtime.session import SingleQuerySession
        return SingleQuerySession(self,database,search_index,top_k,initial_capacity)

    def _finish_encoding(self,z_theta,p,o):
        # Shared by normal encoding and session-owned single-query storage.
        from leap.common.runtime.context import current
        plan = current()
        if plan is not None and self.config.distance == 'dtw':
            return plan.encode_dtw(z_theta, p, o)
        if self.config.distance=='dtw' and self.config.dtw_seed in ('ordered','weighted','signal'):
            from leap.dtw.ordered_dtw import summarize, signature
            if self.config.dtw_routing=='learned':
                from leap.dtw.dtw_routing import encode_index
                summary=encode_index(self.dtw_router,TrajectoryIndex(z_theta,p,o,self.asset_id))
            else:
                summary=summarize(p,o)
            if not torch.isfinite(summary).all():
                raise ValueError('Ordered summary overflowed')
            index=TrajectoryIndex(z_theta,p,o,self.asset_id,summary,signature(p,o,summary))
            if self.config.dtw_seed=='signal':
                from dataclasses import replace
                from leap.dtw.dtw_probes import encode, signature as probe_signature
                cached=encode(index,self.dtw_probes)
                return replace(index,dtw_probe_summary=cached,
                               dtw_probe_signature=probe_signature(index,cached,self.dtw_probes))
            return index
        if self.config.distance=='dfd':
            from leap.dfd.dfd_geometry import prepare, signature
            index=TrajectoryIndex(z_theta,p,o,self.asset_id)
            summary=prepare(index)
            return TrajectoryIndex(z_theta,p,o,self.asset_id,dfd_summary=summary,
                                   dfd_signature=signature(index,summary))
        return TrajectoryIndex(z_theta,p,o,self.asset_id)

    def _dfd_candidates(self,queries,database,tree,qids,k):
        from leap.dfd.dfd_tree import initial_candidates
        if getattr(tree,'dfd_versions',None)!=database.dfd_signature:
            raise ValueError('DFD tree has missing or stale geometry; rebuild')
        return initial_candidates(queries,database,tree,qids,k,
                                  pool=self.config.initial_pool,strategy=self.config.dfd_initial)

    def _routing_tree(self,database):
        from leap.dtw.dtw_routing_tree import RoutingTree
        if (self._dtw_route_cache is None or self._dtw_route_cache.database is not database
                or self._dtw_route_cache.fast != (self.config.search_engine == 'fast')):
            self._dtw_route_cache=RoutingTree(database, fast=self.config.search_engine == 'fast')
        return self._dtw_route_cache

    def _dtw_candidates(self,queries,database,k,qids,*,routing_tree=None,candidate_search='auto'):
        pool=self.config.dtw_pool if self.config.initial_pool is None else self.config.initial_pool
        if self.config.dtw_seed=='signal':
            from leap.dtw.dtw_probes import candidates
            available=len(database.coordinates)-int(qids is not None)
            width=min(available,max(pool,k))
            from leap.common.candidate_policy import MAX_TREE_CANDIDATES, tree_initializer
            use_tree=(candidate_search=='tree' or (candidate_search=='auto' and
                       tree_initializer(self.config,width)))
            if use_tree:
                if width>MAX_TREE_CANDIDATES:
                    raise ValueError('DTW routing tree supports pools <=8192; use auto or full for larger pools')
                routing_tree=self._routing_tree(database) if routing_tree is None else routing_tree
            else:
                routing_tree=None
            return candidates(queries,database,k,qids,pool=pool,
                              routing_tree=routing_tree,bucket=self.config.dtw_execution=='optimized')
        if candidate_search=='tree':
            raise ValueError('DTW routing tree requires signal ranking')
        if self.config.dtw_seed=='weighted':
            from leap.dtw.weighted_dtw import candidates
            return candidates(queries,database,k,qids,pool=pool)
        if self.config.dtw_seed=='ordered':
            from leap.dtw.ordered_dtw import candidates
        else:
            from leap.dtw.dtw import pse_candidates as candidates
        return candidates(queries,database,k,qids)

    @torch.no_grad()
    def candidate_indices(self,queries,database=None,candidate_k=None,*,search_index=None,candidate_search='auto'):
        """Configured coarse candidates; no original-distance or teacher lookup.

        Hausdorff auto uses a supplied summary tree for K <= 8192, independently
        of database or query-batch size. Use full for an explicit scan control.
        """
        self._check_search_engine(search_index)
        if self.config.parameter_mode=='auto' and candidate_k is not None:
            from leap.common.runtime.configuration import configured_model
            same=database is None or database is queries
            db=queries if database is None else database
            plan=self.config.resolve(len(db.coordinates),candidate_k,
                query_count=len(queries.coordinates),exclude_self=same)
            return configured_model(self,plan).candidate_indices(queries,database,candidate_k,
                search_index=search_index,candidate_search=candidate_search)
        if self.config.hausdorff_fields=='near_far':
            from leap.hausdorff.pointset_search import candidates
            return candidates(self,queries,database,candidate_k,search_index,candidate_search)
        if self.config.precision == 'fp64':
            from leap.common.precision import ranking_model
            self._check_index(queries)
            if database is not None:
                self._check_index(database)
            if search_index is not None:
                search_index.validate(queries if database is None else database)
            return ranking_model(self).candidate_indices(queries.ranking_index,
                None if database is None else database.ranking_index,candidate_k,
                search_index=None if search_index is None else search_index.ranking_tree,
                candidate_search=candidate_search)
        if self.config.large_k:
            if search_index is not None or candidate_search not in ('auto','full'):
                raise ValueError('large_k ranking uses a full scan without a tree')
            return self.search_large_k(queries,database,mode='ranking',
                top_k=1000 if candidate_k is None else candidate_k)['indices']
        from leap.common.interval_index import reference
        same=database is None or database is queries
        if candidate_k is None:
            candidate_k=32 if self.config.distance=='dfd' and self.config.dfd_initial=='light' else 50
        database=queries if database is None else database
        self._check_index(queries)
        self._check_index(database)
        if candidate_search not in ('auto','full','tree'):
            raise ValueError('candidate_search must be auto, full or tree')
        available=len(database.coordinates)-int(same)
        candidate_k=min(int(candidate_k),available)
        if candidate_k<1:
            raise ValueError('At least one candidate is required')
        if search_index is not None:
            search_index.validate(database)
        if self.config.distance=='dfd':
            if search_index is None:
                raise ValueError('DFD candidates require a built search_index')
            qids=(torch.arange(len(queries.coordinates),device=self.device) if same else
                  torch.full((len(queries.coordinates),),-1,device=self.device,dtype=torch.int64))
            from leap.common.candidate_policy import MAX_TREE_CANDIDATES
            pool=self.config.initial_pool
            width=max(candidate_k,64 if self.config.dfd_initial=='light' else 200) if pool is None else pool
            dense=candidate_search=='full' or self.config.initial_search=='full' or width>MAX_TREE_CANDIDATES
            if dense or candidate_k>64 or width>256:
                if candidate_search=='tree' and dense:
                    raise ValueError('DFD tree candidate pool exceeds capacity or conflicts with full search')
                if self.config.dfd_initial!='light':
                    raise ValueError('Wide DFD ranking supports the light strategy')
                from copy import copy
                from dataclasses import replace
                from leap.common.full_large_k import initial_candidates, subset, MAX_SCORE_PAIRS
                candidate_model=copy(self)
                candidate_model.config=replace(self.config,initial_pool=width,
                    initial_search='full' if dense else self.config.initial_search)
                batch_options=dict(default=128,database_size=len(database.coordinates),
                    max_score_pairs=MAX_SCORE_PAIRS) if dense else {}
                batch=resolve_query_batch(self.config,**batch_options)
                # Preserve the wide path's final FP32 scoring and tie order.
                return torch.cat([initial_candidates(candidate_model,subset(queries,lo,lo+batch),database,
                    qids[lo:lo+batch],candidate_k,lo if same else None,
                    search_index=None if dense else search_index)
                    for lo in range(0,len(qids),batch)])
            from leap.dfd.dfd_geometry import subset
            chunks=[self._dfd_candidates(subset(queries,lo,min(lo+512,len(qids))),
                    database,search_index,qids[lo:lo+512],candidate_k)
                    for lo in range(0,len(qids),512)]
            return (torch.cat(chunks) if chunks else
                    torch.empty((0,candidate_k),device=self.device,dtype=torch.int64))
        if self.config.distance=='dtw':
            qids=torch.arange(len(queries.coordinates),device=self.device) if same else None
            return self._dtw_candidates(queries,database,candidate_k,qids,
                routing_tree=getattr(search_index,'dtw_routing_tree',None),candidate_search=candidate_search)
        from leap.common.candidate_policy import MAX_TREE_CANDIDATES, tree_initializer
        use_tree=candidate_search=='tree' or (
            candidate_search=='auto' and tree_initializer(self.config,candidate_k,has_index=search_index is not None))
        if same:
            qids=torch.arange(len(queries.coordinates),device=self.device)
        else:
            qids=torch.full((len(queries.coordinates),),-1,device=self.device,dtype=torch.int64)
        if use_tree:
            if search_index is None or candidate_k>MAX_TREE_CANDIDATES:
                raise ValueError('Tree search requires a built index and candidate_k <= 8192')
            return search_index.candidates(queries.coordinates,qids,candidate_k)[0]
        keys=reference(queries.coordinates,database.coordinates,qids,candidate_k)
        return (keys & 0xffffffff).contiguous()

    @torch.no_grad()
    def search_range(self,queries,database=None,*,search_index=None,seed_k=None,top_k=20,query_batch=None,initial_k=None,
                     reuse_initial=None,ragged_dtw=None,progress=None):
        """Expand a raw-derived radius; FP32 slack is empirical, not certified.

        Query/database identity excludes self. Separately encoded queries are
        external, even if their coordinates equal database trajectories.
        Build the reusable tree explicitly before calling this method.
        initial_k is s = |S(q)|; seed_k remains a compatible spelling.
        """
        self._check_search_engine(search_index)
        if self.config.parameter_mode=='auto':
            from leap.common.runtime.configuration import search
            return search(self,queries,database,search_index=search_index,seed_k=seed_k,
                top_k=top_k,query_batch=query_batch,initial_k=initial_k,
                reuse_initial=reuse_initial,ragged_dtw=ragged_dtw,progress=progress)
        if seed_k is None and initial_k is None:
            seed_k=self.config.initial_k
        if self.config.hausdorff_fields=='near_far':
            from leap.hausdorff.pointset_search import search
            if initial_k is not None and seed_k is not None and initial_k!=seed_k:
                raise ValueError('initial_k and seed_k specify different candidate counts')
            if reuse_initial is not None or ragged_dtw is not None:
                raise ValueError('near_far uses its own exact execution policy')
            return search(self,queries,database,search_index,top_k,
                initial_k if initial_k is not None else seed_k,query_batch,progress)
        if self.config.precision == 'fp64':
            from leap.common.precision import search
            if initial_k is not None and seed_k is not None and initial_k != seed_k:
                raise ValueError('initial_k and seed_k specify different candidate counts')
            return search(self,queries,database,search_index=search_index,
                seed_k=initial_k if initial_k is not None else seed_k,top_k=top_k,
                query_batch=query_batch,progress=progress)
        if self.config.full_large_k:
            if initial_k is not None and seed_k is not None and initial_k!=seed_k:
                raise ValueError('initial_k and seed_k specify different candidate counts')
            if reuse_initial is False or ragged_dtw is False:
                raise ValueError('full_large_k keeps seed reuse and bounded ragged DTW enabled')
            from leap.common.full_large_k import search
            return search(self,queries,database,search_index=search_index,top_k=top_k,
                          seed_k=initial_k if initial_k is not None else seed_k,
                          query_batch=query_batch,progress=progress)
        if progress is not None:
            raise ValueError('progress callbacks require full_large_k')
        if self.config.initial_search=='full':
            raise ValueError('Explicit scan initialization requires full_large_k=True or parameter_mode=auto')
        if top_k>64:
            raise ValueError('Full search above64 requires the explicit full_large_k=True optimization switch')
        if self.config.large_k:
            raise ValueError('large_k is candidate-limited, not full-database range search')
        from leap.hausdorff.range_search import search_chunk
        if initial_k is not None:
            if seed_k is not None and initial_k != seed_k:
                raise ValueError('initial_k and seed_k specify different candidate counts')
            seed_k=initial_k
        same=database is None or database is queries
        database=queries if database is None else database
        self._check_index(queries)
        self._check_index(database)
        if search_index is None:
            raise ValueError('Range search requires a built search_index')
        search_index.validate(database)
        optimized=self.config.distance=='dtw' and self.config.dtw_execution=='optimized'
        reuse_initial=optimized if reuse_initial is None else reuse_initial
        ragged_dtw=optimized if ragged_dtw is None else ragged_dtw
        query_batch=resolve_query_batch(self.config,query_batch)
        if seed_k is None:
            weighted=self.config.distance=='dtw' and self.config.dtw_seed in ('weighted','signal')
            light=self.config.distance=='dfd' and self.config.dfd_initial=='light'
            seed_k=max(20,int(top_k)) if weighted else (32 if light else 50)
        seed_k,top_k,query_batch=int(seed_k),int(top_k),int(query_batch)
        if not 1<=top_k<=seed_k<=64 or query_batch<1:
            raise ValueError('Require 1 <= top_k <= seed_k <= 64 and query_batch >= 1')
        seed_k=min(seed_k,len(database.coordinates)-int(same))
        if seed_k<top_k:
            raise ValueError('Database has fewer available trajectories than top_k')
        if self.config.distance=='dtw':
            from leap.dtw.dtw_range import sidecars
            versions=(database.points.data_ptr(),database.points._version,
                      database.offsets.data_ptr(),database.offsets._version)
            if getattr(search_index,'dtw_versions',None)!=versions:
                raise ValueError('DTW tree is missing geometry or its raw points changed; rebuild')
            if self.config.dtw_seed=='signal' and getattr(search_index,'dtw_probe_versions',None)!=database.dtw_probe_signature:
                raise ValueError('DTW tree has missing or stale probe envelopes; rebuild')
            query_geometry=search_index.dtw_geometry if same else sidecars(queries)
        outputs=[]
        initial_cache=None
        if self.config.distance=='dtw' and reuse_initial:
            all_ids=torch.arange(len(queries.coordinates),device=self.device) if same else None
            initial_cache=self._dtw_candidates(queries,database,seed_k,all_ids,
                routing_tree=getattr(search_index,'dtw_routing_tree',None))
        def process(lo,hi,dfd_cache=None,dtw_cache=None):
            q=TrajectoryIndex(queries.coordinates[lo:hi],queries.points,
                              queries.offsets[lo:hi+1],queries.asset_id,
                              None if queries.ordered_summary is None else queries.ordered_summary[lo:hi])
            if self.config.distance=='dtw' and self.config.dtw_seed=='signal':
                from dataclasses import replace
                from leap.dtw.dtw_probes import slice_summary
                q=replace(q,dtw_probe_summary=slice_summary(queries.dtw_probe_summary,lo,hi))
            if self.config.distance=='dfd':
                summary={'ids':{s:v[lo:hi] for s,v in queries.dfd_summary['ids'].items()},
                         'block':queries.dfd_summary['block']}
                q=TrajectoryIndex(q.coordinates,q.points,q.offsets,q.asset_id,dfd_summary=summary)
            ids=(torch.arange(lo,hi,device=self.device) if same else
                 torch.full((hi-lo,),-1,device=self.device,dtype=torch.int64))
            try:
                if self.config.distance=='dtw':
                    from leap.dtw.dtw_range import search as dtw_search, initial_distances
                    from leap.common.runtime import options as opt
                    if dtw_cache is None:
                        S_q=(initial_cache[lo:hi].contiguous() if initial_cache is not None else
                             self._dtw_candidates(q,database,seed_k,ids if same else None,
                                routing_tree=getattr(search_index,'dtw_routing_tree',None)))
                        if hi-lo>1 and opt.dtw_initial_cache:
                            dtw_cache=(S_q,initial_distances(q,database,S_q,bucket=optimized,
                                maximum=search_index.dtw_geometry['block'] if opt.dtw_length_cache else None))
                    else:
                        S_q=dtw_cache[0]
                    geometry={'box':query_geometry['box'][lo:hi], 'block':query_geometry['block']}
                    options={}
                    if self.config.dtw_seed=='signal':
                        a,b=q.dtw_probe_summary,database.dtw_probe_summary
                        options={'probe_fields':(a['radial'],b['radial']),
                                 'probe_tree':search_index.dtw_probe_tree,
                                 'ordered_fields':(a['ordered'],b['ordered'])}
                    options['ragged']=ragged_dtw
                    options['dynamic_threshold']=self.config.dynamic_threshold
                    if optimized:
                        options.update(bounded_dtw=True,bucket_dtw=True,one_pass_capacity=2048,parallel_leaves=True)
                    if dtw_cache is not None:
                        options['raw_initial']=dtw_cache[1]
                    result=dtw_search(q,database,search_index,search_index.dtw_geometry,geometry,ids,S_q,top_k,**options)
                elif self.config.distance=='dfd':
                    from leap.dfd.dfd_range import search as dfd_search, initial_distances
                    if dfd_cache is None:
                        S_q=self._dfd_candidates(q,database,search_index,ids,seed_k)
                        if hi-lo>1:
                            dfd_cache=(S_q,initial_distances(q,database,S_q))
                    else:
                        S_q=dfd_cache[0]
                    options={} if dfd_cache is None else {'raw_initial':dfd_cache[1]}
                    options['dynamic_threshold']=self.config.dynamic_threshold
                    result=dfd_search(q,database,search_index,ids,S_q,top_k,**options)
                else:
                    result=search_chunk(q,database,search_index,ids,seed_k,top_k)
            except MemoryError:
                if hi-lo<=1:
                    raise
                mid=(lo+hi)//2
                # A buffer retry changes only row grouping, not seeds or their distances.
                left=None if dfd_cache is None else tuple(v[:mid-lo] for v in dfd_cache)
                right=None if dfd_cache is None else tuple(v[mid-lo:] for v in dfd_cache)
                dtw_left=None if dtw_cache is None else tuple(v[:mid-lo] for v in dtw_cache)
                dtw_right=None if dtw_cache is None else tuple(v[mid-lo:] for v in dtw_cache)
                process(lo,mid,left,dtw_left)
                process(mid,hi,right,dtw_right)
                return
            outputs.append({k:result[k] for k in ('indices','distances','counts','radius')})
        for lo in range(0,len(queries.coordinates),query_batch):
            process(lo,min(lo+query_batch,len(queries.coordinates)))
        if not outputs:
            return {'indices':torch.empty((0,top_k),device=self.device,dtype=torch.int64),
                    'distances':torch.empty((0,top_k),device=self.device),
                    'counts':torch.empty(0,device=self.device,dtype=torch.int32),
                    'radius':torch.empty(0,device=self.device)}
        return {k:torch.cat([r[k] for r in outputs]) for k in outputs[0]}

    @torch.no_grad()
    def search_large_k(self,queries,database=None,*,mode='refine',top_k=1000,candidate_k=None,
                       query_batch=None,execution='fused'):
        """Opt-in SDD large-k ranking/refinement; never a full-search certificate."""
        if self.config.hausdorff_fields=='near_far':
            raise ValueError('near_far uses candidate_indices/search/search_range, not the legacy large_k API')
        from leap.common.large_k import search
        return search(self,queries,database,mode=mode,top_k=top_k,candidate_k=candidate_k,
                      query_batch=query_batch,execution=execution)

    @torch.no_grad()
    def search(self,queries,database=None,candidate_k=None,top_k=None,*,search_index=None,candidate_search='auto',refinement='auto'):
        """Raw Top-K within coarse candidates, or expanded search in range mode."""
        if refinement not in ('auto','full'):
            raise ValueError('refinement must be auto or full')
        if self.config.parameter_mode=='auto':
            if candidate_search not in ('auto','range'):
                raise ValueError('Automatic configuration uses Full search; use candidate_indices for ranking')
            return self.search_range(queries,database,search_index=search_index,
                seed_k=candidate_k,top_k=20 if top_k is None else top_k)
        if self.config.hausdorff_fields=='near_far':
            if self.config.full_large_k or candidate_search=='range':
                if candidate_search not in ('auto','range'):
                    raise ValueError('Full search requires auto or range')
                return self.search_range(queries,database,search_index=search_index,
                    seed_k=candidate_k,top_k=(1000 if self.config.full_large_k else 20) if top_k is None else top_k)
            from leap.hausdorff.pointset_search import refine
            return refine(self,queries,database,candidate_k,20 if top_k is None else top_k,search_index,candidate_search)
        if self.config.precision == 'fp64':
            from leap.common.precision import refine_candidates
            if self.config.full_large_k and candidate_search not in ('auto','range'):
                raise ValueError('full_large_k uses global range search, not candidate-pool retrieval')
            if self.config.full_large_k or candidate_search == 'range':
                return self.search_range(queries,database,search_index=search_index,seed_k=candidate_k,
                    top_k=(1000 if self.config.full_large_k else 20) if top_k is None else top_k)
            return refine_candidates(self,queries,database,search_index,candidate_k,
                20 if top_k is None else top_k,candidate_search)
        if self.config.full_large_k:
            if candidate_search not in ('auto','range'):
                raise ValueError('full_large_k uses global range search, not candidate-pool retrieval')
            return self.search_range(queries,database,search_index=search_index,
                                     seed_k=candidate_k,top_k=1000 if top_k is None else top_k)
        if self.config.large_k:
            if search_index is not None or candidate_search not in ('auto','full'):
                raise ValueError('large_k refinement uses a fixed pool without tree/range search')
            return self.search_large_k(queries,database,candidate_k=candidate_k,
                                      top_k=1000 if top_k is None else top_k)
        top_k=20 if top_k is None else top_k
        if candidate_search=='range':
            return self.search_range(queries,database,search_index=search_index,
                                     seed_k=candidate_k,top_k=top_k)
        if candidate_k is None:
            candidate_k=32 if self.config.distance=='dfd' and self.config.dfd_initial=='light' else 50
        ids=self.candidate_indices(queries,database,candidate_k,search_index=search_index,candidate_search=candidate_search)
        database=queries if database is None else database
        top_k=int(top_k)
        if not 1<=top_k<=ids.shape[1]:
            raise ValueError('Require 1 <= top_k <= candidate_k')
        if self.config.distance in ('dfd','dtw'):
            if self.config.distance=='dfd':
                from leap.dfd.dfd import dfd_pairs as distance_pairs
            else:
                from leap.dtw.dtw import dtw_pairs as distance_pairs
            from leap.common.interval_index import lex_keys
            rows=torch.arange(len(ids),device=self.device)[:,None].expand_as(ids)
            scores=distance_pairs(queries,database,rows.reshape(-1),ids.reshape(-1)).reshape_as(ids)
            keys=lex_keys(scores,ids).topk(top_k,largest=False).values
            return {'indices':keys&0xffffffff,'distances':(keys>>32).int().contiguous().view(torch.float32)}
        if refinement=='auto' and top_k==20 and ids.shape[1]>32:
            from leap.common.early_refinement import early_refine
            result=early_refine(queries,database,ids,seed=32,source_block=8,striped=True)
            return {k:result[k] for k in ('indices','distances')}
        values,order=fused_raw_scores(queries,database,ids).topk(top_k,largest=False)
        return {'indices':ids.gather(1,order),'distances':values}


# Keep the historical class identity for saved Python objects and older scripts.
LEAP = DualFieldRetriever
