"""Retrieval target; the default Hausdorff pipeline is unchanged."""
from dataclasses import dataclass
import json
from pathlib import Path


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class TopKProfile:
    """Optional execution overrides for one returned-neighbor count, not recall."""
    top_k: int
    initial_k: int | None = None
    initial_pool: int | None = None
    query_batch: int | None = None

    def __post_init__(self):
        _positive(self.top_k, 'top_k')
        if self.top_k > 1000:
            raise ValueError('Automatic profiles support top_k in [1,1000]')
        for name in ('initial_k', 'initial_pool', 'query_batch'):
            value = getattr(self, name)
            if value is not None:
                _positive(value, name)
        if self.initial_k is not None and self.initial_k < self.top_k:
            raise ValueError('initial_k must cover top_k')


@dataclass(frozen=True)
class RetrievalConfig:
    """Target distance D and initial-ranking settings.

    dtw_seed is the persisted name of the DTW initial-ranking mode.
    It does not denote a random seed. dtw_pool is C after eq:dtw-seed-scan.
    dtw_routing selects fixed bins or learned mass in eq:dtw-routing-summary.
    """
    distance: str = 'hausdorff'
    dtw_seed: str = 'weighted'
    dtw_pool: int = 128
    dfd_initial: str = 'light'
    dtw_routing: str = 'fixed'
    dtw_execution: str = 'optimized'
    large_k: bool = False
    hausdorff_target: str = 'polyline'
    full_large_k: bool = False
    query_profile: str = 'throughput'
    single_query_engine: str = 'specialized'
    query_batch: int | None = None
    hausdorff_fields: str = 'default'
    hausdorff_execution: str = 'auto'
    precision: str = 'fp32'
    parameter_mode: str = 'manual'
    top_k_profiles: tuple[TopKProfile, ...] = ()
    initial_k: int | None = None
    initial_pool: int | None = None
    initial_search: str = 'auto'
    search_engine: str = 'original'
    dynamic_threshold: bool = False

    def __post_init__(self):
        if not isinstance(self.dynamic_threshold, bool):
            raise ValueError('dynamic_threshold must be a boolean')
        if self.search_engine not in ('original', 'fast'):
            raise ValueError('search_engine must be original or fast')
        if self.search_engine == 'fast':
            if self.precision != 'fp32' or self.large_k:
                raise ValueError('fast search requires FP32 without legacy large_k')
            if self.distance == 'hausdorff' and (
                    self.hausdorff_fields != 'near_far' or self.hausdorff_execution != 'auto'):
                raise ValueError('fast Hausdorff requires near_far fields and ordinary Full')
            if self.distance == 'dtw' and (
                    self.dtw_seed != 'signal' or self.dtw_execution != 'optimized'):
                raise ValueError('fast DTW requires signal ranking and optimized execution')
        if self.initial_search not in ('auto', 'tree', 'full'):
            raise ValueError('initial_search must be auto, tree or full')
        if self.parameter_mode not in ('manual', 'auto'):
            raise ValueError('parameter_mode must be manual or auto')
        for name in ('initial_k', 'initial_pool'):
            if getattr(self, name) is not None:
                _positive(getattr(self, name), name)
        profiles = self.top_k_profiles
        if isinstance(profiles, dict):
            rows = []
            for key, value in profiles.items():
                if isinstance(key, bool) or not (isinstance(key, int) or
                        isinstance(key, str) and key.isascii() and key.isdecimal()):
                    raise ValueError('Profile keys must be integer Top-k values')
                if not isinstance(value, dict) or 'top_k' in value:
                    raise ValueError('Profile values contain execution parameters only')
                rows.append(dict(value, top_k=int(key)))
            profiles = rows
        if not isinstance(profiles, (list, tuple)):
            raise ValueError('top_k_profiles must be a mapping or sequence')
        try:
            profiles = tuple(p if isinstance(p, TopKProfile) else TopKProfile(**p) for p in profiles)
        except TypeError as error:
            raise ValueError('Unknown or malformed Top-k profile fields') from error
        if len({p.top_k for p in profiles}) != len(profiles):
            raise ValueError('Duplicate Top-k profiles')
        object.__setattr__(self, 'top_k_profiles', tuple(sorted(profiles, key=lambda p: p.top_k)))
        if profiles and self.parameter_mode != 'auto':
            raise ValueError('top_k_profiles require parameter_mode=auto')
        if self.parameter_mode == 'auto' and self.large_k:
            raise ValueError('Automatic configuration uses Full, not legacy large_k refinement')
        if self.parameter_mode == 'auto' and self.distance == 'dtw' and (
                self.dtw_seed != 'signal' or self.dtw_execution != 'optimized'):
            raise ValueError('Automatic DTW requires signal ranking and optimized execution')
        if self.distance == 'hausdorff' and (self.initial_pool is not None or
                any(p.initial_pool is not None for p in profiles)):
            raise ValueError('Hausdorff has initial_k but no separate initial_pool')
        if self.hausdorff_fields not in ('default', 'near_far'):
            raise ValueError('hausdorff_fields must be default or near_far')
        if self.hausdorff_execution not in ('auto', 'two_wave'):
            raise ValueError('hausdorff_execution must be auto or two_wave')
        if self.hausdorff_fields == 'near_far' and (
                self.distance != 'hausdorff' or self.hausdorff_target != 'pointset' or self.large_k):
            raise ValueError('near_far requires point-set Hausdorff without legacy large_k')
        if self.hausdorff_fields == 'near_far' and getattr(self, 'precision', 'fp32') != 'fp32':
            raise ValueError('near_far execution is validated for FP32 only')
        if self.hausdorff_execution != 'auto' and self.hausdorff_fields != 'near_far':
            raise ValueError('two_wave execution requires near_far fields')
        if self.query_batch is not None:
            from leap.common.runtime.batching import positive_batch
            positive_batch(self.query_batch)
        if self.precision not in ('fp32', 'fp64'):
            raise ValueError('precision must be fp32 or fp64')
        if self.precision == 'fp64' and self.large_k:
            raise ValueError('FP64 supports ordinary retrieval or full_large_k, not the legacy large_k profile')
        if self.precision == 'fp64' and self.distance == 'dtw' and self.dtw_execution != 'optimized':
            raise ValueError('FP64 DTW requires optimized execution; no silent reference-path substitution')
        if self.single_query_engine not in ('buffered', 'specialized'):
            raise ValueError('single_query_engine must be buffered or specialized')
        if self.query_profile not in ('throughput', 'single_query'):
            raise ValueError('query_profile must be throughput or single_query')
        if not isinstance(self.large_k, bool):
            raise ValueError('large_k must be a boolean')
        if not isinstance(self.full_large_k, bool):
            raise ValueError('full_large_k must be a boolean')
        if self.full_large_k and self.large_k:
            raise ValueError('Choose full_large_k global search OR large_k candidate refinement')
        if self.full_large_k and self.distance == 'dtw' and (
                self.dtw_seed != 'signal' or self.dtw_execution != 'optimized'):
            raise ValueError('full_large_k DTW requires signal ranking and optimized execution')
        if self.hausdorff_target not in ('polyline', 'pointset'):
            raise ValueError('hausdorff_target must be polyline or pointset')
        if self.hausdorff_target == 'pointset' and (not (self.large_k or self.full_large_k or self.hausdorff_fields == 'near_far' or self.parameter_mode == 'auto') or self.distance != 'hausdorff'):
            raise ValueError('Point-set Hausdorff requires the explicit large_k Hausdorff profile')
        if self.large_k and (self.distance not in ('hausdorff', 'dfd') or
                            (self.distance == 'hausdorff' and self.hausdorff_target != 'pointset') or
                            (self.distance == 'dfd' and self.dfd_initial != 'light')):
            raise ValueError('large_k supports point-set Hausdorff or light DFD, not DTW/polyline')
        if self.distance not in ('hausdorff', 'dfd', 'dtw'):
            raise ValueError('distance must be hausdorff, dfd or dtw')
        if self.dtw_seed not in ('signal', 'weighted', 'ordered', 'pse'):
            raise ValueError('dtw_seed must be signal, weighted, ordered or pse')
        if self.dfd_initial not in ('light', 'progressive'):
            raise ValueError('dfd_initial must be light or progressive')
        if self.dtw_routing not in ('fixed', 'learned'):
            raise ValueError('dtw_routing must be fixed or learned')
        if self.dtw_execution not in ('optimized', 'reference'):
            raise ValueError('dtw_execution must be optimized or reference')
        if self.dtw_routing == 'learned' and (self.distance != 'dtw' or self.dtw_seed != 'signal'):
            raise ValueError('Learned routing requires DTW signal ranking')
        if isinstance(self.dtw_pool, bool) or not isinstance(self.dtw_pool, int) or self.dtw_pool < 1:
            raise ValueError('dtw_pool must be a positive integer')

    def resolve(self, database_size, top_k=20, *, query_count=None,
                exclude_self=False, initial_k=None, query_batch=None):
        """CPU-only arithmetic. Does not train, benchmark or change the model."""
        from leap.common.runtime.configuration import resolve
        return resolve(self, database_size, top_k, query_count=query_count,
                       exclude_self=exclude_self, initial_k=initial_k,
                       query_batch=query_batch)

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError('Unknown retrieval configuration fields')
        return cls(**value)
