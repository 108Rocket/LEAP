"""LEAP public API with lazy loading of CUDA implementation modules."""
from importlib import import_module

__version__ = '1.0.0'
__all__ = ['LEAP', 'DualFieldRetriever', 'TrajectoryIndex',
           'RetrievalConfig', 'TopKProfile', 'PackedTrajectories', 'load_checkpoint', '__version__']

_EXPORTS = {
    'LEAP': ('leap.common.dual_field', 'DualFieldRetriever'),
    'DualFieldRetriever': ('leap.common.dual_field', 'DualFieldRetriever'),
    'TrajectoryIndex': ('leap.common.model', 'TrajectoryIndex'),
    'RetrievalConfig': ('leap.common.config', 'RetrievalConfig'),
    'TopKProfile': ('leap.common.config', 'TopKProfile'),
    'PackedTrajectories': ('leap.common.data', 'PackedTrajectories'),
    'load_checkpoint': ('leap.checkpoints', 'load_checkpoint'),
}


def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module), attribute)
        globals()[name] = value
        return value
    raise AttributeError(name)
