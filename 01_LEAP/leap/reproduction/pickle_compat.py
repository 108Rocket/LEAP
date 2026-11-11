"""Compatibility for explicitly trusted historical NumPy pickle inputs."""
import pickle


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core','numpy.core',1)
        return super().find_class(module,name)
