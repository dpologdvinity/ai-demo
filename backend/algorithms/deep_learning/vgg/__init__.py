"""VGG Network implementation."""

from .schema import VGGRequest, VGGResponse

# Import model only when torch is available
try:
    from .model import VGGModel
    __all__ = ['VGGModel', 'VGGRequest', 'VGGResponse']
except ImportError:
    # Torch not available, only export schemas
    __all__ = ['VGGRequest', 'VGGResponse']
