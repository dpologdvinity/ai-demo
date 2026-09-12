"""Part-of-Speech (POS) Tagging algorithm implementation."""

from .model import POSTaggingModel
from .schema import (
    POSTaggingParameters,
    POSTaggingResponse,
    TaggedWord,
    DependencyEdge
)
from .data import (
    get_sample_sentences,
    get_pos_tag_descriptions,
    get_pos_tag_colors,
    get_dataset_info
)

__all__ = [
    'POSTaggingModel',
    'POSTaggingParameters',
    'POSTaggingResponse',
    'TaggedWord',
    'DependencyEdge',
    'get_sample_sentences',
    'get_pos_tag_descriptions',
    'get_pos_tag_colors',
    'get_dataset_info'
]
