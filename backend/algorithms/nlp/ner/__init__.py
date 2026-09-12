"""Named Entity Recognition (NER) algorithm implementation."""

from .model import NERModel
from .schema import NERParameters, NERResponse, Entity, AnnotatedSpan
from .data import (
    get_default_texts,
    get_entity_type_info,
    get_all_entity_types,
    get_sample_text_by_id
)

__all__ = [
    'NERModel',
    'NERParameters',
    'NERResponse',
    'Entity',
    'AnnotatedSpan',
    'get_default_texts',
    'get_entity_type_info',
    'get_all_entity_types',
    'get_sample_text_by_id'
]
