"""Seq2Seq (Sequence-to-Sequence) algorithm module.

This module provides encoder-decoder architecture for sequence transformation tasks
including machine translation, sequence reversal, and date format conversion.
"""

from .model import Seq2SeqModel
from .schema import (
    Seq2SeqParameters,
    Seq2SeqResponse,
    TranslationPair,
    TrainingHistory
)
from .data import (
    get_translation_pairs,
    get_reversal_pairs,
    get_date_conversion_pairs,
    get_dataset_by_task,
    get_dataset_info
)

__all__ = [
    'Seq2SeqModel',
    'Seq2SeqParameters',
    'Seq2SeqResponse',
    'TranslationPair',
    'TrainingHistory',
    'get_translation_pairs',
    'get_reversal_pairs',
    'get_date_conversion_pairs',
    'get_dataset_by_task',
    'get_dataset_info',
]
