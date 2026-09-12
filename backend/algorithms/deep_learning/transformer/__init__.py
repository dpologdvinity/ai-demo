"""Transformer module for sequence-to-sequence learning.

This module provides a complete implementation of Transformer for seq2seq tasks
using PyTorch, including data generation, model training, and evaluation.
"""

from .model import TransformerModel, TransformerSeq2Seq, PositionalEncoding
from .schema import TransformerRequest, TransformerResponse
from .data import (
    generate_reversal_data,
    generate_addition_data,
    SequenceDataset,
    get_dataset_info
)

__all__ = [
    "TransformerModel",
    "TransformerSeq2Seq",
    "PositionalEncoding",
    "TransformerRequest",
    "TransformerResponse",
    "generate_reversal_data",
    "generate_addition_data",
    "SequenceDataset",
    "get_dataset_info"
]
