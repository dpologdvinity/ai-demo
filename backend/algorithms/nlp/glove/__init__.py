"""GloVe (Global Vectors) word embeddings module.

This module provides functionality for working with GloVe word embeddings,
including loading pre-trained vectors, finding similar words, and solving
word analogies.
"""

from .model import GloVeModel
from .schema import (
    GloVeParameters,
    GloVeResponse,
    SimilarWord,
    AnalogyResult
)
from .data import (
    get_demo_glove_embeddings,
    get_vocabulary_stats,
    get_sample_queries,
    get_sample_analogies,
    get_embeddings_info
)

__all__ = [
    "GloVeModel",
    "GloVeParameters",
    "GloVeResponse",
    "SimilarWord",
    "AnalogyResult",
    "get_demo_glove_embeddings",
    "get_vocabulary_stats",
    "get_sample_queries",
    "get_sample_analogies",
    "get_embeddings_info",
]
