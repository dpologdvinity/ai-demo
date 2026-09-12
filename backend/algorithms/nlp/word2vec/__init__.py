"""Word2Vec algorithm implementation."""

from .model import Word2VecModel
from .schema import Word2VecParameters, Word2VecResponse, SimilarWord, AnalogyResult
from .data import get_default_corpus, prepare_custom_corpus, get_corpus_info

__all__ = [
    'Word2VecModel',
    'Word2VecParameters',
    'Word2VecResponse',
    'SimilarWord',
    'AnalogyResult',
    'get_default_corpus',
    'prepare_custom_corpus',
    'get_corpus_info'
]
