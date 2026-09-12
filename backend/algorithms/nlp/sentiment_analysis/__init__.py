"""Sentiment Analysis algorithm module.

This module provides sentiment analysis capabilities using VADER
(Valence Aware Dictionary and sEntiment Reasoner).
"""

from .model import SentimentAnalysisModel
from .schema import (
    SentimentAnalysisParameters,
    SentimentAnalysisResponse,
    SentimentPrediction,
    SentimentDistribution
)
from .data import get_default_texts, get_sample_reviews

__all__ = [
    'SentimentAnalysisModel',
    'SentimentAnalysisParameters',
    'SentimentAnalysisResponse',
    'SentimentPrediction',
    'SentimentDistribution',
    'get_default_texts',
    'get_sample_reviews',
]
