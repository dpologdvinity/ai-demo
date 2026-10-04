"""Smoke tests for Sentiment Analysis algorithm."""

import pytest
from algorithms.nlp.sentiment_analysis.model import SentimentAnalysisModel
from algorithms.nlp.sentiment_analysis.schema import SentimentAnalysisParameters


class TestSentimentAnalysis:
    """Basic smoke tests for SentimentAnalysisModel."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = SentimentAnalysisModel()
        assert model is not None

    def test_analyze_with_defaults(self):
        """Test analyze method with default parameters."""
        model = SentimentAnalysisModel()
        params = SentimentAnalysisParameters(
            model_type="vader",
            confidence_threshold=0.5,
            neutral_threshold=0.05
        )

        response = model.analyze(params)

        assert response is not None
        assert hasattr(response, 'predictions')
        assert hasattr(response, 'metrics')
        assert len(response.predictions) > 0
        assert 'avg_compound_score' in response.metrics
