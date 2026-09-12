"""Sentiment Analysis algorithm implementation using VADER."""

import time
from typing import Dict, Any, List, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .schema import (
    SentimentAnalysisParameters,
    SentimentAnalysisResponse,
    SentimentPrediction,
    SentimentDistribution
)
from .data import (
    get_default_texts,
    prepare_custom_texts,
    get_texts_info
)


class SentimentAnalysisModel:
    """Sentiment Analysis algorithm implementation.

    This class implements sentiment analysis using the VADER (Valence Aware
    Dictionary and sEntiment Reasoner) lexicon and rule-based sentiment
    analysis tool.

    VADER is specifically attuned to sentiments expressed in social media
    and works well on texts from other domains as well.

    Attributes:
        analyzer: VADER sentiment intensity analyzer
        parameters: Analysis parameters used
        texts: Input texts for analysis

    Example:
        >>> params = SentimentAnalysisParameters(model_type='vader')
        >>> model = SentimentAnalysisModel()
        >>> response = model.analyze(params)
        >>> print(f"Analyzed {len(response.predictions)} texts")
    """

    def __init__(self):
        """Initialize the Sentiment Analysis model."""
        self.analyzer: Optional[SentimentIntensityAnalyzer] = None
        self.parameters: Optional[SentimentAnalysisParameters] = None
        self.texts: Optional[List[str]] = None

    def analyze(self, parameters: SentimentAnalysisParameters) -> SentimentAnalysisResponse:
        """Analyze sentiment of text samples.

        Args:
            parameters: Analysis parameters including model_type and thresholds

        Returns:
            SentimentAnalysisResponse containing predictions and statistics

        Example:
            >>> params = SentimentAnalysisParameters(confidence_threshold=0.6)
            >>> model = SentimentAnalysisModel()
            >>> result = model.analyze(params)
            >>> if result.success:
            ...     print(f"Analysis completed in {result.execution_time_ms:.2f}ms")
        """
        try:
            start_time = time.time()
            self.parameters = parameters

            # Initialize VADER analyzer
            self.analyzer = SentimentIntensityAnalyzer()

            # Load or prepare texts
            if parameters.use_custom_texts and parameters.custom_texts:
                self.texts = prepare_custom_texts(parameters.custom_texts)
            else:
                self.texts = get_default_texts()

            # Get texts info
            texts_info = get_texts_info(self.texts)

            # Analyze all texts
            predictions = self._analyze_texts()

            # Calculate distribution
            distribution = self._calculate_distribution(predictions)

            # Get top positive and negative texts
            top_positive = self._get_top_sentiments(predictions, 'positive', n=5)
            top_negative = self._get_top_sentiments(predictions, 'negative', n=5)

            # Calculate metrics
            metrics = self._calculate_metrics(predictions, texts_info)

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data(
                predictions, distribution
            )

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            return SentimentAnalysisResponse(
                success=True,
                predictions=predictions,
                distribution=distribution,
                top_positive=top_positive,
                top_negative=top_negative,
                metrics=metrics,
                visualization_data=visualization_data,
                execution_time_ms=execution_time_ms,
                parameters_used=parameters.model_dump(),
                model_info={
                    "model_type": parameters.model_type,
                    "analyzer": "VADER (Valence Aware Dictionary and sEntiment Reasoner)",
                    "version": "3.3.2",
                    "description": "Rule-based sentiment analysis tool specifically attuned to social media text"
                }
            )

        except Exception as e:
            return SentimentAnalysisResponse(
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def _analyze_texts(self) -> List[SentimentPrediction]:
        """Analyze sentiment for all texts.

        Returns:
            List of sentiment predictions
        """
        predictions = []

        for text in self.texts:
            # Get VADER scores
            scores = self.analyzer.polarity_scores(text)

            # Extract compound score (overall sentiment)
            compound = scores['compound']

            # Determine sentiment label based on compound score and threshold
            neutral_threshold = self.parameters.neutral_threshold

            if compound >= neutral_threshold:
                sentiment = 'positive'
                confidence = scores['pos']
            elif compound <= -neutral_threshold:
                sentiment = 'negative'
                confidence = scores['neg']
            else:
                sentiment = 'neutral'
                confidence = scores['neu']

            # Create prediction object
            prediction = SentimentPrediction(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                scores={
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu']
                },
                compound=compound
            )

            predictions.append(prediction)

        return predictions

    def _calculate_distribution(
        self, predictions: List[SentimentPrediction]
    ) -> SentimentDistribution:
        """Calculate sentiment distribution statistics.

        Args:
            predictions: List of sentiment predictions

        Returns:
            SentimentDistribution object with counts and percentages
        """
        total = len(predictions)
        positive_count = sum(1 for p in predictions if p.sentiment == 'positive')
        negative_count = sum(1 for p in predictions if p.sentiment == 'negative')
        neutral_count = sum(1 for p in predictions if p.sentiment == 'neutral')

        return SentimentDistribution(
            positive=positive_count,
            negative=negative_count,
            neutral=neutral_count,
            positive_pct=round((positive_count / total * 100) if total > 0 else 0, 2),
            negative_pct=round((negative_count / total * 100) if total > 0 else 0, 2),
            neutral_pct=round((neutral_count / total * 100) if total > 0 else 0, 2)
        )

    def _get_top_sentiments(
        self,
        predictions: List[SentimentPrediction],
        sentiment_type: str,
        n: int = 5
    ) -> List[SentimentPrediction]:
        """Get top N texts with specified sentiment.

        Args:
            predictions: List of sentiment predictions
            sentiment_type: Type of sentiment ('positive' or 'negative')
            n: Number of top results to return

        Returns:
            List of top N sentiment predictions
        """
        # Filter by sentiment type
        filtered = [p for p in predictions if p.sentiment == sentiment_type]

        # Sort by compound score (absolute value for negative)
        if sentiment_type == 'positive':
            sorted_predictions = sorted(
                filtered,
                key=lambda x: x.compound if x.compound is not None else 0,
                reverse=True
            )
        elif sentiment_type == 'negative':
            sorted_predictions = sorted(
                filtered,
                key=lambda x: abs(x.compound) if x.compound is not None else 0,
                reverse=True
            )
        else:
            sorted_predictions = filtered

        return sorted_predictions[:n]

    def _calculate_metrics(
        self,
        predictions: List[SentimentPrediction],
        texts_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate analysis metrics.

        Args:
            predictions: List of sentiment predictions
            texts_info: Information about the text corpus

        Returns:
            Dictionary containing analysis metrics
        """
        if not predictions:
            return {
                "total_texts": 0,
                "avg_confidence": 0.0,
                "avg_compound_score": 0.0,
                "texts_info": texts_info
            }

        # Calculate average confidence
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)

        # Calculate average compound score
        compound_scores = [p.compound for p in predictions if p.compound is not None]
        avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0.0

        # Calculate confidence by sentiment
        pos_predictions = [p for p in predictions if p.sentiment == 'positive']
        neg_predictions = [p for p in predictions if p.sentiment == 'negative']
        neu_predictions = [p for p in predictions if p.sentiment == 'neutral']

        return {
            "total_texts": len(predictions),
            "avg_confidence": round(avg_confidence, 4),
            "avg_compound_score": round(avg_compound, 4),
            "avg_positive_confidence": round(
                sum(p.confidence for p in pos_predictions) / len(pos_predictions)
                if pos_predictions else 0.0,
                4
            ),
            "avg_negative_confidence": round(
                sum(p.confidence for p in neg_predictions) / len(neg_predictions)
                if neg_predictions else 0.0,
                4
            ),
            "avg_neutral_confidence": round(
                sum(p.confidence for p in neu_predictions) / len(neu_predictions)
                if neu_predictions else 0.0,
                4
            ),
            "texts_info": texts_info
        }

    def _prepare_visualization_data(
        self,
        predictions: List[SentimentPrediction],
        distribution: SentimentDistribution
    ) -> Dict[str, Any]:
        """Prepare data for visualization.

        Args:
            predictions: List of sentiment predictions
            distribution: Sentiment distribution statistics

        Returns:
            Dictionary containing formatted visualization data
        """
        # Pie chart data for sentiment distribution
        pie_chart_data = {
            "labels": ["Positive", "Negative", "Neutral"],
            "values": [
                distribution.positive,
                distribution.negative,
                distribution.neutral
            ],
            "percentages": [
                distribution.positive_pct,
                distribution.negative_pct,
                distribution.neutral_pct
            ],
            "colors": ["#22c55e", "#ef4444", "#6b7280"]  # green, red, gray
        }

        # Confidence score distribution (bar chart)
        confidence_data = {
            "labels": [
                f"Text {i+1}"[:20] for i in range(min(20, len(predictions)))
            ],
            "positive_scores": [
                p.scores.get('positive', 0) for p in predictions[:20]
            ],
            "negative_scores": [
                p.scores.get('negative', 0) for p in predictions[:20]
            ],
            "neutral_scores": [
                p.scores.get('neutral', 0) for p in predictions[:20]
            ],
            "compound_scores": [
                p.compound for p in predictions[:20]
            ]
        }

        # Compound score histogram
        compound_scores = [p.compound for p in predictions if p.compound is not None]
        histogram_data = {
            "scores": compound_scores,
            "bins": 20,
            "range": [-1, 1]
        }

        return {
            "pie_chart": pie_chart_data,
            "confidence_bars": confidence_data,
            "compound_histogram": histogram_data,
            "sample_predictions": [
                {
                    "text": p.text[:100] + "..." if len(p.text) > 100 else p.text,
                    "sentiment": p.sentiment,
                    "confidence": round(p.confidence, 4),
                    "compound": round(p.compound, 4) if p.compound is not None else None
                }
                for p in predictions[:10]
            ]
        }
