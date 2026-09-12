#!/usr/bin/env python3
"""
Test script for Sentiment Analysis implementation.
Run this after all dependencies are installed to verify the implementation works.
"""

import sys
import time

def test_sentiment_analysis():
    """Test the sentiment analysis implementation."""
    print("=" * 60)
    print("Testing Sentiment Analysis Implementation")
    print("=" * 60)

    # Test 1: Import modules
    print("\n[1/4] Testing imports...")
    try:
        from algorithms.nlp.sentiment_analysis import (
            SentimentAnalysisModel,
            SentimentAnalysisParameters,
            SentimentAnalysisResponse
        )
        print("✓ Successfully imported sentiment analysis modules")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

    # Test 2: Create parameters
    print("\n[2/4] Testing parameter creation...")
    try:
        params = SentimentAnalysisParameters(
            model_type='vader',
            confidence_threshold=0.5,
            neutral_threshold=0.05
        )
        print(f"✓ Created parameters: model={params.model_type}, "
              f"confidence={params.confidence_threshold}")
    except Exception as e:
        print(f"✗ Parameter creation failed: {e}")
        return False

    # Test 3: Initialize model
    print("\n[3/4] Testing model initialization...")
    try:
        model = SentimentAnalysisModel()
        print("✓ Successfully initialized sentiment analysis model")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False

    # Test 4: Run analysis
    print("\n[4/4] Testing sentiment analysis...")
    try:
        start_time = time.time()
        result = model.analyze(params)
        elapsed = (time.time() - start_time) * 1000

        if not result.success:
            print(f"✗ Analysis failed: {result.error}")
            return False

        print(f"✓ Analysis completed in {elapsed:.2f}ms")
        print(f"  - Analyzed {len(result.predictions)} texts")
        print(f"  - Distribution: {result.distribution.positive} positive, "
              f"{result.distribution.negative} negative, "
              f"{result.distribution.neutral} neutral")
        print(f"  - Average confidence: {result.metrics['avg_confidence']:.4f}")
        print(f"  - Average compound score: {result.metrics['avg_compound_score']:.4f}")

        # Show a sample prediction
        if result.predictions:
            sample = result.predictions[0]
            print(f"\n  Sample prediction:")
            print(f"    Text: \"{sample.text[:60]}...\"")
            print(f"    Sentiment: {sample.sentiment}")
            print(f"    Confidence: {sample.confidence:.4f}")
            print(f"    Compound: {sample.compound:.4f}")

    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_sentiment_analysis()
    sys.exit(0 if success else 1)
