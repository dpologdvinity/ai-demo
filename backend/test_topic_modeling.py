#!/usr/bin/env python3
"""Test script for Topic Modeling (LDA) implementation."""

import sys
import json
from algorithms.nlp.topic_modeling import (
    TopicModelingModel,
    TopicModelingParameters,
    get_dataset_info,
    get_default_documents
)


def test_basic_training():
    """Test basic training with default parameters."""
    print("=" * 60)
    print("Test 1: Basic Training with Default Parameters")
    print("=" * 60)

    params = TopicModelingParameters(n_topics=5, max_iterations=100)
    model = TopicModelingModel()
    result = model.train(params)

    assert result.success, f"Training failed: {result.error}"
    assert len(result.topics) == 5, f"Expected 5 topics, got {len(result.topics)}"
    assert result.coherence_score is not None, "Coherence score is None"
    assert result.perplexity is not None, "Perplexity is None"

    print(f"✓ Success: {result.success}")
    print(f"✓ Topics discovered: {len(result.topics)}")
    print(f"✓ Documents processed: {len(result.document_topics)}")
    print(f"✓ Coherence score: {result.coherence_score:.4f}")
    print(f"✓ Perplexity: {result.perplexity:.2f}")
    print(f"✓ Execution time: {result.execution_time_ms:.2f}ms")
    print(f"✓ Vocabulary size: {result.metrics['vocabulary_size']}")

    # Print topics
    print("\nDiscovered Topics:")
    for topic in result.topics:
        print(f"  Topic {topic.topic_id}: {topic.keywords}")

    print("\n✓ Test 1 PASSED\n")
    return result


def test_parameter_variations():
    """Test with different parameter combinations."""
    print("=" * 60)
    print("Test 2: Parameter Variations")
    print("=" * 60)

    test_cases = [
        {"n_topics": 3, "max_iterations": 50, "alpha": 0.1, "beta": 0.01},
        {"n_topics": 8, "max_iterations": 100, "alpha": "auto", "beta": "auto"},
        {"n_topics": 5, "max_iterations": 200, "alpha": 0.5, "beta": 0.1},
    ]

    for i, params_dict in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {params_dict}")
        params = TopicModelingParameters(**params_dict)
        model = TopicModelingModel()
        result = model.train(params)

        assert result.success, f"Training failed: {result.error}"
        assert len(result.topics) == params_dict['n_topics']

        print(f"  ✓ Topics: {len(result.topics)}")
        print(f"  ✓ Coherence: {result.coherence_score:.4f}")
        print(f"  ✓ Time: {result.execution_time_ms:.0f}ms")

    print("\n✓ Test 2 PASSED\n")


def test_document_topic_distribution():
    """Test document-topic distributions."""
    print("=" * 60)
    print("Test 3: Document-Topic Distribution")
    print("=" * 60)

    params = TopicModelingParameters(n_topics=5, max_iterations=100)
    model = TopicModelingModel()
    result = model.train(params)

    assert result.success, "Training failed"
    assert len(result.document_topics) > 0, "No document topics returned"

    # Check that distributions sum to ~1.0
    for doc_topic in result.document_topics[:5]:
        dist_sum = sum(doc_topic.topic_distribution)
        assert 0.99 <= dist_sum <= 1.01, f"Distribution sum {dist_sum} not close to 1.0"

        print(f"Document {doc_topic.document_id}:")
        print(f"  Preview: {doc_topic.document_preview[:80]}...")
        print(f"  Dominant topic: {doc_topic.dominant_topic}")
        print(f"  Distribution: {[f'{p:.3f}' for p in doc_topic.topic_distribution]}")

    print("\n✓ Test 3 PASSED\n")


def test_visualization_data():
    """Test visualization data structure."""
    print("=" * 60)
    print("Test 4: Visualization Data")
    print("=" * 60)

    params = TopicModelingParameters(n_topics=4, max_iterations=80)
    model = TopicModelingModel()
    result = model.train(params)

    assert result.success, "Training failed"
    assert "word_clouds" in result.visualization_data
    assert "topic_keywords" in result.visualization_data
    assert "heatmap" in result.visualization_data
    assert "documents_by_topic" in result.visualization_data

    # Check word clouds
    word_clouds = result.visualization_data["word_clouds"]
    assert len(word_clouds) == 4, f"Expected 4 word clouds, got {len(word_clouds)}"

    print(f"✓ Word clouds: {len(word_clouds)}")
    print(f"✓ Topic keywords: {len(result.visualization_data['topic_keywords'])}")
    print(f"✓ Heatmap matrix: {len(result.visualization_data['heatmap']['matrix'])} x {len(result.visualization_data['heatmap']['matrix'][0])}")
    print(f"✓ Documents by topic: {len(result.visualization_data['documents_by_topic'])} topics")

    # Show sample word cloud data
    if word_clouds:
        wc = word_clouds[0]
        print(f"\nSample word cloud for Topic 0:")
        for item in wc['data'][:5]:
            print(f"  - {item['text']}: {item['value']:.4f}")

    print("\n✓ Test 4 PASSED\n")


def test_custom_documents():
    """Test with custom documents."""
    print("=" * 60)
    print("Test 5: Custom Documents")
    print("=" * 60)

    custom_docs = [
        "Machine learning is a subset of artificial intelligence focused on data-driven algorithms.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and analyze visual information from images.",
        "Reinforcement learning trains agents to make decisions through trial and error.",
        "Supervised learning uses labeled data to train predictive models.",
        "Unsupervised learning discovers patterns in unlabeled data.",
        "Transfer learning applies knowledge from one task to improve performance on another.",
    ]

    params = TopicModelingParameters(
        n_topics=3,
        max_iterations=100,
        use_custom_documents=True,
        custom_documents=custom_docs
    )
    model = TopicModelingModel()
    result = model.train(params)

    assert result.success, f"Training failed: {result.error}"
    assert len(result.document_topics) == len(custom_docs)

    print(f"✓ Custom documents: {len(custom_docs)}")
    print(f"✓ Topics: {len(result.topics)}")
    print(f"✓ Coherence: {result.coherence_score:.4f}")

    print("\nDiscovered Topics from Custom Documents:")
    for topic in result.topics:
        print(f"  Topic {topic.topic_id}: {topic.keywords}")

    print("\n✓ Test 5 PASSED\n")


def test_dataset_info():
    """Test dataset info function."""
    print("=" * 60)
    print("Test 6: Dataset Info")
    print("=" * 60)

    info = get_dataset_info()

    assert "num_documents" in info
    assert "total_words" in info
    assert "topics_covered" in info

    print(f"✓ Dataset name: {info['name']}")
    print(f"✓ Number of documents: {info['num_documents']}")
    print(f"✓ Total words: {info['total_words']}")
    print(f"✓ Average document length: {info['avg_document_length']}")
    print(f"✓ Topics covered: {len(info['topics_covered'])}")

    for topic in info['topics_covered']:
        print(f"  - {topic}")

    print("\n✓ Test 6 PASSED\n")


def test_metrics():
    """Test metrics calculation."""
    print("=" * 60)
    print("Test 7: Metrics")
    print("=" * 60)

    params = TopicModelingParameters(
        n_topics=6,
        max_iterations=150,
        alpha=0.5,
        beta=0.1,
        min_df=2,
        max_df=0.9
    )
    model = TopicModelingModel()
    result = model.train(params)

    assert result.success, "Training failed"

    metrics = result.metrics
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    assert metrics["n_topics"] == 6
    assert metrics["n_documents"] > 0
    assert metrics["vocabulary_size"] > 0

    print("\n✓ Test 7 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TOPIC MODELING (LDA) - COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")

    try:
        # Run all tests
        test_basic_training()
        test_parameter_variations()
        test_document_topic_distribution()
        test_visualization_data()
        test_custom_documents()
        test_dataset_info()
        test_metrics()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
