"""Smoke tests for Topic Modeling algorithm."""

from algorithms.nlp.topic_modeling.model import TopicModelingModel
from algorithms.nlp.topic_modeling.schema import TopicModelingParameters


def test_topic_modeling_model_initialization():
    """Test Topic Modeling model can be initialized."""
    model = TopicModelingModel()
    assert model is not None


def test_topic_modeling_train_with_defaults():
    """Test Topic Modeling train method with default parameters."""
    model = TopicModelingModel()
    params = TopicModelingParameters(
        n_topics=5,
        max_iterations=100,
        alpha="auto",
        beta="auto",
        min_df=2,
        max_df=0.95
    )

    response = model.train(params)

    assert response is not None
    assert hasattr(response, 'success')
    assert response.success is True
    assert hasattr(response, 'topics')
