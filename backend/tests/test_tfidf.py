"""Smoke tests for TF-IDF algorithm."""

from algorithms.nlp.tfidf.model import TFIDFModel, train_tfidf
from algorithms.nlp.tfidf.schema import TFIDFRequest


def test_tfidf_model_initialization():
    """Test TF-IDF model can be initialized."""
    model = TFIDFModel()
    assert model is not None


def test_tfidf_train_with_defaults():
    """Test TF-IDF train method with default parameters."""
    request = TFIDFRequest(
        custom_documents=["hello world test example document", "foo bar world example text"],
        max_features=100,
        min_df=1,
        max_df=1.0
    )

    response = train_tfidf(request)

    assert response is not None
    assert hasattr(response, 'success')
    assert response.success is True
    assert hasattr(response, 'tfidf_matrix')
