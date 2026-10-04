"""Smoke tests for Text Classification algorithm."""

from algorithms.nlp.text_classification.model import TextClassificationModel
from algorithms.nlp.text_classification.schema import TextClassificationParameters


def test_text_classification_model_initialization():
    """Test Text Classification model can be initialized."""
    model = TextClassificationModel()
    assert model is not None


def test_text_classification_train_with_defaults():
    """Test Text Classification train method with default parameters."""
    model = TextClassificationModel()
    params = TextClassificationParameters(
        classifier_type="naive_bayes",
        max_features=1000,
        test_size=0.2,
        ngram_range=(1, 2)
    )

    response = model.train(params)

    assert response is not None
    assert response.success is True
    assert hasattr(response, 'overall_metrics')
    assert len(response.confusion_matrix) > 0
    assert len(response.class_names) > 0
    assert len(response.predictions) > 0
