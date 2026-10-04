"""Smoke tests for Tokenization algorithm."""

from algorithms.nlp.tokenization.model import TokenizationModel
from algorithms.nlp.tokenization.schema import TokenizationRequest


def test_tokenization_model_initialization():
    """Test Tokenization model can be initialized."""
    model = TokenizationModel()
    assert model is not None


def test_tokenization_tokenize_with_defaults():
    """Test Tokenization tokenize method with default parameters."""
    model = TokenizationModel()
    request = TokenizationRequest(
        custom_text="Hello world this is a test",
        tokenizer_type="word",
        lowercase=True,
        remove_punctuation=False,
        remove_stopwords=False
    )

    response = model.tokenize(request)

    assert response is not None
    assert hasattr(response, 'main_result')
    assert response.main_result is not None
    assert response.main_result.tokens is not None
    assert isinstance(response.main_result.tokens, list)
    assert len(response.main_result.tokens) > 0
