"""Smoke tests for POS Tagging algorithm."""

from algorithms.nlp.pos_tagging.model import POSTaggingModel
from algorithms.nlp.pos_tagging.schema import POSTaggingParameters


def test_pos_tagging_model_initialization():
    """Test POS Tagging model can be initialized."""
    model = POSTaggingModel()
    assert model is not None


def test_pos_tagging_tag_with_defaults():
    """Test POS Tagging tag method with default parameters."""
    model = POSTaggingModel()
    params = POSTaggingParameters(
        tagger="spacy",
        text_index=0,
        show_fine_grained=True,
        show_dependencies=True,
        tag_scheme="penn"
    )

    response = model.process(params)

    assert response is not None
    assert hasattr(response, 'tagged_words')
    assert isinstance(response.tagged_words, list)
