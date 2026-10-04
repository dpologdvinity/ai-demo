"""Smoke tests for Named Entity Recognition algorithm."""

from algorithms.nlp.ner.model import NERModel
from algorithms.nlp.ner.schema import NERParameters


def test_ner_model_initialization():
    """Test NER model can be initialized."""
    model = NERModel()
    assert model is not None


def test_ner_extract_with_defaults():
    """Test NER extract method with default parameters."""
    model = NERModel()
    params = NERParameters(
        model_name="en_core_web_sm",
        entity_types=["PERSON", "ORG", "GPE"],
        confidence_threshold=0.0,
        text_index=0,
        merge_entities=True
    )

    response = model.process(params)

    assert response is not None
    assert hasattr(response, 'entities')
    assert isinstance(response.entities, list)
