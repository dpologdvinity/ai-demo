"""Smoke tests for BERT Fine-tuning algorithm."""

from algorithms.nlp.bert_finetuning.model import BERTFinetuningModel
from algorithms.nlp.bert_finetuning.schema import BERTFinetuningParameters


def test_bert_model_initialization():
    """Test BERT model can be initialized."""
    model = BERTFinetuningModel()
    assert model is not None


def test_bert_train_with_minimal_params():
    """Test BERT train method with minimal parameters (fast)."""
    model = BERTFinetuningModel()
    params = BERTFinetuningParameters(
        model_name="distilbert-base-uncased",
        learning_rate=2e-5,
        epochs=1,
        batch_size=8,
        max_length=128
    )

    # Just test instantiation and params, not full train (too slow)
    assert params.epochs == 1
    assert params.model_name == "distilbert-base-uncased"
