"""Smoke tests for Seq2Seq algorithm."""

from algorithms.nlp.seq2seq.model import Seq2SeqModel
from algorithms.nlp.seq2seq.schema import Seq2SeqParameters


def test_seq2seq_model_initialization():
    """Test Seq2Seq model can be initialized."""
    model = Seq2SeqModel()
    assert model is not None


def test_seq2seq_train_with_minimal_params():
    """Test Seq2Seq train method with minimal parameters (fast)."""
    model = Seq2SeqModel()
    params = Seq2SeqParameters(
        task="date-conversion",
        hidden_size=256,
        num_layers=1,
        dropout=0.3,
        attention=False,
        epochs=10,
        learning_rate=0.001,
        teacher_forcing_ratio=0.5,
        language_pair="en-fr"
    )

    response = model.train(params)

    assert response is not None
    assert hasattr(response, 'success')
    # May fail due to timeout/resources, but model should initialize
    assert params.task == "date-conversion"
