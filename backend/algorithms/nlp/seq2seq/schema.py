"""Request and response schemas for Seq2Seq algorithm."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class Seq2SeqParameters(BaseModel):
    """Parameters for Seq2Seq model.

    Attributes:
        task: Task type (translation, reversal, date-conversion)
        hidden_size: Number of hidden units in LSTM layers
        num_layers: Number of LSTM layers
        dropout: Dropout rate for regularization
        attention: Whether to use attention mechanism
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        teacher_forcing_ratio: Ratio of teacher forcing during training
        language_pair: Language pair for translation task (en-fr, en-es)
        use_custom_pairs: Whether to use custom sequence pairs
        custom_input_sequences: Custom input sequences
        custom_target_sequences: Custom target sequences
    """

    task: str = Field(
        default="translation",
        description="Task type: translation, reversal, or date-conversion"
    )
    hidden_size: int = Field(
        default=256,
        ge=64,
        le=512,
        description="Number of hidden units in LSTM layers"
    )
    num_layers: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Number of LSTM layers in encoder/decoder"
    )
    dropout: float = Field(
        default=0.3,
        ge=0.0,
        le=0.5,
        description="Dropout rate for regularization"
    )
    attention: bool = Field(
        default=True,
        description="Whether to use attention mechanism (Bahdanau attention)"
    )
    epochs: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Learning rate for Adam optimizer"
    )
    teacher_forcing_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of using teacher forcing during training"
    )
    language_pair: str = Field(
        default="en-fr",
        description="Language pair for translation: en-fr or en-es"
    )
    use_custom_pairs: bool = Field(
        default=False,
        description="Whether to use custom sequence pairs"
    )
    custom_input_sequences: Optional[List[str]] = Field(
        default=None,
        description="Custom input sequences for training"
    )
    custom_target_sequences: Optional[List[str]] = Field(
        default=None,
        description="Custom target sequences (parallel to input)"
    )

    @field_validator('task')
    @classmethod
    def validate_task(cls, v: str) -> str:
        """Validate that task is supported."""
        valid_tasks = ['translation', 'reversal', 'date-conversion']
        if v not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}")
        return v

    @field_validator('language_pair')
    @classmethod
    def validate_language_pair(cls, v: str) -> str:
        """Validate that language_pair is supported."""
        valid_pairs = ['en-fr', 'en-es']
        if v not in valid_pairs:
            raise ValueError(f"language_pair must be one of {valid_pairs}")
        return v

    @field_validator('custom_target_sequences')
    @classmethod
    def validate_custom_sequences(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        """Validate custom sequences if provided."""
        if v is not None:
            custom_input = info.data.get('custom_input_sequences')
            if custom_input and len(v) != len(custom_input):
                raise ValueError("custom_target_sequences must have the same length as custom_input_sequences")
        return v


class TranslationPair(BaseModel):
    """A single input-output translation pair.

    Attributes:
        input: Input sequence
        target: Target sequence (ground truth)
        prediction: Model prediction
        bleu_score: BLEU score for this prediction
        attention_weights: Attention weights (if attention enabled)
    """
    input: str
    target: str
    prediction: str
    bleu_score: float
    attention_weights: Optional[List[List[float]]] = None


class TrainingHistory(BaseModel):
    """Training history for each epoch.

    Attributes:
        epoch: Epoch number
        loss: Training loss for this epoch
        bleu: Average BLEU score on validation samples
    """
    epoch: int
    loss: float
    bleu: float


class Seq2SeqResponse(BaseModel):
    """Response schema for Seq2Seq model results.

    Attributes:
        success: Whether training completed successfully
        task: Task type used
        sample_predictions: Sample predictions with BLEU scores
        training_history: Loss and BLEU scores per epoch
        attention_heatmap: Attention weights for visualization
        metrics: Overall metrics (avg BLEU, final loss, etc.)
        architecture_info: Model architecture details
        vocabulary_info: Information about source/target vocabularies
        visualization_data: Data formatted for frontend visualization
        execution_time_ms: Training execution time in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed
    """

    success: bool
    task: str
    sample_predictions: List[TranslationPair] = Field(default_factory=list)
    training_history: List[TrainingHistory] = Field(default_factory=list)
    attention_heatmap: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    architecture_info: Dict[str, Any] = Field(default_factory=dict)
    vocabulary_info: Dict[str, Any] = Field(default_factory=dict)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
