"""Request and response schemas for BERT Fine-tuning algorithm."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class BERTFinetuningParameters(BaseModel):
    """Parameters for BERT Fine-tuning.

    Attributes:
        model_name: BERT model variant to use
        learning_rate: Learning rate for fine-tuning
        epochs: Number of training epochs
        batch_size: Batch size for training
        max_length: Maximum sequence length for tokenization
        use_custom_dataset: Whether to use custom dataset
        custom_texts: Custom text samples for training
        custom_labels: Labels corresponding to custom texts
    """

    model_name: str = Field(
        default="bert-base-uncased",
        description="BERT model variant: bert-base-uncased or distilbert-base-uncased"
    )
    learning_rate: float = Field(
        default=2e-5,
        ge=1e-5,
        le=5e-5,
        description="Learning rate for fine-tuning"
    )
    epochs: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of training epochs"
    )
    batch_size: int = Field(
        default=16,
        ge=8,
        le=32,
        description="Batch size for training"
    )
    max_length: int = Field(
        default=128,
        ge=64,
        le=512,
        description="Maximum sequence length for tokenization"
    )
    use_custom_dataset: bool = Field(
        default=False,
        description="Whether to use custom dataset"
    )
    custom_texts: Optional[List[str]] = Field(
        default=None,
        description="Custom text samples for training"
    )
    custom_labels: Optional[List[int]] = Field(
        default=None,
        description="Labels for custom texts (0, 1, or 2 for 3-class classification)"
    )

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate that model_name is supported."""
        valid_models = ['bert-base-uncased', 'distilbert-base-uncased']
        if v not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}")
        return v

    @field_validator('custom_labels')
    @classmethod
    def validate_custom_labels(cls, v: Optional[List[int]], info) -> Optional[List[int]]:
        """Validate custom labels if provided."""
        if v is not None:
            custom_texts = info.data.get('custom_texts')
            if custom_texts and len(v) != len(custom_texts):
                raise ValueError("custom_labels must have the same length as custom_texts")
            if any(label not in [0, 1, 2] for label in v):
                raise ValueError("All labels must be 0, 1, or 2 for 3-class classification")
        return v


class TrainingHistory(BaseModel):
    """Training history for each epoch.

    Attributes:
        epoch: Epoch number
        train_loss: Training loss
        train_accuracy: Training accuracy
        val_loss: Validation loss (if validation set available)
        val_accuracy: Validation accuracy (if validation set available)
    """
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None


class PredictionResult(BaseModel):
    """Prediction result for a single text.

    Attributes:
        text: Input text
        predicted_label: Predicted class label
        predicted_class: Predicted class name
        confidence: Confidence score for the prediction
        probabilities: Probability distribution across all classes
        true_label: True label (if available)
    """
    text: str
    predicted_label: int
    predicted_class: str
    confidence: float
    probabilities: List[float]
    true_label: Optional[int] = None


class AttentionVisualization(BaseModel):
    """Attention weights visualization data.

    Attributes:
        text: Input text
        tokens: List of tokens
        attention_weights: Attention weights matrix (averaged across heads and layers)
        sample_index: Index of the sample
    """
    text: str
    tokens: List[str]
    attention_weights: List[List[float]]
    sample_index: int


class ConfusionMatrix(BaseModel):
    """Confusion matrix data.

    Attributes:
        matrix: 2D confusion matrix
        labels: Class labels
        accuracy: Overall accuracy
        precision: Precision per class
        recall: Recall per class
        f1_score: F1 score per class
    """
    matrix: List[List[int]]
    labels: List[str]
    accuracy: float
    precision: List[float]
    recall: List[float]
    f1_score: List[float]


class BERTFinetuningResponse(BaseModel):
    """Response schema for BERT Fine-tuning results.

    Attributes:
        success: Whether fine-tuning completed successfully
        training_history: Training history for each epoch
        predictions: Sample predictions on validation/test set
        confusion_matrix: Confusion matrix on validation set
        attention_visualizations: Attention visualizations for sample texts
        metrics: Model metrics and statistics
        visualization_data: Data formatted for frontend visualization
        execution_time_ms: Training execution time in milliseconds
        parameters_used: Actual parameters used for training
        model_info: Information about the model
        error: Error message if training failed
    """

    success: bool
    training_history: List[TrainingHistory] = Field(default_factory=list)
    predictions: List[PredictionResult] = Field(default_factory=list)
    confusion_matrix: Optional[ConfusionMatrix] = None
    attention_visualizations: List[AttentionVisualization] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
