"""Request and response schemas for Text Classification algorithm."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class TextClassificationParameters(BaseModel):
    """Parameters for Text Classification.

    Attributes:
        classifier_type: Model type (naive_bayes, logistic_regression, svm)
        max_features: TF-IDF max features
        test_size: Train/test split ratio
        ngram_range: N-gram range for TF-IDF
        use_custom_dataset: Whether to use custom dataset
        custom_texts: Custom texts for classification
        custom_labels: Custom labels corresponding to texts
    """

    classifier_type: str = Field(
        default="naive_bayes",
        description="Classifier type: naive_bayes, logistic_regression, or svm"
    )
    max_features: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum number of TF-IDF features"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.4,
        description="Train/test split ratio"
    )
    ngram_range: tuple = Field(
        default=(1, 2),
        description="N-gram range for TF-IDF vectorization"
    )
    use_custom_dataset: bool = Field(
        default=False,
        description="Whether to use custom dataset"
    )
    custom_texts: Optional[List[str]] = Field(
        default=None,
        description="Custom texts for classification"
    )
    custom_labels: Optional[List[str]] = Field(
        default=None,
        description="Custom labels corresponding to texts"
    )

    @field_validator('classifier_type')
    @classmethod
    def validate_classifier_type(cls, v: str) -> str:
        """Validate that classifier_type is supported."""
        valid_classifiers = ['naive_bayes', 'logistic_regression', 'svm']
        if v not in valid_classifiers:
            raise ValueError(f"classifier_type must be one of {valid_classifiers}")
        return v

    @field_validator('test_size')
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        """Validate that test_size is within valid range."""
        if not 0.1 <= v <= 0.4:
            raise ValueError("test_size must be between 0.1 and 0.4")
        return v

    @field_validator('ngram_range')
    @classmethod
    def validate_ngram_range(cls, v: tuple) -> tuple:
        """Validate that ngram_range is valid."""
        valid_ranges = [(1, 1), (1, 2), (1, 3)]
        if v not in valid_ranges:
            raise ValueError(f"ngram_range must be one of {valid_ranges}")
        return v


class TextPrediction(BaseModel):
    """Prediction for a single text.

    Attributes:
        text: The input text
        true_label: True label (if available)
        predicted_label: Predicted label
        confidence: Confidence score for the prediction
        probabilities: Probability distribution across all classes
    """
    text: str
    true_label: Optional[str] = None
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float] = Field(default_factory=dict)


class ClassMetrics(BaseModel):
    """Metrics for a single class.

    Attributes:
        class_name: Name of the class
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        support: Number of samples in this class
    """
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int


class TopFeature(BaseModel):
    """Top discriminative feature for a class.

    Attributes:
        feature: The feature (word or n-gram)
        weight: Feature weight/importance
    """
    feature: str
    weight: float


class TextClassificationResponse(BaseModel):
    """Response schema for Text Classification results.

    Attributes:
        success: Whether classification completed successfully
        predictions: List of predictions for test set
        confusion_matrix: Confusion matrix as 2D array
        class_names: List of class names
        class_metrics: Metrics for each class
        overall_metrics: Overall accuracy, macro avg, weighted avg
        top_features_per_class: Top discriminative features for each class
        sample_predictions: Sample predictions for visualization
        visualization_data: Data formatted for visualization
        execution_time_ms: Classification execution time in milliseconds
        parameters_used: Actual parameters used for classification
        model_info: Information about the model used
        error: Error message if classification failed
    """

    success: bool
    predictions: List[TextPrediction] = Field(default_factory=list)
    confusion_matrix: List[List[int]] = Field(default_factory=list)
    class_names: List[str] = Field(default_factory=list)
    class_metrics: List[ClassMetrics] = Field(default_factory=list)
    overall_metrics: Dict[str, float] = Field(default_factory=dict)
    top_features_per_class: Dict[str, List[TopFeature]] = Field(default_factory=dict)
    sample_predictions: List[TextPrediction] = Field(default_factory=list)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
