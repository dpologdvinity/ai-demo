"""Request and response schemas for Word2Vec algorithm."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class Word2VecParameters(BaseModel):
    """Parameters for Word2Vec training.

    Attributes:
        vector_size: Dimensionality of word embeddings (default: 100)
        window: Context window size (default: 5)
        min_count: Minimum word frequency threshold (default: 5)
        sg: Training algorithm - 0=CBOW, 1=Skip-gram (default: 0)
        epochs: Number of training iterations (default: 10)
        use_custom_corpus: Whether to use custom corpus text (default: False)
        custom_corpus: Custom text corpus for training (optional)
    """

    vector_size: int = Field(
        default=100,
        ge=50,
        le=300,
        description="Dimensionality of word embeddings"
    )
    window: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Context window size"
    )
    min_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Minimum word frequency threshold"
    )
    sg: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Training algorithm: 0=CBOW, 1=Skip-gram"
    )
    epochs: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Number of training iterations"
    )
    use_custom_corpus: bool = Field(
        default=False,
        description="Whether to use custom corpus text"
    )
    custom_corpus: Optional[str] = Field(
        default=None,
        description="Custom text corpus for training"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    @field_validator('vector_size')
    @classmethod
    def validate_vector_size(cls, v: int) -> int:
        """Validate that vector_size is within valid range."""
        if not 50 <= v <= 300:
            raise ValueError("vector_size must be between 50 and 300")
        return v

    @field_validator('window')
    @classmethod
    def validate_window(cls, v: int) -> int:
        """Validate that window is within valid range."""
        if not 2 <= v <= 10:
            raise ValueError("window must be between 2 and 10")
        return v


class SimilarWord(BaseModel):
    """Similar word with similarity score.

    Attributes:
        word: The similar word
        similarity: Cosine similarity score (0-1)
    """
    word: str
    similarity: float


class AnalogyResult(BaseModel):
    """Word analogy result.

    Attributes:
        query: The analogy query (e.g., "king - man + woman")
        result: The resulting word
        similarity: Confidence score
    """
    query: str
    result: str
    similarity: float


class Word2VecResponse(BaseModel):
    """Response schema for Word2Vec training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Training metrics including vocabulary size
        embeddings_2d: 2D projections of word embeddings for visualization
        similar_words: Dictionary of words to their similar words
        analogies: Word analogy examples
        vocabulary_sample: Sample of vocabulary words
        visualization_data: Data formatted for t-SNE/PCA visualization
        execution_time_ms: Training execution time in milliseconds
        parameters_used: Actual parameters used for training
        model_info: Information about the trained model
        error: Error message if training failed
    """

    success: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    embeddings_2d: List[Dict[str, Any]] = Field(default_factory=list)
    similar_words: Dict[str, List[SimilarWord]] = Field(default_factory=dict)
    analogies: List[AnalogyResult] = Field(default_factory=list)
    vocabulary_sample: List[str] = Field(default_factory=list)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
