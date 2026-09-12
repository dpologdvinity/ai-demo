"""Request and response schemas for GloVe algorithm."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class GloVeParameters(BaseModel):
    """Parameters for GloVe word embeddings.

    Attributes:
        embedding_dim: Vector dimension (default: 100)
        top_k: Similar words to show (default: 10)
        query_word: Word to find similar words (default: 'king')
        analogy_word_a: First word in analogy A - B + C (default: 'king')
        analogy_word_b: Second word in analogy A - B + C (default: 'man')
        analogy_word_c: Third word in analogy A - B + C (default: 'woman')
    """

    embedding_dim: int = Field(
        default=100,
        description="Vector dimension for embeddings"
    )
    top_k: int = Field(
        default=10,
        ge=5,
        le=20,
        description="Number of similar words to show"
    )
    query_word: str = Field(
        default="king",
        description="Word to find similar words for"
    )
    analogy_word_a: str = Field(
        default="king",
        description="First word in analogy (A - B + C)"
    )
    analogy_word_b: str = Field(
        default="man",
        description="Second word in analogy (A - B + C)"
    )
    analogy_word_c: str = Field(
        default="woman",
        description="Third word in analogy (A - B + C)"
    )

    @field_validator('embedding_dim')
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        """Validate that embedding_dim is valid."""
        valid_dims = [50, 100, 200, 300]
        if v not in valid_dims:
            raise ValueError(f"embedding_dim must be one of {valid_dims}")
        return v

    @field_validator('query_word', 'analogy_word_a', 'analogy_word_b', 'analogy_word_c')
    @classmethod
    def validate_word(cls, v: str) -> str:
        """Validate that word is not empty."""
        if not v or not v.strip():
            raise ValueError("Word cannot be empty")
        return v.strip().lower()


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
        result_word: The resulting word
        similarity: Confidence score
        top_results: Top K results for the analogy
    """
    query: str
    result_word: str
    similarity: float
    top_results: List[SimilarWord] = Field(default_factory=list)


class WordVector(BaseModel):
    """Word vector representation.

    Attributes:
        word: The word
        vector: Vector representation
    """
    word: str
    vector: List[float]


class GloVeResponse(BaseModel):
    """Response schema for GloVe embeddings.

    Attributes:
        success: Whether operation completed successfully
        metrics: Statistics about embeddings
        embeddings_2d: 2D projections of word embeddings for visualization
        similar_words: Similar words for query word
        analogy_result: Word analogy result
        cosine_similarity_matrix: Cosine similarity heatmap data
        vocabulary_sample: Sample of vocabulary words
        visualization_data: Data formatted for t-SNE visualization
        execution_time_ms: Execution time in milliseconds
        parameters_used: Actual parameters used
        model_info: Information about the embeddings
        error: Error message if operation failed
    """

    success: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    embeddings_2d: List[Dict[str, Any]] = Field(default_factory=list)
    similar_words: List[SimilarWord] = Field(default_factory=list)
    analogy_result: Optional[AnalogyResult] = None
    cosine_similarity_matrix: Dict[str, Any] = Field(default_factory=dict)
    vocabulary_sample: List[str] = Field(default_factory=list)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
