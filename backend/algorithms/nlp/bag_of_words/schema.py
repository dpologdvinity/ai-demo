"""Request and response schemas for Bag of Words algorithm."""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class BagOfWordsRequest(BaseModel):
    """Parameters for Bag of Words vectorization.

    Attributes:
        max_features: Maximum vocabulary size (default: 100)
        ngram_range: N-gram range as tuple (min_n, max_n)
        min_df: Minimum document frequency
        max_df: Maximum document frequency as proportion
        binary: Use binary counts instead of term frequencies
        custom_documents: Optional custom document corpus
    """

    max_features: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum vocabulary size"
    )
    ngram_range: Tuple[int, int] = Field(
        default=(1, 1),
        description="N-gram range (min_n, max_n)"
    )
    min_df: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Minimum document frequency"
    )
    max_df: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Maximum document frequency as proportion"
    )
    binary: bool = Field(
        default=False,
        description="Use binary counts (presence/absence) instead of frequencies"
    )
    custom_documents: Optional[List[str]] = Field(
        default=None,
        description="Custom document corpus (optional)"
    )

    @field_validator('ngram_range')
    @classmethod
    def validate_ngram_range(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Validate that ngram_range is valid."""
        if v not in [(1, 1), (1, 2), (2, 2)]:
            raise ValueError("ngram_range must be (1,1), (1,2), or (2,2)")
        return v

    @field_validator('max_features')
    @classmethod
    def validate_max_features(cls, v: int) -> int:
        """Validate that max_features is within valid range."""
        if not 10 <= v <= 1000:
            raise ValueError("max_features must be between 10 and 1000")
        return v


class DocumentBow(BaseModel):
    """Bag of Words representation for a single document.

    Attributes:
        doc_id: Document identifier/index
        doc_preview: Preview of document text (first 100 chars)
        top_terms: List of top terms with their frequencies
        total_terms: Total number of terms in this document's BoW
        unique_terms: Number of unique terms in this document
    """
    doc_id: int
    doc_preview: str
    top_terms: List[Dict[str, Any]]
    total_terms: int
    unique_terms: int


class BagOfWordsResponse(BaseModel):
    """Response schema for Bag of Words results.

    Attributes:
        success: Whether computation completed successfully
        metrics: Metrics including vocabulary size, document count, sparsity
        vocabulary: List of vocabulary terms
        document_term_matrix: BoW matrix as list of lists (docs x terms)
        top_terms_per_doc: Top N terms for each document with frequencies
        most_frequent_terms: Most frequent terms across all documents
        least_frequent_terms: Least frequent terms in vocabulary
        document_previews: Preview text for each document
        heatmap_data: Data formatted for heatmap visualization
        word_frequency_data: Data for word frequency bar chart
        vocabulary_stats: Statistics about the vocabulary
        execution_time_ms: Execution time in milliseconds
        parameters_used: Actual parameters used
        model_info: Information about the CountVectorizer
        error: Error message if computation failed
    """

    success: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    vocabulary: List[str] = Field(default_factory=list)
    document_term_matrix: List[List[float]] = Field(default_factory=list)
    top_terms_per_doc: List[DocumentBow] = Field(default_factory=list)
    most_frequent_terms: List[Dict[str, Any]] = Field(default_factory=list)
    least_frequent_terms: List[Dict[str, Any]] = Field(default_factory=list)
    document_previews: List[str] = Field(default_factory=list)
    heatmap_data: Dict[str, Any] = Field(default_factory=dict)
    word_frequency_data: Dict[str, Any] = Field(default_factory=dict)
    vocabulary_stats: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
