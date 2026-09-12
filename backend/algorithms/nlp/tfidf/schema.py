"""Request and response schemas for TF-IDF algorithm."""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class TFIDFRequest(BaseModel):
    """Parameters for TF-IDF vectorization.

    Attributes:
        max_features: Maximum number of features/terms to extract
        ngram_range: N-gram range as tuple (min_n, max_n)
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        use_idf: Whether to use inverse document frequency weighting
        custom_documents: Optional custom document corpus
    """

    max_features: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of features to extract"
    )
    ngram_range: Tuple[int, int] = Field(
        default=(1, 1),
        description="N-gram range (min_n, max_n)"
    )
    min_df: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum document frequency"
    )
    max_df: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Maximum document frequency as proportion"
    )
    use_idf: bool = Field(
        default=True,
        description="Use inverse document frequency weighting"
    )
    custom_documents: Optional[List[str]] = Field(
        default=None,
        description="Custom document corpus (optional)"
    )
    normalize: bool = Field(
        default=True,
        description="Normalize TF-IDF vectors to unit length"
    )

    @field_validator('ngram_range')
    @classmethod
    def validate_ngram_range(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Validate that ngram_range is valid."""
        if v not in [(1, 1), (1, 2), (1, 3)]:
            raise ValueError("ngram_range must be (1,1), (1,2), or (1,3)")
        return v

    @field_validator('max_features')
    @classmethod
    def validate_max_features(cls, v: int) -> int:
        """Validate that max_features is within valid range."""
        if not 10 <= v <= 500:
            raise ValueError("max_features must be between 10 and 500")
        return v


class DocumentTerms(BaseModel):
    """Top terms for a single document.

    Attributes:
        doc_id: Document identifier/index
        doc_preview: Preview of document text
        top_terms: List of (term, score) tuples
    """
    doc_id: int
    doc_preview: str
    top_terms: List[Dict[str, Any]]


class TFIDFResponse(BaseModel):
    """Response schema for TF-IDF results.

    Attributes:
        success: Whether computation completed successfully
        metrics: Metrics including vocabulary size, document count
        feature_names: List of extracted feature names
        tfidf_matrix: TF-IDF matrix as list of lists (docs x terms)
        top_terms_per_doc: Top N terms for each document
        top_terms_global: Most important terms across all documents
        document_previews: Preview text for each document
        heatmap_data: Data formatted for heatmap visualization
        visualization_data: Additional visualization data
        execution_time_ms: Execution time in milliseconds
        parameters_used: Actual parameters used
        model_info: Information about the vectorizer
        error: Error message if computation failed
    """

    success: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    feature_names: List[str] = Field(default_factory=list)
    tfidf_matrix: List[List[float]] = Field(default_factory=list)
    top_terms_per_doc: List[DocumentTerms] = Field(default_factory=list)
    top_terms_global: List[Dict[str, Any]] = Field(default_factory=list)
    document_previews: List[str] = Field(default_factory=list)
    heatmap_data: Dict[str, Any] = Field(default_factory=dict)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
