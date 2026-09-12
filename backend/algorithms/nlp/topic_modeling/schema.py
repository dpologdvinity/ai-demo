"""Request and response schemas for Topic Modeling (LDA) algorithm."""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator


class TopicModelingParameters(BaseModel):
    """Parameters for Topic Modeling (LDA).

    Attributes:
        n_topics: Number of topics to discover
        max_iterations: Maximum LDA iterations
        alpha: Document-topic density parameter
        beta: Topic-word density parameter
        min_df: Minimum document frequency for vocabulary
        max_df: Maximum document frequency for vocabulary
        use_custom_documents: Whether to use custom documents
        custom_documents: Custom document corpus for analysis
    """

    n_topics: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of topics to discover"
    )
    max_iterations: int = Field(
        default=100,
        ge=20,
        le=500,
        description="Maximum LDA iterations"
    )
    alpha: Union[str, float] = Field(
        default="auto",
        description="Document-topic density ('auto' or numeric value)"
    )
    beta: Union[str, float] = Field(
        default="auto",
        description="Topic-word density ('auto' or numeric value)"
    )
    min_df: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum document frequency"
    )
    max_df: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Maximum document frequency"
    )
    use_custom_documents: bool = Field(
        default=False,
        description="Whether to use custom documents"
    )
    custom_documents: Optional[List[str]] = Field(
        default=None,
        description="Custom document corpus"
    )

    @field_validator('alpha')
    @classmethod
    def validate_alpha(cls, v: Union[str, float]) -> Union[str, float]:
        """Validate alpha parameter."""
        if isinstance(v, str):
            if v not in ['auto']:
                raise ValueError("alpha must be 'auto' or a numeric value")
        elif isinstance(v, (int, float)):
            if v <= 0:
                raise ValueError("alpha must be positive")
        return v

    @field_validator('beta')
    @classmethod
    def validate_beta(cls, v: Union[str, float]) -> Union[str, float]:
        """Validate beta parameter."""
        if isinstance(v, str):
            if v not in ['auto']:
                raise ValueError("beta must be 'auto' or a numeric value")
        elif isinstance(v, (int, float)):
            if v <= 0:
                raise ValueError("beta must be positive")
        return v


class TopicWord(BaseModel):
    """Word in a topic with its weight.

    Attributes:
        word: The word
        weight: Weight/probability of the word in the topic
    """
    word: str
    weight: float


class Topic(BaseModel):
    """Topic with its top words.

    Attributes:
        topic_id: Topic identifier
        top_words: List of top words with weights
        keywords: Comma-separated keyword string
    """
    topic_id: int
    top_words: List[TopicWord]
    keywords: str


class DocumentTopic(BaseModel):
    """Document with its topic distribution.

    Attributes:
        document_id: Document identifier
        document_preview: Preview of document text
        dominant_topic: The most prominent topic
        topic_distribution: Distribution across all topics
    """
    document_id: int
    document_preview: str
    dominant_topic: int
    topic_distribution: List[float]


class TopicModelingResponse(BaseModel):
    """Response from Topic Modeling (LDA).

    Attributes:
        success: Whether the analysis was successful
        topics: List of discovered topics with keywords
        document_topics: Document-topic distributions
        topic_word_matrix: Topic-word distribution matrix for heatmap
        vocabulary: List of vocabulary words
        coherence_score: Topic coherence score
        perplexity: Model perplexity (lower is better)
        metrics: Additional metrics
        visualization_data: Data for frontend visualizations
        execution_time_ms: Execution time in milliseconds
        error: Error message if analysis failed
    """
    success: bool
    topics: List[Topic] = Field(default_factory=list)
    document_topics: List[DocumentTopic] = Field(default_factory=list)
    topic_word_matrix: List[List[float]] = Field(default_factory=list)
    vocabulary: List[str] = Field(default_factory=list)
    coherence_score: Optional[float] = None
    perplexity: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = 0.0
    error: Optional[str] = None
