"""Request and response schemas for Part-of-Speech Tagging algorithm."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class POSTaggingParameters(BaseModel):
    """Parameters for Part-of-Speech Tagging.

    Attributes:
        tagger: Tagger type to use (default: 'spacy')
        text_index: Sample text selector (default: 0)
        show_fine_grained: Show detailed Penn Treebank tags (default: True)
        show_dependencies: Include dependency parsing (default: True)
        tag_scheme: Tag scheme to use (default: 'penn')
        use_custom_text: Whether to use custom text input (default: False)
        custom_text: Custom text for POS tagging (optional)
    """

    tagger: str = Field(
        default='spacy',
        description="Tagger type (spacy, nltk, universal)"
    )
    text_index: int = Field(
        default=0,
        ge=0,
        le=24,
        description="Sample text index (0-24)"
    )
    show_fine_grained: bool = Field(
        default=True,
        description="Show fine-grained POS tags"
    )
    show_dependencies: bool = Field(
        default=True,
        description="Include dependency parsing"
    )
    tag_scheme: str = Field(
        default='penn',
        description="Tag scheme (penn or universal)"
    )
    use_custom_text: bool = Field(
        default=False,
        description="Whether to use custom text"
    )
    custom_text: Optional[str] = Field(
        default=None,
        description="Custom text for POS tagging"
    )

    @field_validator('tagger')
    @classmethod
    def validate_tagger(cls, v: str) -> str:
        """Validate that tagger is supported."""
        supported_taggers = ['spacy', 'nltk', 'universal']
        if v not in supported_taggers:
            raise ValueError(f"tagger must be one of {supported_taggers}")
        return v

    @field_validator('text_index')
    @classmethod
    def validate_text_index(cls, v: int) -> int:
        """Validate that text_index is within valid range."""
        if not 0 <= v <= 24:
            raise ValueError("text_index must be between 0 and 24")
        return v

    @field_validator('tag_scheme')
    @classmethod
    def validate_tag_scheme(cls, v: str) -> str:
        """Validate that tag_scheme is supported."""
        supported_schemes = ['penn', 'universal']
        if v not in supported_schemes:
            raise ValueError(f"tag_scheme must be one of {supported_schemes}")
        return v


class TaggedWord(BaseModel):
    """Individual tagged word with POS information.

    Attributes:
        text: The word text
        pos_coarse: Coarse POS tag (Universal)
        pos_fine: Fine-grained POS tag (Penn Treebank)
        tag: Primary tag to display
        description: Human-readable description
        index: Token index in sentence
        lemma: Lemmatized form of the word
        is_stop: Whether it's a stop word
        dependency: Dependency relation (optional)
        head_text: Head word text (optional)
        head_index: Head word index (optional)
    """
    text: str
    pos_coarse: str
    pos_fine: str
    tag: str
    description: str
    index: int
    lemma: str
    is_stop: bool
    dependency: Optional[str] = None
    head_text: Optional[str] = None
    head_index: Optional[int] = None


class DependencyEdge(BaseModel):
    """Dependency parsing edge for visualization.

    Attributes:
        source: Source token index
        target: Target token index
        label: Dependency relation label
        source_text: Source word text
        target_text: Target word text
    """
    source: int
    target: int
    label: str
    source_text: str
    target_text: str


class POSTaggingResponse(BaseModel):
    """Response schema for Part-of-Speech Tagging results.

    Attributes:
        success: Whether POS tagging completed successfully
        metrics: POS tagging metrics
        tagged_words: List of tagged words with POS information
        pos_distribution: Count of POS tags
        tag_frequencies: Top tag frequencies
        dependency_edges: Dependency parsing edges
        visualization_data: Data formatted for visualization
        execution_time_ms: Execution time in milliseconds
        parameters_used: Actual parameters used for tagging
        model_info: Information about the tagger used
        text_analyzed: The text that was analyzed
        error: Error message if tagging failed
    """

    success: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    tagged_words: List[TaggedWord] = Field(default_factory=list)
    pos_distribution: Dict[str, int] = Field(default_factory=dict)
    tag_frequencies: List[Dict[str, Any]] = Field(default_factory=list)
    dependency_edges: List[DependencyEdge] = Field(default_factory=list)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    text_analyzed: str = ""
    error: Optional[str] = None
