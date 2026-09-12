"""Request and response schemas for Tokenization algorithm."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class TokenizationRequest(BaseModel):
    """Parameters for text tokenization.

    Attributes:
        tokenizer_type: Type of tokenizer to use
        lowercase: Convert text to lowercase before tokenization
        remove_punctuation: Remove punctuation from tokens
        remove_stopwords: Remove common stop words
        max_tokens: Maximum number of tokens to return (limit for display)
        text_index: Index of sample text to use (0-19)
        compare_mode: Enable comparison of multiple tokenizers
        custom_text: Optional custom text to tokenize
    """

    tokenizer_type: str = Field(
        default='word',
        description="Type of tokenizer: 'whitespace', 'word', 'sentence', 'wordpiece', 'bpe', 'character', 'spacy'"
    )
    lowercase: bool = Field(
        default=True,
        description="Convert text to lowercase before tokenization"
    )
    remove_punctuation: bool = Field(
        default=False,
        description="Remove punctuation from tokens"
    )
    remove_stopwords: bool = Field(
        default=False,
        description="Remove stop words from tokens"
    )
    max_tokens: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of tokens to return"
    )
    text_index: int = Field(
        default=0,
        ge=0,
        le=19,
        description="Index of sample text to use (0-19)"
    )
    compare_mode: bool = Field(
        default=False,
        description="Enable comparison mode (all tokenizers)"
    )
    custom_text: Optional[str] = Field(
        default=None,
        description="Custom text to tokenize (optional)"
    )

    @field_validator('tokenizer_type')
    @classmethod
    def validate_tokenizer_type(cls, v: str) -> str:
        """Validate tokenizer type."""
        valid_types = ['whitespace', 'word', 'sentence', 'wordpiece', 'bpe', 'character', 'spacy']
        if v not in valid_types:
            raise ValueError(f"tokenizer_type must be one of {valid_types}")
        return v

    @field_validator('text_index')
    @classmethod
    def validate_text_index(cls, v: int) -> int:
        """Validate text index range."""
        if not 0 <= v <= 19:
            raise ValueError("text_index must be between 0 and 19")
        return v


class TokenInfo(BaseModel):
    """Information about a single token.

    Attributes:
        text: The token text
        index: Position in the token sequence
        start_char: Starting character position in original text
        end_char: Ending character position in original text
        is_stopword: Whether the token is a stopword (if applicable)
        pos_tag: Part-of-speech tag (for spaCy tokenizer)
    """
    text: str
    index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    is_stopword: Optional[bool] = None
    pos_tag: Optional[str] = None


class TokenizerResult(BaseModel):
    """Results from a single tokenizer.

    Attributes:
        tokenizer_name: Name of the tokenizer used
        tokens: List of token information
        token_count: Total number of tokens
        unique_token_count: Number of unique tokens
        avg_token_length: Average token length
        vocabulary_size: Size of vocabulary (unique tokens)
        token_frequency: Frequency distribution of tokens (top 20)
    """
    tokenizer_name: str
    tokens: List[TokenInfo]
    token_count: int
    unique_token_count: int
    avg_token_length: float
    vocabulary_size: int
    token_frequency: List[Dict[str, Any]]


class TokenizationResponse(BaseModel):
    """Response schema for tokenization results.

    Attributes:
        success: Whether tokenization completed successfully
        original_text: The original input text
        text_preview: Preview of the original text (first 200 chars)
        main_result: Primary tokenization result (single tokenizer mode)
        comparison_results: Results from all tokenizers (compare mode)
        statistics: Overall statistics
        frequency_distribution: Token frequency data for visualization
        execution_time_ms: Execution time in milliseconds
        parameters_used: Actual parameters used
        error: Error message if tokenization failed
    """

    success: bool
    original_text: str = ""
    text_preview: str = ""
    main_result: Optional[TokenizerResult] = None
    comparison_results: List[TokenizerResult] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    frequency_distribution: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
