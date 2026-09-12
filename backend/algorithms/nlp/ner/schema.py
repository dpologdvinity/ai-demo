"""Request and response schemas for Named Entity Recognition algorithm."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class NERParameters(BaseModel):
    """Parameters for Named Entity Recognition.

    Attributes:
        model_name: spaCy model to use (default: 'en_core_web_sm')
        entity_types: Types of entities to extract (default: all types)
        confidence_threshold: Minimum confidence score (default: 0.0)
        text_index: Sample text selector (default: 0)
        merge_entities: Merge adjacent entities (default: True)
        use_custom_text: Whether to use custom text input (default: False)
        custom_text: Custom text for NER (optional)
    """

    model_name: str = Field(
        default='en_core_web_sm',
        description="spaCy model for NER"
    )
    entity_types: List[str] = Field(
        default=['PERSON', 'ORG', 'GPE', 'DATE'],
        description="Entity types to extract"
    )
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for entities"
    )
    text_index: int = Field(
        default=0,
        ge=0,
        le=29,
        description="Sample text selector (0-29)"
    )
    merge_entities: bool = Field(
        default=True,
        description="Merge adjacent entities of same type"
    )
    use_custom_text: bool = Field(
        default=False,
        description="Whether to use custom text"
    )
    custom_text: Optional[str] = Field(
        default=None,
        description="Custom text for NER"
    )

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate that model_name is a supported spaCy model."""
        supported_models = ['en_core_web_sm', 'en_core_web_md']
        if v not in supported_models:
            raise ValueError(f"model_name must be one of {supported_models}")
        return v

    @field_validator('confidence_threshold')
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Validate that confidence_threshold is within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v


class Entity(BaseModel):
    """Individual extracted entity.

    Attributes:
        text: The entity text
        label: Entity type (PERSON, ORG, GPE, etc.)
        start: Start position in text
        end: End position in text
        confidence: Confidence score (0-1)
    """
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


class AnnotatedSpan(BaseModel):
    """Text span with optional entity annotation.

    Attributes:
        text: The text content
        label: Entity label if this is an entity, None otherwise
        start: Start position in original text
        end: End position in original text
    """
    text: str
    label: Optional[str] = None
    start: int
    end: int


class NERResponse(BaseModel):
    """Response schema for Named Entity Recognition results.

    Attributes:
        success: Whether NER completed successfully
        metrics: NER metrics including entity counts
        entities: List of extracted entities
        entity_distribution: Count of entities by type
        annotated_text: Text broken into spans with entity annotations
        visualization_data: Data formatted for visualization
        execution_time_ms: NER execution time in milliseconds
        parameters_used: Actual parameters used for NER
        model_info: Information about the spaCy model used
        error: Error message if NER failed
    """

    success: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    entities: List[Entity] = Field(default_factory=list)
    entity_distribution: Dict[str, int] = Field(default_factory=dict)
    annotated_text: List[AnnotatedSpan] = Field(default_factory=list)
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
